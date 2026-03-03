"""
PyMC BART Wrapper for Multiclass Categorical and Ordinal Classification.

This module provides a wrapper class around PyMC's BART implementation
    for multiclass categorical or ordinal outcomes.
Two target types are supported via the target_type parameter:
    * "categorical": unordered multiclass outcome.
    Uses pmb.BART → softmax → pm.Categorical.
    * "ordinal": ordered integer / ordinal outcome.
    Uses pmb.BART (1-D latent) → pm.OrderedLogistic with learned cutpoints.
It handles data preprocessing (missing value imputation or removal,
    one-hot encoding of categorical variables) and follows the model
    specification from the official PyMC-BART categorical and ordinal regression example:
    https://www.pymc.io/projects/bart/en/latest/examples/bart_categorical_hawks.html
    https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/11-Ordered_Categories.html
"""

import warnings
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from sklearn.preprocessing import OneHotEncoder


# Constant used to fill missing numeric values outside typical range
# Numeric columns in our dataset are positive so -99 would be a safe out-of-range constant
MISSING_NUMERIC_FILL = -99


class BARTModelWrapper:
    """
    Wrapper for fitting a PyMC BART model on categorical or ordinal data.

    Parameters
    ----------
    target_var : str
        Name of the target (outcome) column.
    predictor_vars : list[str]
        Names of the predictor columns to use in the model.
    non_numeric_vars : list[str] or None
        Names of the predictor columns that are non-numeric (categorical).
        These will be one-hot encoded before modelling.  
        If None, non-numeric predictors are dropped from the model because
            PyMC-BART does not natively handle categorical features.
    target_type : {"categorical", "ordinal"}
        "categorical": unordered multiclass (softmax + Categorical).
        "ordinal": ordered integer outcome (OrderedLogistic with
            learned cutpoints).  
        Default is "categorical".
    ordinal_order : list[str] or None
        When target_type="ordinal" and the raw target is string-valued,
            supply the desired low-to-high ordering of class labels.  If
        None and the target is already integer, natural ordering is
            used; if string-valued, alphabetical ordering is used.
    fill_missing : bool
        If True, fill missing values:
        - Numeric columns → filled with missing_numeric_fill (default -99).
        - Categorical columns → filled with a new "missing" category.
        If False, rows containing any missing value (across target
            and selected predictors) are dropped.
    missing_numeric_fill : float or int
        Value used to impute missing numeric predictors when fill_missing=True.
        Defaults to the module-level constant MISSING_NUMERIC_FILL (-99).

    Notes
    -----
    If the full dataset is available before splitting into training and
        test sets, call register_data(df) first so that the one-hot
        encoder and target encoding are learned from all categories.  
    If register_data is never called, encoders are learned from the
        training set only and unknown test-set categories are encoded as all-zeros.
    """

    _VALID_TARGET_TYPES = ("categorical", "ordinal")

    def __init__(
        self,
        target_var: str,
        predictor_vars: list[str],
        non_numeric_vars: Optional[list[str]] = None,
        target_type: str = "categorical",
        ordinal_order: Optional[list[str]] = None,
        fill_missing: bool = True,
        missing_numeric_fill: float | int = MISSING_NUMERIC_FILL,
    ):
        if target_type not in self._VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {self._VALID_TARGET_TYPES}, got '{target_type}'"
            )
        self.target_var = target_var
        self.predictor_vars = list(predictor_vars)
        self.non_numeric_vars = list(non_numeric_vars) if non_numeric_vars is not None else None
        self.target_type = target_type
        self.ordinal_order = list(ordinal_order) if ordinal_order is not None else None
        self.fill_missing = fill_missing
        self.missing_numeric_fill = missing_numeric_fill

        # Populated during fit / preprocess --------------------------------
        self.model_ = None
        self.idata_ = None
        self.category_codes_ = None   # pd.Index of target class labels
        self.category_map_ = None     # dict: code → label
        self.n_classes_ = None        # number of target classes
        self.ohe_encoder_ = None      # fitted sklearn OneHotEncoder
        self.ohe_columns_ = None      # list of columns after one-hot encoding
        self.numeric_vars_ = None     # numeric predictor columns kept
        self.X_shared_ = None         # pm.Data reference for out-of-sample prediction
        self.encoder_fitted_ = False  # True after register_data() pre-fits encoders
        self.fitted_ = False

    # ------------------------------------------------------------------
    # Data registration (optional)
    # ------------------------------------------------------------------
    def register_data(self, df: pd.DataFrame) -> "BARTModelWrapper":
        """
        Pre-fit encoders on the full dataset before train / test split.

        Call this before fit() so that the one-hot encoder and
            target-variable encoding are learned from all available data.
        This guarantees that every category present in the full dataset
            is represented in the encoding, and avoids the all-zeros
            fallback for categories that appear only in the test set.
        If this method is never called, fit() and predict(),
            encoders are learned from the training set only, 
            and unknown test-set categories are encoded as all-zeros.

        Parameters
        ----------
        df : pd.DataFrame
            The **complete** dataset (before splitting into training
            and test sets).  Must contain the target column and all
            predictor columns.

        Returns
        -------
        self
            The wrapper instance (for method chaining).
        """
        df = df.copy()

        # ----- Determine predictor variable types -------------------------
        if self.non_numeric_vars is not None:
            numeric_vars = [v for v in self.predictor_vars if v not in self.non_numeric_vars]
            cat_vars = [v for v in self.non_numeric_vars if v in self.predictor_vars]
        else:
            numeric_vars = [v for v in self.predictor_vars if pd.api.types.is_numeric_dtype(df[v])]
            cat_vars = []
            excluded = set(self.predictor_vars) - set(numeric_vars)
            if excluded:
                warnings.warn(
                    f"Non-numeric predictors excluded (no non_numeric_vars list provided): {excluded}"
                )

        self.numeric_vars_ = numeric_vars

        # ----- Handle missing values for encoding -------------------------
        if self.fill_missing:
            for col in cat_vars:
                df[col] = df[col].fillna("missing").astype(str)
        else:
            cols_needed = numeric_vars + cat_vars + [self.target_var]
            df = df.dropna(subset=cols_needed).reset_index(drop=True)

        # ----- Learn target encoding from full data -----------------------
        if self.target_type == "ordinal" and self.ordinal_order is not None:
            self.category_codes_ = pd.Index(self.ordinal_order)
        elif self.target_type == "ordinal" and pd.api.types.is_integer_dtype(df[self.target_var]):
            sorted_vals = sorted(df[self.target_var].dropna().unique())
            self.category_codes_ = pd.Index(sorted_vals)
        else:
            _, self.category_codes_ = pd.factorize(df[self.target_var], sort=True)

        self.category_map_ = {i: label for i, label in enumerate(self.category_codes_)}
        self.n_classes_ = len(self.category_codes_)

        # ----- Fit OHE on full data ---------------------------------------
        if cat_vars:
            self.ohe_encoder_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.float64,
            )
            self.ohe_encoder_.fit(df[cat_vars])
            self.ohe_columns_ = list(self.ohe_encoder_.get_feature_names_out(cat_vars))
        else:
            self.ohe_encoder_ = None
            self.ohe_columns_ = []

        self.encoder_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame, *, fit: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
        """
        Prepare the data for the BART model.

        When register_data() has been called first, the one-hot
            encoder and target encoding learned from 
            the full dataset are reused here.  
        Otherwise, encoders are fitted from scratch on the training data (fit=True) 
            or reused from the training run (fit=False).

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.
        fit : bool
            If True (training), learn and store encoding metadata and
                return the encoded target array.  
            If False (prediction), reuse stored metadata and 
                return None for the target.

        Returns
        -------
        X : pd.DataFrame
            Processed predictor matrix (all numeric, no missing values).
        y_codes : np.ndarray or None
            Integer-coded target variable (only when fit=True).
        """
        df = df.copy()

        # ----- Determine which predictors to keep -------------------------
        if self.non_numeric_vars is not None:
            numeric_vars = [v for v in self.predictor_vars if v not in self.non_numeric_vars]
            cat_vars = [v for v in self.non_numeric_vars if v in self.predictor_vars]
        else:
            # Drop any non-numeric predictor columns
            numeric_vars = [v for v in self.predictor_vars if pd.api.types.is_numeric_dtype(df[v])]
            cat_vars = []
            excluded = set(self.predictor_vars) - set(numeric_vars)
            if excluded:
                warnings.warn(
                    f"Non-numeric predictors excluded (no non_numeric_vars list provided): {excluded}"
                )

        # ----- Handle missing values --------------------------------------
        if self.fill_missing:
            # Numeric columns: fill NaN with a constant outside data range
            for col in numeric_vars:
                df[col] = df[col].fillna(self.missing_numeric_fill)
            # Categorical columns: fill NaN with a new "missing" level
            for col in cat_vars:
                df[col] = df[col].fillna("missing").astype(str)
        else:
            # Drop rows with any NaN in target + selected predictors
            cols_needed = numeric_vars + cat_vars
            if fit:
                cols_needed = cols_needed + [self.target_var]
            df = df.dropna(subset=cols_needed).reset_index(drop=True)

        # ----- Encode target variable -------------------------------------
        y_codes = None
        if fit:
            if self.encoder_fitted_:
                # Encoders were pre-fitted via register_data(); 
                #   reuse the stored category_codes_ to map target values to integer codes
                cat_target = pd.Categorical(
                    df[self.target_var],
                    categories=self.category_codes_,
                    ordered=(self.target_type == "ordinal"),
                )
                y_codes = cat_target.codes.astype(np.int64)
            else:
                # Learn encodings from this (training) data
                if self.target_type == "ordinal" and self.ordinal_order is not None:
                    ordered_cat = pd.Categorical(
                        df[self.target_var],
                        categories=self.ordinal_order,
                        ordered=True,
                    )
                    y_codes = ordered_cat.codes.astype(np.int64)
                    self.category_codes_ = pd.Index(self.ordinal_order)
                elif self.target_type == "ordinal" and pd.api.types.is_integer_dtype(df[self.target_var]):
                    sorted_vals = sorted(df[self.target_var].dropna().unique())
                    mapping = {v: i for i, v in enumerate(sorted_vals)}
                    y_codes = df[self.target_var].map(mapping).astype(np.int64).values
                    self.category_codes_ = pd.Index(sorted_vals)
                else:
                    cat_target = pd.Categorical(df[self.target_var])
                    y_codes = cat_target.codes.astype(np.int64)
                    _, self.category_codes_ = pd.factorize(df[self.target_var], sort=True)

                self.category_map_ = {i: label for i, label in enumerate(self.category_codes_)}
                self.n_classes_ = len(self.category_codes_)

        # ----- One-hot encode categorical predictors ----------------------
        if cat_vars:
            if fit and not self.encoder_fitted_:
                # Fit the encoder on training data; unknown categories at
                #   prediction time will be encoded as all-zeros.
                self.ohe_encoder_ = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    dtype=np.float64,
                )
                encoded = self.ohe_encoder_.fit_transform(df[cat_vars])
                self.ohe_columns_ = list(self.ohe_encoder_.get_feature_names_out(cat_vars))
            else:
                # Reuse the encoder fitted during register_data() or training
                encoded = self.ohe_encoder_.transform(df[cat_vars])
            df_cat = pd.DataFrame(encoded, columns=self.ohe_columns_, index=df.index)
        else:
            df_cat = pd.DataFrame(index=df.index)
            if fit and not self.encoder_fitted_:
                self.ohe_encoder_ = None
                self.ohe_columns_ = []

        # ----- Assemble final predictor matrix ----------------------------
        X = pd.concat([df[numeric_vars].reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)

        if fit:
            self.numeric_vars_ = numeric_vars

        return X, y_codes

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        m: int = 50,
        chains: int = 4,
        cores: int = 1,
        draws: int = 1000,
        tune: int = 1000,
        separate_trees: bool = False,
        random_seed: int = 42,
        sample_posterior_predictive: bool = True,
        compute_convergence_checks: bool = False,
        **sample_kwargs,
    ):
        """
        Fit the BART model (categorical or ordinal).

        Parameters
        ----------
        df : pd.DataFrame
            Training data containing target and predictor columns.
        m : int
            Number of trees for the BART prior (default 50).
        chains : int
            Number of MCMC chains (default 4).
        cores : int
            Number of parallel cores for sampling (default 1).
            Use 1 for sequential sampling, which is more reliable in
                Jupyter notebooks.  
        draws : int
            Number of posterior draws per chain (default 1000).
        tune : int
            Number of tuning steps per chain (default 1000).
        separate_trees : bool
            If True, fit independent trees per category.  
            Only used when target_type="categorical".  
            Reduces posterior variance but increases computation time.
            See Fitting independent trees in the PyMC-BART docs for details:
            https://www.pymc.io/projects/bart/en/latest/examples/bart_categorical_hawks.html
        random_seed : int
            Random seed for reproducibility.
        sample_posterior_predictive : bool
            Whether to also sample from the posterior predictive.
        compute_convergence_checks : bool
            Whether to compute convergence diagnostics during sampling.
        **sample_kwargs
            Extra keyword arguments forwarded to pm.sample().

        Returns
        -------
        self
            The fitted wrapper instance (for method chaining).
        """
        X, y_codes = self.preprocess(df, fit=True)

        if self.target_type == "categorical":
            model, idata, X_shared = self._fit_categorical(
                X, y_codes, m=m, separate_trees=separate_trees,
                chains=chains, cores=cores, draws=draws, tune=tune,
                random_seed=random_seed,
                sample_posterior_predictive=sample_posterior_predictive,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )
        else:  # ordinal
            model, idata, X_shared = self._fit_ordinal(
                X, y_codes, m=m,
                chains=chains, cores=cores, draws=draws, tune=tune,
                random_seed=random_seed,
                sample_posterior_predictive=sample_posterior_predictive,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )

        self.model_ = model
        self.idata_ = idata
        self.X_shared_ = X_shared
        self.fitted_ = True

        return self

    # --- Private model builders -------------------------------------------

    def _fit_categorical(
        self, X, y_codes, *, m, separate_trees, chains, cores, draws, tune,
        random_seed, sample_posterior_predictive, compute_convergence_checks,
        **sample_kwargs,
    ):
        """Build and sample the multiclass categorical BART model."""
        n_obs = len(X)

        coords = {
            "n_obs": np.arange(n_obs),
            "classes": self.category_codes_,
        }

        with pm.Model(coords=coords) as model:
            X_shared = pm.Data("X", X.values)
            mu = pmb.BART(
                "mu",
                X_shared,
                y_codes,
                m=m,
                separate_trees=separate_trees,
                dims=["classes", "n_obs"],
            )
            theta = pm.Deterministic("theta", pm.math.softmax(mu, axis=0))
            # shape=mu.shape[1] makes the observation count dynamic so
            #   pm.sample_posterior_predictive works after X.set_value(X_test)
            y = pm.Categorical("y", p=theta.T, observed=y_codes, shape=mu.shape[1])

            idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                random_seed=random_seed,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )
            if sample_posterior_predictive:
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return model, idata, X_shared

    def _fit_ordinal(
        self, X, y_codes, *, m, chains, cores, draws, tune,
        random_seed, sample_posterior_predictive, compute_convergence_checks,
        **sample_kwargs,
    ):
        """Build and sample the ordinal BART model (OrderedLogistic)."""
        n_classes = self.n_classes_

        # Evenly spaced initial cutpoints so every class starts with
        # non-negligible probability, avoiding log(0) at initialisation.
        init_cutpoints = np.linspace(-2, 2, n_classes - 1)

        with pm.Model() as model:
            X_shared = pm.Data("X", X.values)

            # BART provides a 1-D latent score per observation
            mu = pmb.BART("mu", X_shared, y_codes, m=m)

            # Cutpoints that partition the latent scale into n_classes bins.
            # The ordered transform ensures cutpoints[0] < cutpoints[1] < ...
            cutpoints = pm.Normal(
                "cutpoints",
                mu=init_cutpoints,
                sigma=1.5,
                shape=n_classes - 1,
                transform=pm.distributions.transforms.univariate_ordered,
                initval=init_cutpoints,
            )

            # shape=mu.shape makes the observation count dynamic so
            # pm.sample_posterior_predictive works after X.set_value(X_test)
            y = pm.OrderedLogistic(
                "y", eta=mu, cutpoints=cutpoints, observed=y_codes,
                shape=mu.shape,
            )

            idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                random_seed=random_seed,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )
            if sample_posterior_predictive:
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return model, idata, X_shared

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        new_data: pd.DataFrame,
        random_seed: int = 42,
    ) -> dict:
        """
        Generate out-of-sample predictions for new data.

        Swap the shared covariate matrix via pm.Data.set_value, 
            then call pm.sample_posterior_predictive to draw from 
            the full generative model (BART trees + likelihood jointly).

        Parameters
        ----------
        new_data : pd.DataFrame
            New observations with the same predictor columns used in training.
        random_seed : int
            Random seed for reproducibility.

        Returns
        -------
        result : dict
            "posterior_predictive": array of shape
                (n_samples, n_obs) with integer class-code draws
                from the posterior predictive distribution.
            "predicted_classes": array of shape (n_obs,) with the
                most-likely class code (mode across posterior draws).
            "predicted_labels": list of str with the human-readable
                class labels corresponding to predicted_classes.
            "class_prob_mean": array of shape (n_obs, n_classes) with
                posterior-mean class probabilities per observation
                (empirical frequency across draws).
        """
        if not self.fitted_:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        X_new, _ = self.preprocess(new_data, fit=False)

        # Swap the shared covariate matrix with new data and draw
        # from the full posterior predictive (BART + likelihood).
        with self.model_:
            self.X_shared_.set_value(X_new.values)
            ppc = pm.sample_posterior_predictive(
                self.idata_, random_seed=random_seed,
            )

        # Extract predicted class codes: (chain, draw, n_obs)
        y_pred = ppc.posterior_predictive["y"].values
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])  # (n_samples, n_obs)

        n_samples, n_obs = y_pred_flat.shape
        n_classes = self.n_classes_

        # Compute empirical class probabilities from posterior draws
        class_prob_mean = np.zeros((n_obs, n_classes))
        for c in range(n_classes):
            class_prob_mean[:, c] = (y_pred_flat == c).mean(axis=0)

        # Point prediction: class with highest posterior predictive probability
        predicted_classes = class_prob_mean.argmax(axis=1)
        predicted_labels = [self.category_map_[c] for c in predicted_classes]

        return {
            "posterior_predictive": y_pred_flat,
            "predicted_classes": predicted_classes,
            "predicted_labels": predicted_labels,
            "class_prob_mean": class_prob_mean,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_inference_data(self) -> az.InferenceData:
        """Return the ArviZ InferenceData object from the fitted model."""
        if not self.fitted_:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self.idata_

    def get_model(self) -> pm.Model:
        """Return the underlying PyMC Model object."""
        if not self.fitted_:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self.model_

    def summary(self) -> pd.DataFrame:
        """Return an ArviZ summary of the posterior."""
        if not self.fitted_:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return az.summary(self.idata_)

    def __repr__(self) -> str:
        status = "fitted" if self.fitted_ else "not fitted"
        enc = "full-data" if self.encoder_fitted_ else "train-only"
        return (
            f"BARTModelWrapper(target='{self.target_var}', "
            f"target_type='{self.target_type}', "
            f"n_predictors={len(self.predictor_vars)}, "
            f"fill_missing={self.fill_missing}, "
            f"encoding={enc}, "
            f"status={status})"
        )
