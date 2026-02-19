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
    specification from the official PyMC-BART categorical regression example:
    https://www.pymc.io/projects/bart/en/latest/examples/bart_categorical_hawks.html
"""

import warnings
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from scipy.special import softmax


# Constant used to fill missing numeric values (outside typical positive range)
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
        - Numeric columns → filled with MISSING_NUMERIC_FILL (-99).
        - Categorical columns → filled with a new "missing" category.
        If False, rows containing any missing value (across target
            and selected predictors) are dropped.
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

        # Populated during fit / preprocess --------------------------------
        self.model_ = None
        self.idata_ = None
        self.category_codes_ = None   # pd.Index of target class labels
        self.category_map_ = None     # dict: code → label
        self.n_classes_ = None        # number of target classes
        self.ohe_columns_ = None      # list of columns after one-hot encoding
        self.numeric_vars_ = None     # numeric predictor columns kept
        self.fitted_ = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame, *, fit: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
        """
        Prepare the data for the BART model.

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
            dropped = set(self.predictor_vars) - set(numeric_vars)
            if dropped:
                warnings.warn(
                    f"Non-numeric predictors dropped (no non_numeric_vars list provided): {dropped}"
                )

        # ----- Handle missing values --------------------------------------
        if self.fill_missing:
            # Numeric columns: fill NaN with a constant outside data range
            for col in numeric_vars:
                df[col] = df[col].fillna(MISSING_NUMERIC_FILL)
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
            if self.target_type == "ordinal" and self.ordinal_order is not None:
                # Use the user-supplied ordering
                ordered_cat = pd.Categorical(
                    df[self.target_var],
                    categories=self.ordinal_order,
                    ordered=True,
                )
                y_codes = ordered_cat.codes.astype(np.int64)
                self.category_codes_ = pd.Index(self.ordinal_order)
            elif self.target_type == "ordinal" and pd.api.types.is_integer_dtype(df[self.target_var]):
                # Integer target – use natural ordering
                sorted_vals = sorted(df[self.target_var].dropna().unique())
                mapping = {v: i for i, v in enumerate(sorted_vals)}
                y_codes = df[self.target_var].map(mapping).astype(np.int64).values
                self.category_codes_ = pd.Index(sorted_vals)
            else:
                # Categorical (unordered) or ordinal with string target (alpha order)
                cat_target = pd.Categorical(df[self.target_var])
                y_codes = cat_target.codes.astype(np.int64)
                _, self.category_codes_ = pd.factorize(df[self.target_var], sort=True)

            self.category_map_ = {i: label for i, label in enumerate(self.category_codes_)}
            self.n_classes_ = len(self.category_codes_)

        # ----- One-hot encode categorical predictors ----------------------
        if cat_vars:
            df_cat = pd.get_dummies(df[cat_vars], columns=cat_vars, drop_first=False, dtype=float)
            if fit:
                self.ohe_columns_ = list(df_cat.columns)
            else:
                # Align columns with training set (add missing, drop extra)
                for col in self.ohe_columns_:
                    if col not in df_cat.columns:
                        df_cat[col] = 0.0
                df_cat = df_cat[self.ohe_columns_]
        else:
            df_cat = pd.DataFrame(index=df.index)
            if fit:
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
        draws : int
            Number of posterior draws per chain (default 1000).
        tune : int
            Number of tuning steps per chain (default 1000).
        separate_trees : bool
            If True, fit independent trees per category.  
            Only used when target_type="categorical".  
            Reduces posterior variance but increases computation time.
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
            model, idata = self._fit_categorical(
                X, y_codes, m=m, separate_trees=separate_trees,
                chains=chains, draws=draws, tune=tune,
                random_seed=random_seed,
                sample_posterior_predictive=sample_posterior_predictive,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )
        else:  # ordinal
            model, idata = self._fit_ordinal(
                X, y_codes, m=m,
                chains=chains, draws=draws, tune=tune,
                random_seed=random_seed,
                sample_posterior_predictive=sample_posterior_predictive,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )

        self.model_ = model
        self.idata_ = idata
        self.fitted_ = True

        return self

    # --- Private model builders -------------------------------------------

    def _fit_categorical(
        self, X, y_codes, *, m, separate_trees, chains, draws, tune,
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
            mu = pmb.BART(
                "mu",
                X,
                y_codes,
                m=m,
                separate_trees=separate_trees,
                dims=["classes", "n_obs"],
            )
            theta = pm.Deterministic("theta", pm.math.softmax(mu, axis=0))
            y = pm.Categorical("y", p=theta.T, observed=y_codes)

            idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                random_seed=random_seed,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )
            if sample_posterior_predictive:
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return model, idata

    def _fit_ordinal(
        self, X, y_codes, *, m, chains, draws, tune,
        random_seed, sample_posterior_predictive, compute_convergence_checks,
        **sample_kwargs,
    ):
        """Build and sample the ordinal BART model (OrderedLogistic)."""
        n_classes = self.n_classes_

        with pm.Model() as model:
            # BART provides a 1-D latent score per observation
            mu = pmb.BART("mu", X, y_codes, m=m)

            # Cutpoints that partition the latent scale into n_classes bins.
            # The ordered transform ensures cutpoints[0] < cutpoints[1] < ...
            cutpoints = pm.Normal(
                "cutpoints",
                mu=0,
                sigma=1.5,
                shape=n_classes - 1,
                transform=pm.distributions.transforms.ordered,
            )

            y = pm.OrderedLogistic(
                "y", eta=mu, cutpoints=cutpoints, observed=y_codes,
            )

            idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                random_seed=random_seed,
                compute_convergence_checks=compute_convergence_checks,
                **sample_kwargs,
            )
            if sample_posterior_predictive:
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return model, idata

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        new_data: pd.DataFrame,
        random_seed: int = 42,
    ) -> dict:
        """
        Generate predictions for new data.

        Uses pymc_bart.predict to obtain out-of-sample BART predictions.  
        For categorical targets, softmax is applied;
            for ordinal targets, cumulative logistic probabilities are
            computed using the posterior cutpoints.

        Parameters
        ----------
        new_data : pd.DataFrame
            New observations with the same predictor columns used in
            training.
        random_seed : int
            Random seed for reproducibility.

        Returns
        -------
        result : dict
            "probabilities": array of shape (n_samples, n_obs, n_classes)
                with class probabilities per posterior draw.
            "predicted_classes": array of shape (n_obs,) with the
                most-likely class code (mode across posterior draws).
            "predicted_labels": list of str with the human-readable
                class labels corresponding to predicted_classes.
            "class_prob_mean": array of shape (n_obs, n_classes) with
                posterior-mean class probabilities per observation.
        """
        if not self.fitted_:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        X_new, _ = self.preprocess(new_data, fit=False)

        rng = np.random.default_rng(random_seed)

        # pmb.predict returns posterior predictions for the BART component.
        mu_pred = pmb.predict(self.idata_, rng, X_new=X_new, size=None)

        if self.target_type == "categorical":
            probs = self._predict_categorical(mu_pred)
        else:
            probs = self._predict_ordinal(mu_pred)

        # Mean probabilities across posterior draws
        class_prob_mean = probs.mean(axis=0)  # (n_obs, n_classes)

        # Point prediction: class with highest mean probability
        predicted_classes = class_prob_mean.argmax(axis=1)
        predicted_labels = [self.category_map_[c] for c in predicted_classes]

        return {
            "probabilities": probs,
            "predicted_classes": predicted_classes,
            "predicted_labels": predicted_labels,
            "class_prob_mean": class_prob_mean,
        }

    # --- Private prediction helpers ---------------------------------------

    def _predict_categorical(self, mu_pred: np.ndarray) -> np.ndarray:
        """Apply softmax to BART output for categorical prediction."""
        # mu_pred shape: (n_samples, n_classes, n_obs)
        probs = softmax(mu_pred, axis=1)
        return np.transpose(probs, (0, 2, 1))  # (n_samples, n_obs, n_classes)

    @staticmethod
    def _cumulative_log_probs(eta: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
        """
        Compute ordinal class probabilities via the cumulative logit model.

        Parameters
        ----------
        eta : array, shape (n_obs,)
            Latent score per observation (one posterior draw).
        cutpoints : array, shape (K-1,)
            Sorted cutpoints for K ordinal classes.

        Returns
        -------
        probs : array, shape (n_obs, K)
        """
        from scipy.special import expit  # logistic sigmoid

        K = len(cutpoints) + 1
        n_obs = len(eta)
        probs = np.empty((n_obs, K), dtype=np.float64)

        # cumulative P(Y <= k) = sigmoid(cutpoint_k - eta)
        cum = expit(cutpoints[np.newaxis, :] - eta[:, np.newaxis])  # (n_obs, K-1)

        probs[:, 0] = cum[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = cum[:, k] - cum[:, k - 1]
        probs[:, K - 1] = 1.0 - cum[:, -1]

        # Clip to avoid tiny negatives from floating-point arithmetic
        np.clip(probs, 0.0, 1.0, out=probs)
        return probs

    def _predict_ordinal(self, mu_pred: np.ndarray) -> np.ndarray:
        """Compute ordinal class probabilities from BART latent + cutpoints."""
        # mu_pred shape: (n_samples, n_obs)
        # Posterior cutpoints: (n_chains, n_draws, K-1)
        cutpoints_post = self.idata_.posterior["cutpoints"].values  # (chains, draws, K-1)
        n_chains, n_draws, _ = cutpoints_post.shape
        cutpoints_flat = cutpoints_post.reshape(n_chains * n_draws, -1)  # (n_samples, K-1)

        n_samples, n_obs = mu_pred.shape
        n_classes = self.n_classes_
        probs = np.empty((n_samples, n_obs, n_classes), dtype=np.float64)

        for s in range(n_samples):
            probs[s] = self._cumulative_log_probs(mu_pred[s], cutpoints_flat[s])

        return probs

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
        return (
            f"BARTModelWrapper(target='{self.target_var}', "
            f"target_type='{self.target_type}', "
            f"n_predictors={len(self.predictor_vars)}, "
            f"fill_missing={self.fill_missing}, "
            f"status={status})"
        )
