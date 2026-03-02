"""Example: Using the BARTModelWrapper with modelDataDeIdentified.parquet

This script demonstrates both target_type options:
  1. "categorical": unordered multiclass classification
  2. "ordinal"    : ordered integer / ordinal classification

It also demonstrates how to use ``register_data(df)`` to let the
wrapper learn all category encodings from the *full* dataset before
the train / test split.  This ensures that every category is
represented in the encoding.  ``register_data`` is optional — if
omitted, encoders are learned from the training set only.
"""

import pandas as pd
from pathlib import Path
from pymc_bart_wrapper import BARTModelWrapper

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path('./data/modelDataDeIdentified.parquet')
if DATA_PATH.suffix == '.csv':
    df = pd.read_csv(DATA_PATH)
elif DATA_PATH.suffix == '.parquet':
    df = pd.read_parquet(DATA_PATH)
else:
    raise ValueError(f"Unsupported file format: {DATA_PATH.suffix}")

print(f"Dataset shape: {df.shape}")
print(f"Target classes: {df['finalized_label'].unique().tolist()}")
print(f"Target distribution:\n{df['finalized_label'].value_counts()}\n")

# ── Define variables ──────────────────────────────────────────────────────────
TARGET = "finalized_label"

PREDICTOR_VARS = [
    # Numeric predictors
    "pat_age_at_test",
    "hgb_a", "hgb_f", "hgb_s", "hgb_c",
    "hgb_a2", "hgb_a2_variant",
    "hgb_e", "hgb_barts", "hgb_h", "hgb_d",
    "hgb_other_hgb", "total_hgb_count",
    # CBC test results
    "rbc_mean", "hgb_mean", "hct_mean", "mcv_mean", "rdw_cv_mean",
    # Categorical predictors
    "sex",
    "hgb_a_category", "hgb_s_category", "hgb_c_category",
]

NON_NUMERIC_VARS = [
    "sex",
    "hgb_a_category", "hgb_s_category", "hgb_c_category",
]

# ── Simple train / test split ────────────────────────────────────────────────
from sklearn.model_selection import train_test_split  # noqa: E402

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])

print(f"Train size: {len(df_train)}")
print(f"Test  size: {len(df_test)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Example A – Multiclass Categorical model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Example A: target_type='categorical'")
print("=" * 70)

cat_wrapper = BARTModelWrapper(
    target_var=TARGET,
    predictor_vars=PREDICTOR_VARS,
    non_numeric_vars=NON_NUMERIC_VARS,
    target_type="categorical",       # <-- unordered multiclass
    fill_missing=True,               # fill NaN with missing_numeric_fill / "missing"
    missing_numeric_fill=-99,        # custom fill value for missing numerics
)

# Pre-fit encoders on the full dataset so every category is known.
cat_wrapper.register_data(df)
print(cat_wrapper)

# Fit (reduce draws/trees for demo speed; increase for real analysis)
cat_wrapper.fit(
    df_train,
    m=50,
    chains=2,
    draws=500,
    tune=500,
    random_seed=42,
)
print("Categorical model fitted.\n")

# In-sample inference data
idata_cat = cat_wrapper.get_inference_data()
print(idata_cat)

# Predict on held-out data
results_cat = cat_wrapper.predict(df_test, random_seed=42)
print(f"\nCategorical – predicted labels (first 10): {results_cat['predicted_labels'][:10]}")
print(f"Categorical – mean class probs shape:       {results_cat['class_prob_mean'].shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Example B – Ordinal model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Example B: target_type='ordinal'")
print("=" * 70)

# For ordinal modelling we need a meaningful low-to-high ordering of the
# diagnosis labels.  Here we define one based on clinical severity:
ORDINAL_ORDER = [
    "Normal",
    "Hemoglobin_C_Trait",
    "Sickle_Cell_Trait",
    "Beta_Thalassemia",
    "HGB_SC_Disease",
    "Sickle_Cell_Disease",
    "Other",
]

ord_wrapper = BARTModelWrapper(
    target_var=TARGET,
    predictor_vars=PREDICTOR_VARS,
    non_numeric_vars=NON_NUMERIC_VARS,
    target_type="ordinal",           # <-- ordered outcome
    ordinal_order=ORDINAL_ORDER,     #     explicit severity ordering
    fill_missing=True,
)

# Pre-fit encoders on the full dataset.
ord_wrapper.register_data(df)
print(ord_wrapper)

# Fit (reduce draws/trees for demo speed; increase for real analysis)
ord_wrapper.fit(
    df_train,
    m=50,
    chains=2,
    draws=500,
    tune=500,
    random_seed=42,
)
print("Ordinal model fitted.\n")

# Predict on held-out data
results_ord = ord_wrapper.predict(df_test, random_seed=42)
print(f"Ordinal – predicted labels (first 10): {results_ord['predicted_labels'][:10]}")
print(f"Ordinal – mean class probs shape:      {results_ord['class_prob_mean'].shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Quick accuracy comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Quick accuracy comparison on test set")
print("=" * 70)

# Build ground-truth codes for the test set using each wrapper's encoding
_, y_test_cat = cat_wrapper.preprocess(df_test, fit=False)
_, y_test_ord = ord_wrapper.preprocess(df_test, fit=False)

# For test labels we re-encode from the raw data
y_true_labels = df_test[TARGET].values

acc_cat = (results_cat["predicted_labels"] == y_true_labels).mean()
acc_ord = (results_ord["predicted_labels"] == y_true_labels).mean()

print(f"Categorical accuracy: {acc_cat:.3f}")
print(f"Ordinal     accuracy: {acc_ord:.3f}")
