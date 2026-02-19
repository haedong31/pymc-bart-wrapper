"""
Example: Using the BARTModelWrapper with modelDataDeIdentified.csv

This script demonstrates both target_type options:
  1. "categorical": unordered multiclass classification
  2. "ordinal"    : ordered integer / ordinal classification

The dataset contains hemoglobin-related lab results for predicting a
`finalized_label` diagnosis.
"""

import sys
import os

import pandas as pd

# ── make the data/ folder importable ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from pymc_bart_wrapper import BARTModelWrapper  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "modelDataDeIdentified.csv")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Target classes: {df['finalized_label'].unique().tolist()}")
print(f"Target distribution:\n{df['finalized_label'].value_counts()}\n")

# ── Define variables ──────────────────────────────────────────────────────────
TARGET = "finalized_label"

PREDICTOR_VARS = [
    # Numeric hemoglobin quantitation results
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
    impute_missing=True,             # fill NaN with -99 / "missing"
)
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
    impute_missing=True,
)
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
