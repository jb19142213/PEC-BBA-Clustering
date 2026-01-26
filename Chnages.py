from sklearn.datasets import fetch_openml
import pandas as pd

def load_uci_dataset(name, target_col=None):
    """
    Load a UCI dataset from OpenML.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    df : pd.DataFrame (features + target)
    """
    data = fetch_openml(name=name, as_frame=True)
    df = data.frame

    if target_col is None:
        target_col = data.target.name

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    return X, y, df


_, _, df = load_uci_dataset("satimage")

# ============================================
# 2. Inject Missing Values (MCAR)
# ============================================

feature_cols = df.columns.difference(["class"])
X1 = df[feature_cols].to_numpy().astype(float)

def inject_missing_entries(X, missing_ratio, seed=42):
    """
    Entry-wise MCAR missingness.
    Guarantees ~missing_ratio fraction of ALL entries are missing.
    """
    rng = np.random.default_rng(seed)

    X_missing = X.copy()
    n, d = X_missing.shape
    total_entries = n * d
    n_missing = int(missing_ratio * total_entries)

    flat_indices = rng.choice(
        total_entries, size=n_missing, replace=False
    )

    rows = flat_indices // d
    cols = flat_indices % d

    X_missing[rows, cols] = np.nan

    return X_missing

X_missing = inject_missing_entries(X1, missing_ratio=0)

df = df.copy()
df[feature_cols] = X_missing
df=df[feature_cols]

# ============================================
# 3. Create Feature Matrix and Mask
# ============================================

X = df.values.astype(np.float32)

# Binary mask: 1 = observed, 0 = missing
mask = (~np.isnan(X)).astype(np.float32)

# Fill missing values with zero (mask-aware)
X_filled = np.nan_to_num(X, nan=0.0)
