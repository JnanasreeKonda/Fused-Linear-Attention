"""
model/data.py — ETTh1 download, EDA, preprocessing, Dataset, and DataLoaders.
Owner: Rithwik Amajala  |  Milestone: M5  |  Phase 1

Responsibilities:
  1. Download ETTh1.csv from the official repo (if not present)
  2. EDA: inspect shape, columns, missing values, plot OT column
  3. Preprocessing:
       - Zero-mean / unit-variance normalisation per feature
       - Fit scaler on TRAIN split ONLY; apply to val & test (no leakage)
       - Sliding-window sample generation (input_len=96, forecast_len=96)
       - Standard 12 / 4 / 4 month train / val / test split
  4. ETTh1Dataset – PyTorch Dataset with __len__ / __getitem__
  5. get_dataloaders() – returns (train_loader, val_loader, test_loader, mean, std)

Usage (smoke-test):
    python model/data.py
"""

import os
import urllib.request

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on Greene nodes
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ETTh1 feature columns in CSV order
FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
OT_IDX = FEATURE_COLS.index(config.TARGET_COL)   # 6


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download
# ──────────────────────────────────────────────────────────────────────────────

def download_etth1(data_path: str = config.DATA_PATH) -> None:
    """Download ETTh1.csv if not already present."""
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    if not os.path.exists(data_path):
        print(f"[data] Downloading ETTh1 → {data_path} …")
        urllib.request.urlretrieve(config.DATA_URL, data_path)
        print("[data] Download complete.")
    else:
        print(f"[data] ETTh1 already present at {data_path}.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. EDA
# ──────────────────────────────────────────────────────────────────────────────

def inspect_etth1(data_path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load CSV, print basic statistics, return DataFrame."""
    df = pd.read_csv(data_path, parse_dates=["date"])
    print("\n── ETTh1 Dataset Statistics ──────────────────────────────────")
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Date range  : {df['date'].min()}  →  {df['date'].max()}")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    print(f"\n  Numeric summary (7 features):\n{df[FEATURE_COLS].describe().to_string()}")
    print("──────────────────────────────────────────────────────────────\n")
    return df


def plot_ot_column(
    df: pd.DataFrame,
    save_dir: str = "results/figures",
) -> None:
    """Plot the OT (oil temperature) target column; save PNG."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df["date"], df[config.TARGET_COL], linewidth=0.5, color="steelblue")
    ax.set_title("ETTh1 — OT (Oil Temperature) target column", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil Temp (°C)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(save_dir, "etth1_ot_column.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[data] OT plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_etth1(
    data_path: str = config.DATA_PATH,
    input_len: int = config.INPUT_LEN,
    forecast_len: int = config.FORECAST_LEN,
    train_size: int = config.TRAIN_SIZE,
    val_size: int = config.VAL_SIZE,
    test_size: int = config.TEST_SIZE,
) -> tuple:
    """
    Load, split, normalise, and window ETTh1.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test, mean, std)
      X_*   : np.float32  (N, input_len, n_features)
      y_*   : np.float32  (N, forecast_len)   — normalised OT column
      mean  : np.float32  (n_features,)       — train-split mean
      std   : np.float32  (n_features,)       — train-split std
    """
    df = pd.read_csv(data_path, parse_dates=["date"])
    values = df[FEATURE_COLS].values.astype(np.float32)

    # ── Raw split BEFORE normalisation (no leakage) ───────────────────────────
    train_raw = values[:train_size]
    val_raw   = values[train_size : train_size + val_size]
    test_raw  = values[train_size + val_size : train_size + val_size + test_size]

    # ── Fit scaler on train split only ────────────────────────────────────────
    mean = train_raw.mean(axis=0)
    std  = train_raw.std(axis=0) + 1e-8   # avoid div-by-zero

    def _normalise(arr: np.ndarray) -> np.ndarray:
        return (arr - mean) / std

    train_norm = _normalise(train_raw)
    val_norm   = _normalise(val_raw)
    test_norm  = _normalise(test_raw)

    # ── Sliding-window sample generation ─────────────────────────────────────
    def _make_windows(arr: np.ndarray):
        n = len(arr)
        X, y = [], []
        for i in range(n - input_len - forecast_len + 1):
            X.append(arr[i : i + input_len])                              # all features
            y.append(arr[i + input_len : i + input_len + forecast_len, OT_IDX])  # OT only
        return np.stack(X), np.stack(y)

    X_train, y_train = _make_windows(train_norm)
    X_val,   y_val   = _make_windows(val_norm)
    X_test,  y_test  = _make_windows(test_norm)

    print(f"[data] Windows → train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")
    print(f"[data] X shape : {X_train.shape}   y shape : {y_train.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, mean, std


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ETTh1Dataset(Dataset):
    """PyTorch Dataset wrapping pre-processed ETTh1 sliding-window samples."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, input_len, n_features)
        self.y = torch.from_numpy(y)   # (N, forecast_len)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 5. DataLoaders
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_path: str   = config.DATA_PATH,
    batch_size: int  = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
):
    """
    Download data if needed, run preprocessing, return DataLoaders.

    Returns
    -------
    (train_loader, val_loader, test_loader, mean, std)
    """
    download_etth1(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test, mean, std = preprocess_etth1(
        data_path
    )

    train_loader = DataLoader(
        ETTh1Dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ETTh1Dataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ETTh1Dataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, mean, std


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_etth1()
    df = inspect_etth1()
    os.makedirs("results/figures", exist_ok=True)
    plot_ot_column(df)

    train_loader, val_loader, test_loader, mean, std = get_dataloaders(num_workers=0)
    xb, yb = next(iter(train_loader))
    print(f"[data] Batch — X: {tuple(xb.shape)}  y: {tuple(yb.shape)}  dtype: {xb.dtype}")
    print(f"[data] Mean (OT): {mean[OT_IDX]:.4f}  Std (OT): {std[OT_IDX]:.4f}")
    print("[data] ✓ data.py smoke-test passed.")
