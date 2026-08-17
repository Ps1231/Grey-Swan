"""
Grey-Swan Baseline Models & AutoML
====================================
Implements 6 baseline classifiers + Optuna AutoML search.

Usage:
    python code/baselines_automl.py                 # all baselines + AutoML
    python code/baselines_automl.py --baselines-only # baselines only
    python code/baselines_automl.py --automl-only    # AutoML only
    python code/baselines_automl.py --optuna-trials 200  # custom trial count

Output:
    data/processed/model_results.csv
    data/processed/model_comparison.png
    data/processed/automl_best_params.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
VIZ_DIR = OUT / "graph_snapshots"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Torch setup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, roc_curve, brier_score_loss,
    confusion_matrix, classification_report
)

# Boosting
import xgboost as xgb
import lightgbm as lgb


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data(target_horizon: str = "5d"):
    """
    Load features + regime labels, create train/val/test splits.
    Uses temporal split: train < 2015, val 2015-2020, test 2021+.
    """
    console.print("  [cyan]Loading data...[/]")
    features = pd.read_parquet(OUT / "features_dataset.parquet")
    regimes = pd.read_parquet(OUT / "regime_labels.parquet")

    # Align
    common_idx = features.index.intersection(regimes.index)
    features = features.loc[common_idx]
    regimes = regimes.loc[common_idx]

    target_col = f"target_extreme_{target_horizon}"
    if target_col not in regimes.columns:
        console.print(f"[red]Target column {target_col} not found[/]")
        sys.exit(1)

    y = regimes[target_col].values.astype(int)
    X = features.copy()

    # Drop rows with NaN target
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]

    # Replace inf, fill NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Temporal split
    train_mask = X.index < "2015-01-01"
    val_mask = (X.index >= "2015-01-01") & (X.index < "2021-01-01")
    test_mask = X.index >= "2021-01-01"

    X_train, y_train = X[train_mask].values, y[train_mask]
    X_val, y_val = X[val_mask].values, y[val_mask]
    X_test, y_test = X[test_mask].values, y[test_mask]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    console.print(f"  Train: {X_train.shape[0]:,} (extreme: {y_train.mean():.1%})")
    console.print(f"  Val:   {X_val.shape[0]:,} (extreme: {y_val.mean():.1%})")
    console.print(f"  Test:  {X_test.shape[0]:,} (extreme: {y_test.mean():.1%})")

    meta = {
        "feature_names": list(features.columns),
        "n_features": X_train.shape[1],
        "target_col": target_col,
        "class_balance": float(y_train.mean()),
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, meta


# ═══════════════════════════════════════════════════════════════════════════
# 2. EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def recall_at_fpr(y_true, y_prob, target_fpr=0.05):
    """Recall at a controlled false positive rate."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        return tpr[-1] if len(tpr) > 0 else 0.0
    return tpr[idx]


def evaluate(y_true, y_prob, model_name="model"):
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "recall_at_5pct_fpr": recall_at_fpr(y_true, y_prob, 0.05),
        "recall_at_10pct_fpr": recall_at_fpr(y_true, y_prob, 0.10),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 3. BASELINE MODELS
# ═══════════════════════════════════════════════════════════════════════════

def run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Logistic Regression baseline."""
    console.print("  [cyan]Logistic Regression...[/]")
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=0.1, solver="lbfgs",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    return prob_val, prob_test, model


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Random Forest baseline."""
    console.print("  [cyan]Random Forest...[/]")
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    return prob_val, prob_test, model


def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """XGBoost baseline."""
    console.print("  [cyan]XGBoost...[/]")
    scale_pos = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr", early_stopping_rounds=30,
        random_state=42, n_jobs=-1, verbosity=0,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    return prob_val, prob_test, model


def run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """LightGBM baseline."""
    console.print("  [cyan]LightGBM...[/]")
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    return prob_val, prob_test, model


def run_lstm(X_train, y_train, X_val, y_val, X_test, y_test, n_features):
    """LSTM baseline with sequence input."""
    console.print("  [cyan]LSTM...[/]")

    seq_len = 20

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i])
        if not Xs:
            return np.array([]), np.array([])
        return np.array(Xs), np.array(ys)

    Xtr_seq, ytr = make_sequences(X_train, y_train, seq_len)
    Xva_seq, yva = make_sequences(X_val, y_val, seq_len)
    Xte_seq, yte = make_sequences(X_test, y_test, seq_len)

    if len(Xtr_seq) == 0:
        console.print("  [yellow]Not enough data for LSTM sequences[/]")
        return np.zeros(len(y_val)), np.zeros(len(y_test)), None

    Xtr_t = torch.FloatTensor(Xtr_seq)
    ytr_t = torch.FloatTensor(ytr)
    Xva_t = torch.FloatTensor(Xva_seq)
    yva_t = torch.FloatTensor(yva)
    Xte_t = torch.FloatTensor(Xte_seq)

    # Handle class imbalance with manual weighted BCE
    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    def weighted_bce(pred, target):
        bce = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
        weight = torch.where(target == 1, pw, 1.0)
        return (bce * weight).mean()

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden=64, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=2,
                                batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = LSTMModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_loss = float("inf")
    best_state = None

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    with Progress(SpinnerColumn(), TextColumn("[cyan]LSTM[/] {task.description}"),
                  BarColumn(20), TextColumn("{task.fields[val_loss]:.4f}"), TimeElapsedColumn(),
                  console=console, transient=True) as progress:
        task = progress.add_task("training", total=30, val_loss=float("inf"))
        for epoch in range(30):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = weighted_bce(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(Xva_t)
                val_loss = weighted_bce(val_pred, yva_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            progress.update(task, advance=1, val_loss=val_loss)

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        prob_val = model(Xva_t).numpy()
        prob_test = model(Xte_t).numpy()

    return prob_val, prob_test, model


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


def run_tcn(X_train, y_train, X_val, y_val, X_test, y_test, n_features):
    """Temporal Convolutional Network baseline."""
    console.print("  [cyan]TCN...[/]")

    seq_len = 20

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i])
        if not Xs:
            return np.array([]), np.array([])
        return np.array(Xs), np.array(ys)

    Xtr_seq, ytr = make_sequences(X_train, y_train, seq_len)
    Xva_seq, yva = make_sequences(X_val, y_val, seq_len)
    Xte_seq, yte = make_sequences(X_test, y_test, seq_len)

    if len(Xtr_seq) == 0:
        return np.zeros(len(y_val)), np.zeros(len(y_test)), None

    Xtr_t = torch.FloatTensor(Xtr_seq).permute(0, 2, 1)
    ytr_t = torch.FloatTensor(ytr)
    Xva_t = torch.FloatTensor(Xva_seq).permute(0, 2, 1)
    yva_t = torch.FloatTensor(yva)
    Xte_t = torch.FloatTensor(Xte_seq).permute(0, 2, 1)

    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    def weighted_bce(pred, target):
        bce = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
        weight = torch.where(target == 1, pw, 1.0)
        return (bce * weight).mean()

    class TCNModel(nn.Module):
        def __init__(self, input_dim, dropout=0.3):
            super().__init__()
            self.tcn = nn.Sequential(
                TCNBlock(input_dim, 64, 3, dropout),
                TCNBlock(64, 32, 3, dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(16, 1), nn.Sigmoid()
            )
        def forward(self, x):
            return self.head(self.tcn(x)).squeeze(-1)

    model = TCNModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_loss = float("inf")
    best_state = None

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    with Progress(SpinnerColumn(), TextColumn("[cyan]TCN[/] {task.description}"),
                  BarColumn(20), TextColumn("{task.fields[val_loss]:.4f}"), TimeElapsedColumn(),
                  console=console, transient=True) as progress:
        task = progress.add_task("training", total=30, val_loss=float("inf"))
        for epoch in range(30):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = weighted_bce(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(Xva_t)
                val_loss = weighted_bce(val_pred, yva_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            progress.update(task, advance=1, val_loss=val_loss)

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        prob_val = model(Xva_t).numpy()
        prob_test = model(Xte_t).numpy()

    return prob_val, prob_test, model


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.ff(x))
        return x


def run_transformer(X_train, y_train, X_val, y_val, X_test, y_test, n_features):
    """Standard Transformer baseline."""
    console.print("  [cyan]Transformer...[/]")

    seq_len = 20
    d_model = 64

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i])
        if not Xs:
            return np.array([]), np.array([])
        return np.array(Xs), np.array(ys)

    Xtr_seq, ytr = make_sequences(X_train, y_train, seq_len)
    Xva_seq, yva = make_sequences(X_val, y_val, seq_len)
    Xte_seq, yte = make_sequences(X_test, y_test, seq_len)

    if len(Xtr_seq) == 0:
        return np.zeros(len(y_val)), np.zeros(len(y_test)), None

    Xtr_t = torch.FloatTensor(Xtr_seq)
    ytr_t = torch.FloatTensor(ytr)
    Xva_t = torch.FloatTensor(Xva_seq)
    yva_t = torch.FloatTensor(yva)
    Xte_t = torch.FloatTensor(Xte_seq)

    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    def weighted_bce(pred, target):
        bce = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
        weight = torch.where(target == 1, pw, 1.0)
        return (bce * weight).mean()

    class TransformerModel(nn.Module):
        def __init__(self, input_dim, d_model, n_heads=4, dropout=0.3):
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, dropout) for _ in range(2)
            ])
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        def forward(self, x):
            x = self.proj(x) + self.pos_emb[:, :x.size(1), :]
            for layer in self.layers:
                x = layer(x)
            return self.head(x.permute(0, 2, 1)).squeeze(-1)

    model = TransformerModel(n_features, d_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_state = None

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    with Progress(SpinnerColumn(), TextColumn("[cyan]Transformer[/] {task.description}"),
                  BarColumn(20), TextColumn("{task.fields[val_loss]:.4f}"), TimeElapsedColumn(),
                  console=console, transient=True) as progress:
        task = progress.add_task("training", total=30, val_loss=float("inf"))
        for epoch in range(30):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = weighted_bce(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(Xva_t)
                val_loss = weighted_bce(val_pred, yva_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            progress.update(task, advance=1, val_loss=val_loss)

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        prob_val = model(Xva_t).numpy()
        prob_test = model(Xte_t).numpy()

    return prob_val, prob_test, model


# ═══════════════════════════════════════════════════════════════════════════
# 4. Optuna AutoML
# ═══════════════════════════════════════════════════════════════════════════

def run_automl(X_train, y_train, X_val, y_val, n_trials=100):
    """Optuna AutoML: searches over XGBoost, LightGBM, RF hyperparameters."""
    import optuna
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    console.print(f"\n  [cyan]Optuna AutoML ({n_trials} trials)...[/]")

    best_value = 0.0
    best_trial_num = 0

    def objective(trial):
        nonlocal best_value, best_trial_num
        trial_num = trial.number + 1
        model_type = trial.suggest_categorical("model", ["xgboost", "lightgbm", "rf"])

        if model_type == "xgboost":
            scale_pos = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child", 1, 20),
                "scale_pos_weight": scale_pos,
                "eval_metric": "aucpr", "early_stopping_rounds": 20,
                "random_state": 42, "n_jobs": -1, "verbosity": 0,
                "tree_method": "hist",
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child", 5, 50),
                "class_weight": "balanced",
                "random_state": 42, "n_jobs": -1, "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])

        else:  # rf
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_leaf", 5, 50),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "class_weight": "balanced",
                "random_state": 42, "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

        prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, prob)

        if score > best_value:
            best_value = score
            best_trial_num = trial_num

        return score

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TextColumn("Best: {task.fields[best_score]:.4f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Optuna search", total=n_trials, best_score=0.0)

        def callback(study, trial):
            nonlocal best_value
            progress.update(task, advance=1, best_score=best_value)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[callback])

    best = study.best_trial
    console.print(f"  Best PR-AUC: {best.value:.4f} ({best.params['model']})")

    # Retrain best model on train+val
    best_model_type = best.params.pop("model")
    n_est = best.params.pop("n_estimators", 300)

    if best_model_type == "xgboost":
        sp = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
        best_model = xgb.XGBClassifier(
            n_estimators=n_est, scale_pos_weight=sp,
            eval_metric="aucpr", random_state=42, n_jobs=-1,
            verbosity=0, tree_method="hist", **best.params
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif best_model_type == "lightgbm":
        best_model = lgb.LGBMClassifier(
            n_estimators=n_est, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1, **best.params
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(20, verbose=False)])
    else:
        best_model = RandomForestClassifier(
            n_estimators=n_est, class_weight="balanced",
            random_state=42, n_jobs=-1, **best.params
        )
        best_model.fit(X_train, y_train)

    return best_model, best.value, study


# ═══════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(results: list[dict]):
    """Plot model comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    console.print("  [cyan]Plotting comparison...[/]")
    df = pd.DataFrame(results).set_index("model")

    metrics = ["pr_auc", "roc_auc", "f1", "recall_at_5pct_fpr", "recall_at_10pct_fpr"]
    labels = ["PR-AUC", "ROC-AUC", "F1", "Recall@5%FPR", "Recall@10%FPR"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    fig.suptitle("Baseline Model Comparison (Test Set)", fontsize=14, fontweight="bold")

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        vals = df[metric].sort_values(ascending=True)
        bars = ax.barh(range(len(vals)), vals.values, color=colors[:len(vals)])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=9)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, vals.values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = VIZ_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved: {path}")


def plot_pr_curves(y_test, prob_dict: dict):
    """Plot precision-recall curves for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    console.print("  [cyan]Plotting PR curves...[/]")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Precision-Recall Curves (Test Set)", fontsize=14, fontweight="bold")

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]

    for i, (name, probs) in enumerate(prob_dict.items()):
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(rec, prec, color=colors[i % len(colors)], linewidth=1.5,
                label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    path = VIZ_DIR / "pr_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run(baselines_only=False, automl_only=False, optuna_trials=100, target="5d"):
    t0 = time.time()
    console.print(Panel.fit(
        "[bold cyan]Grey-Swan Baseline Models & AutoML[/]", border_style="cyan"
    ))

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, meta = prepare_data(target)
    n_features = meta["n_features"]

    all_results = []
    all_probs = {}

    if not automl_only:
        console.print("\n[bold]Running Baselines[/]")

        # 1. Logistic Regression
        try:
            prob_v, prob_t, _ = run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
            m = evaluate(y_test, prob_t, "Logistic Regression")
            all_results.append(m)
            all_probs["Logistic Regression"] = prob_t
            console.print(f"    PR-AUC: {m['pr_auc']:.4f}")
        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

        # 2. Random Forest
        try:
            prob_v, prob_t, _ = run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
            m = evaluate(y_test, prob_t, "Random Forest")
            all_results.append(m)
            all_probs["Random Forest"] = prob_t
            console.print(f"    PR-AUC: {m['pr_auc']:.4f}")
        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

        # 3. XGBoost
        try:
            prob_v, prob_t, _ = run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
            m = evaluate(y_test, prob_t, "XGBoost")
            all_results.append(m)
            all_probs["XGBoost"] = prob_t
            console.print(f"    PR-AUC: {m['pr_auc']:.4f}")
        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

        # 4. LightGBM
        try:
            prob_v, prob_t, _ = run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
            m = evaluate(y_test, prob_t, "LightGBM")
            all_results.append(m)
            all_probs["LightGBM"] = prob_t
            console.print(f"    PR-AUC: {m['pr_auc']:.4f}")
        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

        # 5. LSTM
        try:
            prob_v, prob_t, _ = run_lstm(X_train, y_train, X_val, y_val, X_test, y_test, n_features)
            m = evaluate(y_test, prob_t, "LSTM")
            all_results.append(m)
            all_probs["LSTM"] = prob_t
            console.print(f"    PR-AUC: {m['pr_auc']:.4f}")
        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

        # 6. TCN
        try:
            prob_v, prob_t, _ = run_tcn(X_train, y_train, X_val, y_val, X_test, y_test, n_features)
            m = evaluate(y_test, prob_t, "TCN")
            all_results.append(m)
            all_probs["TCN"] = prob_t
            console.print(f"    PR-AUC: {m['pr_auc']:.4f}")
        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

        # 7. Transformer
        try:
            prob_v, prob_t, _ = run_transformer(X_train, y_train, X_val, y_val, X_test, y_test, n_features)
            m = evaluate(y_test, prob_t, "Transformer")
            all_results.append(m)
            all_probs["Transformer"] = prob_t
            console.print(f"    PR-AUC: {m['pr_auc']:.4f}")
        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

    # AutoML
    if not baselines_only:
        console.print("\n[bold]Running AutoML (Optuna)[/]")
        try:
            best_model, best_score, study = run_automl(
                X_train, y_train, X_val, y_val, n_trials=optuna_trials
            )

            # Evaluate on test
            if hasattr(best_model, "predict_proba"):
                prob_test = best_model.predict_proba(X_test)[:, 1]
            else:
                prob_test = np.zeros(len(y_test))

            m = evaluate(y_test, prob_test, f"AutoML ({study.best_trial.params.get('model', 'best')})")
            all_results.append(m)
            all_probs[m["model"]] = prob_test
            console.print(f"    Test PR-AUC: {m['pr_auc']:.4f}")

            # Save best params
            best_params = study.best_trial.params
            best_params["val_pr_auc"] = best_score
            params_path = OUT / "automl_best_params.json"
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2, default=str)
            console.print(f"    Saved: {params_path}")

        except Exception as e:
            console.print(f"    [red][FAIL] {e}[/]")

    # Results table
    if all_results:
        console.print("\n[bold]Results (Test Set):[/]")
        table = Table(title=f"Baseline Comparison (Target: {target} extreme events)")
        table.add_column("Model", style="cyan")
        table.add_column("PR-AUC", justify="right")
        table.add_column("ROC-AUC", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Recall@5%", justify="right")
        table.add_column("Recall@10%", justify="right")
        table.add_column("Brier", justify="right")
        table.add_column("ECE", justify="right")

        for r in sorted(all_results, key=lambda x: x["pr_auc"], reverse=True):
            table.add_row(
                r["model"],
                f"{r['pr_auc']:.4f}",
                f"{r['roc_auc']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['recall_at_5pct_fpr']:.4f}",
                f"{r['recall_at_10pct_fpr']:.4f}",
                f"{r['brier']:.4f}",
                f"{r['ece']:.4f}",
            )

        console.print(table)

        # Save results
        results_df = pd.DataFrame(all_results)
        results_path = OUT / "model_results.csv"
        results_df.to_csv(results_path, index=False)
        console.print(f"\n  Saved: {results_path}")

        # Visualizations
        plot_comparison(all_results)
        if len(all_probs) > 1:
            plot_pr_curves(y_test, all_probs)

    elapsed = time.time() - t0
    console.print(Panel.fit(
        f"[bold green]Complete in {elapsed:.1f}s[/]\n"
        f"Models evaluated: {len(all_results)}\n"
        f"Target: {target} extreme-event prediction",
        border_style="green"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baselines & AutoML")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--automl-only", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=100)
    parser.add_argument("--target", type=str, default="5d", choices=["5d", "10d", "20d"])
    args = parser.parse_args()
    run(baselines_only=args.baselines_only, automl_only=args.automl_only,
        optuna_trials=args.optuna_trials, target=args.target)
