"""
Grey-Swan: Spatiotemporal Graph-Transformer for Extreme Market Regime Detection
================================================================================
Core proposed model combining:
  1. Dynamic Graph Neural Network (pure PyTorch, no PyG dependency)
  2. Temporal Transformer encoder
  3. Extreme Value Theory (EVT) tail-risk head
  4. Multi-task prediction head (regime, extreme prob, max drawdown)

Usage:
    python code/grey_swan_model.py                    # full model + ablation
    python code/grey_swan_model.py --config 11        # specific ablation config
    python code/grey_swan_model.py --epochs 50        # custom epochs

Output:
    data/processed/grey_swan_results.csv
    data/processed/grey_swan_ablation.csv
    data/processed/grey_swan_predictions.csv
    data/processed/grey_swan_*_curves.png
    data/processed/grey_swan_ablation_chart.png
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.live import Live
from rich.text import Text
from rich.layout import Layout

warnings.filterwarnings("ignore")
console = Console()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
GRAPH_DIR = OUT / "dynamic_graphs"
VIZ_DIR = OUT / "graph_snapshots"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(8, torch.get_num_threads()))
N_NODES = 16
HIDDEN_DIM = 64
N_HEADS = 4
DROPOUT = 0.3
SEQ_LEN = 20


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load features, regime labels, and graph snapshots."""
    console.print("\n[bold cyan]Loading data...[/]")

    features = pd.read_parquet(OUT / "features_dataset.parquet")
    console.print(f"  Features: {features.shape[0]:,} rows x {features.shape[1]} cols")

    regime_df = pd.read_parquet(OUT / "regime_labels.parquet")
    console.print(f"  Regime labels: {regime_df.shape[0]:,} rows x {regime_df.shape[1]} cols")

    # Load graph index
    graph_idx = pd.read_csv(GRAPH_DIR / "graph_index.csv")
    graph_idx["date"] = pd.to_datetime(graph_idx["date"])
    graph_idx = graph_idx.set_index("date").sort_index()
    console.print(f"  Graph snapshots: {len(graph_idx)}")

    return features, regime_df, graph_idx


def load_graph_snapshot(date_str):
    """Load a single graph snapshot from .npz file."""
    path = GRAPH_DIR / f"graph_{date_str}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "edges": data["edges"],
        "edge_weights": data["edge_weights"],
        "node_returns": data["node_returns"],
        "node_vol": data["node_vol"],
        "corr_matrix": data["corr_matrix"],
        "regime": data["regime"].item(),
    }


def build_adj_matrix(edges, edge_weights, n_nodes, corr_matrix):
    """Build adjacency matrix from edges, falling back to correlation matrix."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if len(edges) > 0:
        for (i, j), w in zip(edges, edge_weights):
            adj[i, j] = abs(w)
            adj[j, i] = abs(w)
    else:
        adj = np.abs(corr_matrix).astype(np.float32)
        np.fill_diagonal(adj, 0)
    return adj


# ═══════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════════════

class GreySwanDataset(Dataset):
    """
    Creates sequences of graph snapshots + tabular features + targets.

    Each sample: (graph_sequence, tabular_features, targets)
    - graph_sequence: (seq_len, n_nodes, node_feat_dim) + (seq_len, n_nodes, n_nodes)
    - tabular_features: (n_tabular_features,)
    - targets: dict of regime, extreme_5d/10d/20d, maxdd_5d/10d/20d
    """

    # Class-level cache: load all graph snapshots once, shared across splits
    _all_graphs: dict = None

    def __init__(self, features, regime_df, graph_idx, seq_len=20, split="train"):
        self.seq_len = seq_len
        self.graph_idx = graph_idx
        self.features = features
        self.regime_df = regime_df

        # Align all dates
        common_dates = features.index.intersection(regime_df.index).intersection(graph_idx.index)
        common_dates = common_dates.sort_values()

        # Temporal split
        if split == "train":
            self.dates = common_dates[common_dates < "2015-01-01"]
        elif split == "val":
            self.dates = common_dates[(common_dates >= "2015-01-01") & (common_dates < "2021-01-01")]
        else:
            self.dates = common_dates[common_dates >= "2021-01-01"]

        self.tabular_cols = [c for c in features.columns if c not in regime_df.columns]

        console.print(f"  Building {split} dataset ({len(self.dates):,} samples)...")
        self._load_all_graphs_once()
        self._build_date_map()
        self._cache_tabular()

    @classmethod
    def _load_all_graphs_once(cls):
        """Load all 1284 graph snapshots once, cached at class level."""
        if cls._all_graphs is not None:
            console.print(f"    Using cached {len(cls._all_graphs)} graph snapshots")
            return

        t0 = time.time()
        cls._all_graphs = {}
        loaded = 0
        for f in GRAPH_DIR.glob("graph_*.npz"):
            try:
                data = np.load(f, allow_pickle=True)
                date_str = f.stem.replace("graph_", "")
                date_key = pd.Timestamp(
                    year=int(date_str[:4]), month=int(date_str[4:6]), day=int(date_str[6:8])
                )
                adj = build_adj_matrix(data["edges"], data["edge_weights"],
                                       N_NODES, data["corr_matrix"])
                node_feat = np.stack([data["node_returns"], data["node_vol"]], axis=-1)
                cls._all_graphs[date_key] = (
                    node_feat.astype(np.float32), adj, int(data["regime"].item())
                )
                loaded += 1
            except Exception:
                continue
        elapsed = time.time() - t0
        console.print(f"    Loaded {loaded} graph snapshots in {elapsed:.1f}s")

    def _build_date_map(self):
        """Map each dataset date to its nearest graph snapshot (backward fill)."""
        graph_dates = sorted(GreySwanDataset._all_graphs.keys())
        self.date_to_graph = {}
        missing = 0
        for date in self.dates:
            # Find nearest graph date <= this date
            idx = np.searchsorted(graph_dates, date, side="right") - 1
            if idx >= 0:
                self.date_to_graph[date] = graph_dates[idx]
            else:
                missing += 1
        console.print(f"    Mapped {len(self.date_to_graph)} dates, {missing} without graph")

    def _cache_tabular(self):
        """Pre-compute tabular features."""
        common_cols = [c for c in self.tabular_cols if c in self.features.columns]
        feat_subset = self.features.loc[self.dates, common_cols].copy()
        feat_subset = feat_subset.fillna(0)
        self.tabular_data = feat_subset.values.astype(np.float32)

        # Targets
        regime_col = "regime" if "regime" in self.regime_df.columns else None
        self.target_data = self.regime_df.loc[self.dates].copy()

    def __len__(self):
        return max(0, len(self.dates) - self.seq_len)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        seq_dates = self.dates[start:end]
        target_date = self.dates[end] if end < len(self.dates) else self.dates[-1]

        # Graph sequence via nearest-snapshot mapping
        node_feats = []
        adj_mats = []
        for d in seq_dates:
            graph_date = self.date_to_graph.get(d)
            if graph_date and graph_date in GreySwanDataset._all_graphs:
                nf, adj, _ = GreySwanDataset._all_graphs[graph_date]
            else:
                nf = np.zeros((N_NODES, 2), dtype=np.float32)
                adj = np.zeros((N_NODES, N_NODES), dtype=np.float32)
            node_feats.append(nf)
            adj_mats.append(adj)

        node_feats = torch.FloatTensor(np.stack(node_feats))
        adj_mats = torch.FloatTensor(np.stack(adj_mats))

        # Tabular features (at target date)
        tab_idx = list(self.dates).index(target_date) if target_date in self.dates else -1
        if tab_idx >= 0 and tab_idx < len(self.tabular_data):
            tabular = torch.FloatTensor(self.tabular_data[tab_idx])
        else:
            tabular = torch.zeros(len(self.tabular_cols), dtype=torch.float32)

        # Targets
        targets = {}
        if target_date in self.target_data.index:
            row = self.target_data.loc[target_date]
            targets["regime"] = int(row.get("regime", 0))
            for h in ["5d", "10d", "20d"]:
                targets[f"extreme_{h}"] = float(row.get(f"target_extreme_{h}", 0))
                targets[f"maxdd_{h}"] = float(row.get(f"target_maxdd_{h}", 0))
        else:
            targets["regime"] = 0
            for h in ["5d", "10d", "20d"]:
                targets[f"extreme_{h}"] = 0.0
                targets[f"maxdd_{h}"] = 0.0

        return node_feats, adj_mats, tabular, targets


def collate_fn(batch):
    """Custom collate for variable-structure batches."""
    node_feats = torch.stack([b[0] for b in batch])
    adj_mats = torch.stack([b[1] for b in batch])
    tabulars = torch.stack([b[2] for b in batch])

    targets = {}
    for key in batch[0][3]:
        targets[key] = torch.tensor([b[3][key] for b in batch])

    return node_feats, adj_mats, tabulars, targets


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """Single-head graph attention (GAT-style) without PyG."""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h: (B, N, in_dim)
        adj: (B, N, N)
        Returns: (B, N, out_dim)
        """
        B, N, _ = h.shape
        Wh = self.W(h)

        Wh_i = Wh.unsqueeze(2).expand(B, N, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, -1)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))

        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(mask, 0)
        alpha = self.dropout(alpha)

        return torch.bmm(alpha, Wh)


class GraphNeuralNetwork(nn.Module):
    """Multi-layer GAT for processing dynamic graphs."""

    def __init__(self, in_dim=2, hidden_dim=HIDDEN_DIM, n_layers=2, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphAttentionLayer(in_dim, hidden_dim, dropout))
        for _ in range(n_layers - 1):
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, dropout))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats, adj):
        """
        node_feats: (B, N, in_dim)
        adj: (B, N, N)
        Returns: graph_embedding (B, hidden_dim)
        """
        h = node_feats
        for layer in self.layers:
            h = F.elu(layer(h, adj))
        h = self.norm(h)

        graph_emb = h.mean(dim=1)
        return graph_emb, h


class TemporalTransformer(nn.Module):
    """Transformer encoder over GNN output sequences."""

    def __init__(self, input_dim=HIDDEN_DIM, d_model=HIDDEN_DIM,
                 n_heads=N_HEADS, n_layers=2, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        Returns: (B, d_model)
        """
        x = self.proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x[:, -1, :])
        return x


class EVTLoss(nn.Module):
    """
    Extreme Value Theory inspired loss component.

    Uses a learnable GPD-style tail estimator that penalizes
    the model for missing extreme events.
    """

    def __init__(self, hidden_dim, n_tasks=3):
        super().__init__()
        # Learnable threshold (quantile) for tail detection
        self.tail_threshold = nn.Parameter(torch.tensor(0.0))
        # GPD parameters: shape (xi) and scale (sigma) per horizon
        self.gpd_xi = nn.Parameter(torch.zeros(n_tasks))
        self.gpd_sigma = nn.Parameter(torch.ones(n_tasks))
        # Tail risk scoring head
        self.tail_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, n_tasks),
        )

    def forward(self, hidden, extreme_probs):
        """
        hidden: (B, hidden_dim)
        extreme_probs: (B, 3) predicted probabilities for 5d/10d/20d

        Returns: EVT loss scalar
        """
        tail_scores = torch.sigmoid(self.tail_head(hidden))

        # GPD-inspired loss: penalize underestimation of extremes
        xi = self.gpd_xi.abs() + 0.01
        sigma = self.gpd_sigma.abs() + 0.01

        # Log-likelihood of generalized Pareto for tail events
        evt_loss = torch.tensor(0.0, device=hidden.device)

        for i in range(extreme_probs.shape[1]):
            tail_mask = (extreme_probs[:, i] > 0.5).float()
            if tail_mask.sum() > 0:
                excess = tail_scores[:, i].clamp(min=1e-6)
                # GPD log-likelihood (simplified)
                ll = -torch.log(sigma[i]) - (1 + 1 / xi[i]) * torch.log(1 + xi[i] * excess / sigma[i])
                evt_loss = evt_loss - (ll * tail_mask).mean()

        return evt_loss + 0.1 * tail_scores.mean()


class MultiTaskHead(nn.Module):
    """Prediction head for regime, extreme events, and drawdown."""

    def __init__(self, input_dim, n_regimes=5):
        super().__init__()
        self.regime_head = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, n_regimes),
        )
        self.extreme_head = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 3),  # 5d, 10d, 20d
        )
        self.drawdown_head = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 3),  # 5d, 10d, 20d
        )

    def forward(self, x):
        return {
            "regime": self.regime_head(x),
            "extreme": torch.sigmoid(self.extreme_head(x)),
            "maxdd": self.drawdown_head(x),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════

class GreySwan(nn.Module):
    """
    Full Grey-Swan model: GNN + Transformer + EVT + Multi-task.

    Ablation configs:
        11 = GNN + Transformer + EVT (full)
        10 = Transformer + EVT (no graph)
        9  = GNN + EVT (no temporal)
        8  = GNN + Transformer (no EVT)
        7  = GNN only
    """

    def __init__(self, config=11, n_tabular=0, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.config = config
        self.use_gnn = config in [7, 8, 9, 11]
        self.use_transformer = config in [8, 10, 11]
        self.use_evt = config in [9, 10, 11]

        # Tabular feature projection
        if n_tabular > 0:
            self.tab_proj = nn.Sequential(
                nn.Linear(n_tabular, hidden_dim), nn.ReLU(), nn.Dropout(DROPOUT),
            )
        else:
            self.tab_proj = None

        # GNN
        if self.use_gnn:
            self.gnn = GraphNeuralNetwork(in_dim=2, hidden_dim=hidden_dim)
            gnn_out = hidden_dim
        else:
            gnn_out = 2 * N_NODES

        # Temporal fusion dimension
        temp_in = (gnn_out if self.use_gnn else 2 * N_NODES)
        if self.tab_proj is not None:
            temp_in += hidden_dim

        # Transformer
        if self.use_transformer:
            self.transformer = TemporalTransformer(
                input_dim=temp_in, d_model=hidden_dim,
            )
            head_in = hidden_dim
        else:
            head_in = temp_in

        # EVT
        if self.use_evt:
            self.evt = EVTLoss(head_in, n_tasks=3)

        # Prediction head
        self.head = MultiTaskHead(head_in)

        # Regime classifier weights (for class imbalance)
        self.register_buffer("regime_weights", torch.ones(5))

    def forward(self, node_feats, adj, tabular=None):
        """
        node_feats: (B, T, N, 2)
        adj: (B, T, N, N)
        tabular: (B, n_tabular) or None
        """
        B, T, N, F = node_feats.shape
        gnn_outputs = []

        for t in range(T):
            if self.use_gnn:
                gf, _ = self.gnn(node_feats[:, t], adj[:, t])
                gnn_outputs.append(gf)
            else:
                flat = node_feats[:, t].reshape(B, -1)
                gnn_outputs.append(flat)

        # Stack into sequence
        temporal_seq = torch.stack(gnn_outputs, dim=1)

        # Fuse tabular features
        if self.tab_proj is not None and tabular is not None:
            tab_h = self.tab_proj(tabular).unsqueeze(1).expand(-1, T, -1)
            temporal_seq = torch.cat([temporal_seq, tab_h], dim=-1)

        # Transformer
        if self.use_transformer:
            hidden = self.transformer(temporal_seq)
        else:
            hidden = temporal_seq[:, -1, :]

        # Predictions
        preds = self.head(hidden)

        # EVT loss
        evt_loss = torch.tensor(0.0, device=hidden.device)
        if self.use_evt:
            evt_loss = self.evt(hidden, preds["extreme"])

        return preds, evt_loss, hidden


# ═══════════════════════════════════════════════════════════════════════════
# 5. LOSS & TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(preds, targets, model, evt_loss, regime_weights):
    """Multi-task loss with class-weighted regime classification."""
    # Regime classification
    regime_loss = F.cross_entropy(
        preds["regime"], targets["regime"],
        weight=regime_weights.to(targets["regime"].device),
    )

    # Extreme event (binary cross-entropy per horizon)
    extreme_loss = torch.tensor(0.0, device=preds["extreme"].device)
    for i, h in enumerate(["5d", "10d", "20d"]):
        extreme_loss += F.binary_cross_entropy(
            preds["extreme"][:, i], targets[f"extreme_{h}"].float(),
        )
    extreme_loss /= 3.0

    # Drawdown regression (MSE)
    dd_loss = torch.tensor(0.0, device=preds["extreme"].device)
    for i, h in enumerate(["5d", "10d", "20d"]):
        dd_loss += F.mse_loss(preds["maxdd"][:, i], targets[f"maxdd_{h}"].float())
    dd_loss /= 3.0

    # Combined
    total = regime_loss + 2.0 * extreme_loss + 0.5 * dd_loss + 0.1 * evt_loss

    return total, {
        "regime": regime_loss.item(),
        "extreme": extreme_loss.item(),
        "maxdd": dd_loss.item(),
        "evt": evt_loss.item(),
        "total": total.item(),
    }


def train_epoch(model, loader, optimizer, regime_weights):
    model.train()
    total_loss = 0
    n_batches = 0
    loss_accum = {"regime": 0, "extreme": 0, "maxdd": 0, "evt": 0, "total": 0}

    for node_feats, adj, tabular, targets in loader:
        node_feats = node_feats.to(DEVICE)
        adj = adj.to(DEVICE)
        tabular = tabular.to(DEVICE)
        targets = {k: v.to(DEVICE) for k, v in targets.items()}

        optimizer.zero_grad()
        preds, evt_loss, _ = model(node_feats, adj, tabular)
        loss, loss_dict = compute_loss(preds, targets, model, evt_loss, regime_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        for k in loss_accum:
            loss_accum[k] += loss_dict[k]

    return total_loss / max(n_batches, 1), {k: v / max(n_batches, 1) for k, v in loss_accum.items()}


@torch.no_grad()
def evaluate(model, loader, regime_weights):
    model.eval()
    all_preds = {"regime": [], "extreme": [], "maxdd": []}
    all_targets = {"regime": [], "extreme_5d": [], "extreme_10d": [], "extreme_20d": [],
                    "maxdd_5d": [], "maxdd_10d": [], "maxdd_20d": []}
    total_loss = 0
    n_batches = 0

    for node_feats, adj, tabular, targets in loader:
        node_feats = node_feats.to(DEVICE)
        adj = adj.to(DEVICE)
        tabular = tabular.to(DEVICE)
        targets_dev = {k: v.to(DEVICE) for k, v in targets.items()}

        preds, evt_loss, _ = model(node_feats, adj, tabular)
        loss, _ = compute_loss(preds, targets_dev, model, evt_loss, regime_weights)

        total_loss += loss.item()
        n_batches += 1

        all_preds["regime"].append(preds["regime"].cpu().numpy())
        all_preds["extreme"].append(preds["extreme"].cpu().numpy())
        all_preds["maxdd"].append(preds["maxdd"].cpu().numpy())
        all_targets["regime"].append(targets["regime"].numpy())
        for h in ["5d", "10d", "20d"]:
            all_targets[f"extreme_{h}"].append(targets[f"extreme_{h}"].numpy())
            all_targets[f"maxdd_{h}"].append(targets[f"maxdd_{h}"].numpy())

    all_preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    all_targets = {k: np.concatenate(v) for k, v in all_targets.items()}

    return total_loss / max(n_batches, 1), all_preds, all_targets


# ═══════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(preds, targets):
    from sklearn.metrics import (
        average_precision_score, roc_auc_score,
        f1_score, accuracy_score,
    )
    metrics = {}

    # Regime accuracy
    regime_pred = preds["regime"].argmax(axis=1)
    metrics["regime_acc"] = accuracy_score(targets["regime"], regime_pred)
    metrics["regime_f1_macro"] = f1_score(targets["regime"], regime_pred, average="macro", zero_division=0)

    # Extreme event metrics per horizon
    for i, h in enumerate(["5d", "10d", "20d"]):
        y_true = targets[f"extreme_{h}"]
        y_prob = preds["extreme"][:, i]
        y_pred = (y_prob > 0.5).astype(int)

        if y_true.sum() > 0:
            metrics[f"extreme_{h}_prauc"] = average_precision_score(y_true, y_prob)
            metrics[f"extreme_{h}_rocauc"] = roc_auc_score(y_true, y_prob)
            metrics[f"extreme_{h}_f1"] = f1_score(y_true, y_pred, zero_division=0)
            metrics[f"extreme_{h}_recall"] = f1_score(y_true, y_pred, zero_division=0)
        else:
            metrics[f"extreme_{h}_prauc"] = 0.0
            metrics[f"extreme_{h}_rocauc"] = 0.0
            metrics[f"extreme_{h}_f1"] = 0.0
            metrics[f"extreme_{h}_recall"] = 0.0

    # MaxDD MAE
    for i, h in enumerate(["5d", "10d", "20d"]):
        metrics[f"maxdd_{h}_mae"] = np.mean(np.abs(preds["maxdd"][:, i] - targets[f"maxdd_{h}"]))

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(train_losses, val_losses, config, save=True):
    """Plot training and validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, [l["total"] for l in train_losses], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, [l["total"] for l in val_losses], "r--", label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title(f"Grey-Swan Config {config}: Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-component loss
    for key, style, label in [("regime", "g-", "Regime"), ("extreme", "m--", "Extreme"),
                               ("maxdd", "c-.", "MaxDD"), ("evt", "y:", "EVT")]:
        if key in train_losses[0] and key in val_losses[0]:
            axes[1].plot(epochs, [l[key] for l in train_losses], style, label=f"Train {label}", alpha=0.8)
            axes[1].plot(epochs, [l[key] for l in val_losses], style, label=f"Val {label}", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"Grey-Swan Config {config}: Component Losses")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = VIZ_DIR / f"grey_swan_config{config}_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        console.print(f"  Saved: {path}")
    plt.close(fig)


def plot_ablation_chart(results_df, save=True):
    """Bar chart comparing all ablation configurations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = results_df["config"].values
    labels = [f"C{c}" for c in configs]

    # PR-AUC
    axes[0].bar(labels, results_df["extreme_5d_prauc"], color="#3498db", alpha=0.8)
    axes[0].set_title("Extreme 5d PR-AUC")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Regime Accuracy
    axes[1].bar(labels, results_df["regime_acc"], color="#2ecc71", alpha=0.8)
    axes[1].set_title("Regime Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis="y")

    # MaxDD MAE
    axes[2].bar(labels, results_df["maxdd_5d_mae"], color="#e74c3c", alpha=0.8)
    axes[2].set_title("Max Drawdown 5d MAE")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Grey-Swan Ablation Study", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = VIZ_DIR / "grey_swan_ablation_chart.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        console.print(f"  Saved: {path}")
    plt.close(fig)


def plot_prediction_analysis(preds, targets, save=True):
    """Prediction distribution and calibration plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, h in enumerate(["5d", "10d", "20d"]):
        y_true = targets[f"extreme_{h}"]
        y_prob = preds["extreme"][:, i]

        # Distribution
        axes[0, i].hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Normal", color="#2ecc71", density=True)
        axes[0, i].hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Extreme", color="#e74c3c", density=True)
        axes[0, i].set_title(f"Prediction Distribution ({h})")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        # Calibration
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        axes[1, i].plot(prob_pred, prob_true, "o-", label="Grey-Swan")
        axes[1, i].plot([0, 1], [0, 1], "k--", label="Perfect")
        axes[1, i].set_title(f"Calibration ({h})")
        axes[1, i].set_xlabel("Predicted Probability")
        axes[1, i].set_ylabel("Actual Frequency")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle("Grey-Swan Prediction Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = VIZ_DIR / "grey_swan_predictions.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        console.print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_NAMES = {
    7: "GNN only",
    8: "GNN + Transformer",
    9: "GNN + EVT",
    10: "Transformer + EVT",
    11: "GNN + Transformer + EVT (full)",
}


def run_single_config(config, features, regime_df, graph_idx, epochs=30, lr=1e-3):
    """Train and evaluate a single ablation configuration."""
    name = CONFIG_NAMES.get(config, f"Config {config}")
    console.print(f"\n[bold cyan]Config {config}: {name}[/]")
    t0 = time.time()

    # Datasets
    train_ds = GreySwanDataset(features, regime_df, graph_idx, SEQ_LEN, "train")
    val_ds = GreySwanDataset(features, regime_df, graph_idx, SEQ_LEN, "val")
    test_ds = GreySwanDataset(features, regime_df, graph_idx, SEQ_LEN, "test")

    n_tabular = train_ds.tabular_data.shape[1]
    console.print(f"  Tabular features: {n_tabular}")

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    console.print(f"  Train batches: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")

    # Model
    model = GreySwan(config=config, n_tabular=n_tabular).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Parameters: {n_params:,}")

    # Compute class weights from training regime labels
    train_regime = train_ds.target_data["regime"]
    counts = train_regime.value_counts().sort_index()
    weights = len(train_regime) / (len(counts) * counts.values)
    regime_weights = torch.FloatTensor(weights).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training with progress bar
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold cyan]Config {config}[/] training"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("{task.fields[loss]:.4f}"),
        TextColumn("{task.fields[lr]:.2e}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("training", total=epochs, loss=0.0, lr=lr)

        for epoch in range(epochs):
            t_ep = time.time()

            # Train
            tr_loss, tr_dict = train_epoch(model, train_dl, optimizer, regime_weights)
            train_losses.append(tr_dict)

            # Validate
            val_loss, val_preds, val_targets = evaluate(model, val_dl, regime_weights)
            val_losses.append({"total": val_loss, **{k: v for k, v in compute_metrics(val_preds, val_targets).items() if "mae" in k or "acc" in k}})

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            ep_time = time.time() - t_ep
            progress.update(task, advance=1,
                          loss=val_loss, lr=current_lr,
                          description=f"epoch ({ep_time:.1f}s)")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    _, train_preds, train_targets = evaluate(model, train_dl, regime_weights)
    _, val_preds, val_targets = evaluate(model, val_dl, regime_weights)
    _, test_preds, test_targets = evaluate(model, test_dl, regime_weights)

    train_metrics = compute_metrics(train_preds, train_targets)
    val_metrics = compute_metrics(val_preds, val_targets)
    test_metrics = compute_metrics(test_preds, test_targets)

    elapsed = time.time() - t0

    # Visualizations
    plot_training_curves(train_losses, val_losses, config)
    if config == 11:
        plot_prediction_analysis(test_preds, test_targets)

    return {
        "config": config,
        "name": name,
        "params": n_params,
        "train_time": elapsed,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }, test_preds, test_targets


def main():
    parser = argparse.ArgumentParser(description="Grey-Swan Model")
    parser.add_argument("--config", type=int, default=None, help="Single config (7-11)")
    parser.add_argument("--ablation", action="store_true", help="Run all ablation configs")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]Grey-Swan: Spatiotemporal Graph-Transformer[/]\n"
        "Extreme Market Regime Detection",
        border_style="cyan",
    ))

    t_start = time.time()

    features, regime_df, graph_idx = load_data()

    if args.config:
        configs = [args.config]
    elif args.ablation:
        configs = [7, 8, 9, 10, 11]
    else:
        configs = [11]

    all_results = []
    all_preds_11 = None
    all_targets_11 = None

    for config in configs:
        result, test_preds, test_targets = run_single_config(
            config, features, regime_df, graph_idx, args.epochs, args.lr,
        )
        all_results.append(result)
        if config == 11:
            all_preds_11 = test_preds
            all_targets_11 = test_targets

        # Print summary table
        table = Table(title=f"Config {config}: {CONFIG_NAMES.get(config, '')}", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        for k, v in result.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            elif isinstance(v, int):
                table.add_row(k, f"{v:,}")
            elif isinstance(v, str):
                table.add_row(k, v)
        console.print(table)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT / "grey_swan_results.csv", index=False)
    console.print(f"\n  Saved: {OUT / 'grey_swan_results.csv'}")

    if all_preds_11 is not None:
        pred_df = pd.DataFrame({
            "regime_pred": all_preds_11["regime"].argmax(axis=1),
            "regime_true": all_targets_11["regime"],
            "extreme_5d_pred": all_preds_11["extreme"][:, 0],
            "extreme_5d_true": all_targets_11["extreme_5d"],
            "extreme_10d_pred": all_preds_11["extreme"][:, 1],
            "extreme_10d_true": all_targets_11["extreme_10d"],
            "extreme_20d_pred": all_preds_11["extreme"][:, 2],
            "extreme_20d_true": all_targets_11["extreme_20d"],
        })
        pred_df.to_csv(OUT / "grey_swan_predictions.csv", index=False)
        console.print(f"  Saved: {OUT / 'grey_swan_predictions.csv'}")

    if len(all_results) > 1:
        plot_ablation_chart(results_df)

    total_time = time.time() - t_start
    console.print(f"\n[bold green]Complete in {total_time:.1f}s[/]")


if __name__ == "__main__":
    main()
