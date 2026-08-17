"""
Grey-Swan Regime Labeling & Dynamic Graph Construction
=======================================================
Defines market regime targets and builds time-varying asset correlation
graphs for GNN input.

Usage:
    python code/regime_graph.py              # full pipeline + visuals
    python code/regime_graph.py --no-viz     # skip visualizations

Output:
    data/processed/regime_labels.parquet
    data/processed/dynamic_graphs/           # per-window adjacency matrices
    data/processed/graph_snapshots/          # visualization PNGs
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
GRAPH_DIR = OUT / "dynamic_graphs"
VIZ_DIR = OUT / "graph_snapshots"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# REGIME TAXONOMY (from README section 2)
# ═══════════════════════════════════════════════════════════════════════════
#
# 0 = Normal           : VIX < 20, drawdown < -10%, term spread > 0
# 1 = Elevated-Vol     : VIX 20-30 OR drawdown -10% to -15%
# 2 = Stress           : VIX 30-40 OR drawdown -15% to -25% OR term spread inverted
# 3 = Transition       : Rapid change: VIX jump > 5 in 5d OR corr spike
# 4 = Extreme/Crisis   : VIX > 40 OR drawdown < -25% OR COVID-scale shock
#
# Forward targets: 5/10/20-day extreme labels (regime >= 3 within horizon)

# Key assets for graph construction
GRAPH_NODES = [
    "yf_sp500_close", "yf_nasdaq100_close", "yf_dow_jones_close",
    "yf_vix_close", "yf_gold_close", "yf_crude_oil_close",
    "yf_eurusd_close", "yf_usdjpy_close",
    "fred_10y_yield", "fred_3m_yield", "fred_30y_yield",
    "yf_aapl_close", "yf_msft_close", "yf_nvda_close",
    "cg_bitcoin_close", "cg_ethereum_close",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. REGIME LABELING
# ═══════════════════════════════════════════════════════════════════════════

def compute_drawdown(prices: pd.Series) -> pd.Series:
    """Compute running drawdown from peak."""
    peak = prices.expanding(min_periods=1).max()
    return (prices - peak) / peak


def label_regimes(features: pd.DataFrame) -> pd.DataFrame:
    """
    Assign regime labels based on multi-indicator thresholds.
    Returns DataFrame with regime, drawdown, forward targets.
    """
    console.print("  [cyan]Computing drawdowns...[/]")

    sp500 = features["yf_sp500_close"].copy()
    drawdown = compute_drawdown(sp500)

    # VIX
    vix = features.get("yf_vix_close", features.get("cboe_vix_close", pd.Series(dtype=float, index=features.index)))

    # Term spread
    has_term = "fred_10y_yield" in features.columns and "fred_3m_yield" in features.columns
    if has_term:
        term_spread = features["fred_10y_yield"] - features["fred_3m_yield"]
    else:
        term_spread = pd.Series(0, index=features.index)

    # VIX rate of change (5-day)
    vix_roc = vix.diff(5)

    # Rolling correlation spike (SP500 vs VIX, 20d correlation collapse)
    if "yf_sp500_close" in features.columns and vix is not None and len(vix) > 0:
        sp_ret = np.log(sp500 / sp500.shift(1))
        vix_ret = np.log(vix / vix.shift(1))
        corr_20d = sp_ret.rolling(20).corr(vix_ret)
        # Strong negative correlation = normal; collapse toward 0 = stress
        corr_stress = 1.0 + corr_20d  # ranges 0 (perfect neg) to 2 (perfect pos)
    else:
        corr_stress = pd.Series(1.0, index=features.index)

    # --- Regime assignment ---
    console.print("  [cyan]Assigning regime labels...[/]")
    regime = pd.Series(0, index=features.index, name="regime")

    # Default: Normal (0)
    # Elevated-Vol (1)
    mask_ev = (vix >= 20) & (vix < 30) | ((drawdown <= -0.10) & (drawdown > -0.15))
    regime[mask_ev] = 1

    # Stress (2)
    mask_stress = (
        ((vix >= 30) & (vix < 40)) |
        ((drawdown <= -0.15) & (drawdown > -0.25)) |
        (term_spread < 0)
    )
    regime[mask_stress] = 2

    # Transition (3) -- rapid deterioration
    mask_trans = (
        (vix_roc > 5) |  # VIX jumped 5+ points in 5 days
        ((drawdown <= -0.10) & (drawdown.diff(5) < -0.05)) |  # drawdown accelerating
        (corr_stress > 1.7)  # correlation collapse
    )
    regime[mask_trans & (regime < 3)] = 3

    # Extreme/Crisis (4)
    mask_extreme = (
        (vix >= 40) |
        (drawdown <= -0.25) |
        (vix_roc > 15)  # extreme VIX spike
    )
    regime[mask_extreme] = 4

    # --- Forward-looking targets ---
    console.print("  [cyan]Computing forward targets...[/]")
    extreme_flag = (regime >= 3).astype(int)

    fwd_targets = pd.DataFrame(index=features.index)
    for horizon, name in [(5, "5d"), (10, "10d"), (20, "20d")]:
        # Will an extreme event occur within the next N days?
        fwd_targets[f"target_extreme_{name}"] = extreme_flag.rolling(horizon).max().shift(-horizon).fillna(0).astype(int)

    # Expected max drawdown within horizon
    for horizon, name in [(5, "5d"), (10, "10d"), (20, "20d")]:
        fwd_targets[f"target_maxdd_{name}"] = drawdown.rolling(horizon).min().shift(-horizon)

    # Build output
    out = pd.DataFrame({
        "regime": regime,
        "drawdown": drawdown,
        "vix_level": vix,
        "vix_roc_5d": vix_roc,
        "term_spread": term_spread,
        "corr_stress": corr_stress,
    }, index=features.index)

    out = pd.concat([out, fwd_targets], axis=1)

    # Regime distribution
    counts = regime.value_counts().sort_index()
    total = len(regime)
    labels = {0: "Normal", 1: "Elevated-Vol", 2: "Stress", 3: "Transition", 4: "Extreme"}
    console.print("\n  [bold]Regime Distribution:[/]")
    for r, n in counts.items():
        pct = n / total * 100
        bar = "#" * int(pct / 2)
        console.print(f"    {labels.get(r, r):15s} {n:>6,} ({pct:5.1f}%) {bar}")

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. DYNAMIC GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_dynamic_graphs(features: pd.DataFrame, regime_df: pd.DataFrame,
                          window: int = 60, step: int = 20,
                          corr_threshold: float = 0.3) -> list[dict]:
    """
    Build time-varying correlation graphs at regular intervals.

    Each snapshot:
    - Nodes: asset features
    - Edges: pairwise correlations above threshold
    - Edge weights: correlation strength
    - Node features: returns + volatility at that time

    Returns list of graph snapshots with metadata.
    """
    console.print(f"\n  [cyan]Building dynamic graphs (window={window}, step={step})...[/]")

    # Filter to available nodes
    available = [n for n in GRAPH_NODES if n in features.columns]
    console.print(f"  {len(available)} nodes available")

    # Compute log returns for graph nodes
    returns = pd.DataFrame(index=features.index)
    for col in available:
        series = features[col].replace(0, np.nan)
        returns[col] = np.log(series / series.shift(1))

    returns = returns.dropna(how="all")

    # Build graph snapshots
    snapshots = []
    dates = returns.index[window::step]

    for date in dates:
        loc = returns.index.get_loc(date)
        start = max(0, loc - window)
        window_data = returns.iloc[start:loc + 1]

        if len(window_data) < window // 2:
            continue

        # Correlation matrix
        corr = window_data.corr()
        corr = corr.fillna(0)

        # Build edge list (upper triangle, above threshold)
        nodes = list(corr.columns)
        edges = []
        edge_weights = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                w = corr.iloc[i, j]
                if abs(w) >= corr_threshold:
                    edges.append((i, j))
                    edge_weights.append(float(w))

        # Node features: latest returns + 20d volatility
        node_returns = returns.loc[date].reindex(available).fillna(0).values
        node_vol = returns.loc[:date].tail(20).std().reindex(available).fillna(0).values

        snapshot = {
            "date": date,
            "nodes": available,
            "edges": edges,
            "edge_weights": edge_weights,
            "node_returns": node_returns,
            "node_vol": node_vol,
            "corr_matrix": corr.values,
            "n_edges": len(edges),
            "density": len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0,
            "mean_corr": np.mean([abs(w) for w in edge_weights]) if edge_weights else 0,
            "regime": regime_df.loc[date, "regime"] if date in regime_df.index else 0,
        }
        snapshots.append(snapshot)

    console.print(f"  {len(snapshots)} graph snapshots built")
    console.print(f"  Date range: {snapshots[0]['date'].date()} to {snapshots[-1]['date'].date()}")
    avg_edges = np.mean([s["n_edges"] for s in snapshots])
    avg_density = np.mean([s["density"] for s in snapshots])
    console.print(f"  Avg edges/snapshot: {avg_edges:.0f}, Avg density: {avg_density:.3f}")

    return snapshots


def save_graph_snapshots(snapshots: list[dict]):
    """Save graph snapshots as individual .npz files for GNN loading."""
    for i, snap in enumerate(snapshots):
        date_str = snap["date"].strftime("%Y%m%d")
        path = GRAPH_DIR / f"graph_{date_str}.npz"
        np.savez_compressed(
            path,
            nodes=np.array(snap["nodes"]),
            edges=np.array(snap["edges"]) if snap["edges"] else np.empty((0, 2), dtype=int),
            edge_weights=np.array(snap["edge_weights"]) if snap["edge_weights"] else np.empty(0),
            node_returns=snap["node_returns"],
            node_vol=snap["node_vol"],
            corr_matrix=snap["corr_matrix"],
            regime=snap["regime"],
        )

    # Save metadata index
    meta = pd.DataFrame([{
        "date": s["date"],
        "n_edges": s["n_edges"],
        "density": s["density"],
        "mean_corr": s["mean_corr"],
        "regime": s["regime"],
    } for s in snapshots])
    meta.to_csv(GRAPH_DIR / "graph_index.csv", index=False)
    console.print(f"  Saved {len(snapshots)} graph snapshots to {GRAPH_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    0: "#2ecc71",  # Normal: green
    1: "#f39c12",  # Elevated-Vol: orange
    2: "#e67e22",  # Stress: dark orange
    3: "#e74c3c",  # Transition: red
    4: "#8e44ad",  # Extreme: purple
}
REGIME_LABELS = {0: "Normal", 1: "Elevated-Vol", 2: "Stress", 3: "Transition", 4: "Extreme"}


def plot_regime_timeline(regime_df: pd.DataFrame, features: pd.DataFrame):
    """Plot regime timeline with VIX and drawdown."""
    console.print("  [cyan]Plotting regime timeline...[/]")
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle("Grey-Swan: Market Regime Timeline", fontsize=14, fontweight="bold")

    dates = regime_df.index

    # Panel 1: Regime color bands + SP500
    ax1 = axes[0]
    if "yf_sp500_close" in features.columns:
        sp500 = features.loc[dates, "yf_sp500_close"].dropna()
        ax1.plot(sp500.index, sp500.values, color="#2c3e50", linewidth=0.8, label="S&P 500")
    ax1.set_ylabel("S&P 500", fontsize=10)
    ax1.legend(loc="upper left")

    # Color regime bands
    prev_regime = None
    start_idx = 0
    for i, (date, regime) in enumerate(regime_df["regime"].items()):
        if regime != prev_regime or i == len(regime_df) - 1:
            if prev_regime is not None:
                ax1.axvspan(dates[start_idx], date, alpha=0.25,
                           color=REGIME_COLORS.get(prev_regime, "#95a5a6"))
            start_idx = i
            prev_regime = regime

    # Legend
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=REGIME_LABELS[r])
               for r in sorted(REGIME_COLORS.keys())]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=5)

    # Panel 2: VIX
    ax2 = axes[1]
    if "vix_level" in regime_df.columns:
        vix = regime_df["vix_level"].dropna()
        ax2.plot(vix.index, vix.values, color="#e74c3c", linewidth=0.8)
        ax2.axhline(y=20, color="#f39c12", linestyle="--", alpha=0.5, label="VIX=20")
        ax2.axhline(y=30, color="#e74c3c", linestyle="--", alpha=0.5, label="VIX=30")
        ax2.axhline(y=40, color="#8e44ad", linestyle="--", alpha=0.5, label="VIX=40")
    ax2.set_ylabel("VIX", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)

    # Panel 3: Drawdown
    ax3 = axes[2]
    dd = regime_df["drawdown"].dropna()
    ax3.fill_between(dd.index, dd.values * 100, 0, color="#e74c3c", alpha=0.4)
    ax3.plot(dd.index, dd.values * 100, color="#c0392b", linewidth=0.5)
    ax3.axhline(y=-10, color="#f39c12", linestyle="--", alpha=0.5)
    ax3.axhline(y=-25, color="#8e44ad", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Drawdown (%)", fontsize=10)
    ax3.set_xlabel("Date", fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = VIZ_DIR / "regime_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved: {path}")


def plot_correlation_heatmap(snapshots: list[dict], regime_df: pd.DataFrame):
    """Plot correlation heatmaps for representative dates from each regime."""
    console.print("  [cyan]Plotting correlation heatmaps...[/]")

    # Pick one snapshot per regime (the one with highest density)
    regime_shots = {}
    for snap in snapshots:
        r = int(snap["regime"])
        if r not in regime_shots or snap["density"] > regime_shots[r]["density"]:
            regime_shots[r] = snap

    n_regimes = len(regime_shots)
    if n_regimes == 0:
        return

    fig, axes = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 5))
    if n_regimes == 1:
        axes = [axes]

    fig.suptitle("Correlation Structure by Regime", fontsize=14, fontweight="bold")

    for idx, (regime, snap) in enumerate(sorted(regime_shots.items())):
        ax = axes[idx]
        corr = snap["corr_matrix"]
        nodes = snap["nodes"]
        # Shorten node names
        short_names = [n.replace("yf_", "").replace("close", "").replace("_", " ").strip()
                       for n in nodes]

        im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(nodes)))
        ax.set_yticks(range(len(nodes)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(short_names, fontsize=6)
        ax.set_title(f"{REGIME_LABELS.get(regime, regime)}\n"
                     f"{snap['date'].strftime('%Y-%m-%d')}\n"
                     f"Edges: {snap['n_edges']}, Density: {snap['density']:.3f}",
                     fontsize=9)

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Correlation")

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    path = VIZ_DIR / "correlation_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved: {path}")


def plot_graph_structure(snapshots: list[dict]):
    """Plot network graph for a few representative snapshots."""
    console.print("  [cyan]Plotting graph structures...[/]")

    # Pick snapshots: one normal, one stress, one crisis
    targets = {}
    for snap in snapshots:
        r = int(snap["regime"])
        if r == 0 and 0 not in targets:
            targets[0] = snap
        elif r == 2 and 2 not in targets:
            targets[2] = snap
        elif r == 4 and 4 not in targets:
            targets[4] = snap

    # Fallback: use any available
    if len(targets) < 2:
        for snap in snapshots:
            r = int(snap["regime"])
            if r not in targets:
                targets[r] = snap
            if len(targets) >= 3:
                break

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
    if n == 1:
        axes = [axes]
    fig.suptitle("Asset Correlation Graph by Regime", fontsize=14, fontweight="bold")

    for idx, (regime, snap) in enumerate(sorted(targets.items())):
        ax = axes[idx]
        G = nx.Graph()

        # Add nodes
        for i, node in enumerate(snap["nodes"]):
            short = node.replace("yf_", "").replace("close", "").replace("_", " ").strip()
            G.add_node(i, label=short)

        # Add edges
        for (i, j), w in zip(snap["edges"], snap["edge_weights"]):
            G.add_edge(i, j, weight=abs(w), signed_weight=w)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=2.0)

        # Color edges by sign
        pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["signed_weight"] > 0]
        neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["signed_weight"] <= 0]
        pos_weights = [G[u][v]["weight"] for u, v in pos_edges]
        neg_weights = [G[u][v]["weight"] for u, v in neg_edges]

        # Draw
        node_colors = [REGIME_COLORS.get(regime, "#95a5a6")] * len(G.nodes)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color=node_colors,
                               edgecolors="#2c3e50", linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={i: G.nodes[i]["label"] for i in G.nodes},
                                font_size=7, font_weight="bold")

        if pos_edges:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=pos_edges,
                                   width=[w * 3 for w in pos_weights],
                                   edge_color="#2ecc71", alpha=0.6)
        if neg_edges:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=neg_edges,
                                   width=[w * 3 for w in neg_weights],
                                   edge_color="#e74c3c", alpha=0.6, style="dashed")

        ax.set_title(f"{REGIME_LABELS.get(regime, regime)}\n"
                     f"{snap['date'].strftime('%Y-%m-%d')}\n"
                     f"Edges: {snap['n_edges']}, Density: {snap['density']:.3f}",
                     fontsize=10)
        ax.axis("off")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="#2ecc71", linewidth=2, label="Positive correlation"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=2, linestyle="--", label="Negative correlation"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower left", fontsize=8)

    plt.tight_layout()
    path = VIZ_DIR / "graph_structure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved: {path}")


def plot_graph_density_timeseries(snapshots: list[dict]):
    """Plot graph density and edge count over time."""
    console.print("  [cyan]Plotting graph density time series...[/]")
    dates = [s["date"] for s in snapshots]
    densities = [s["density"] for s in snapshots]
    n_edges = [s["n_edges"] for s in snapshots]
    regimes = [s["regime"] for s in snapshots]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Dynamic Graph Properties Over Time", fontsize=14, fontweight="bold")

    # Color by regime
    for i in range(len(dates) - 1):
        ax1.axvspan(dates[i], dates[i + 1], alpha=0.3,
                   color=REGIME_COLORS.get(regimes[i], "#95a5a6"))
        ax2.axvspan(dates[i], dates[i + 1], alpha=0.3,
                   color=REGIME_COLORS.get(regimes[i], "#95a5a6"))

    ax1.plot(dates, densities, color="#2c3e50", linewidth=0.8)
    ax1.set_ylabel("Graph Density")
    ax1.set_title("Edge Density (higher = more correlated)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates, n_edges, color="#2c3e50", linewidth=0.8)
    ax2.set_ylabel("Number of Edges")
    ax2.set_xlabel("Date")
    ax2.set_title("Active Edges (corr > 0.3)")
    ax2.grid(True, alpha=0.3)

    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=REGIME_LABELS[r])
               for r in sorted(REGIME_COLORS.keys())]
    ax1.legend(handles=patches, loc="upper right", fontsize=8, ncol=5)

    plt.tight_layout()
    path = VIZ_DIR / "graph_density_ts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run(no_viz: bool = False):
    t0 = time.time()
    console.print(Panel.fit(
        "[bold cyan]Grey-Swan Regime Labeling & Dynamic Graph Construction[/]",
        border_style="cyan"
    ))

    # Load features
    features_path = OUT / "features_dataset.parquet"
    if not features_path.exists():
        console.print("[red]features_dataset.parquet not found. Run feature_engineering.py first.[/]")
        sys.exit(1)

    console.print("\n  Loading features dataset...")
    features = pd.read_parquet(features_path)
    console.print(f"  {features.shape[0]:,} rows x {features.shape[1]} cols")

    # 1. Regime labeling
    console.print("\n[bold]Step 1/3: Regime Labeling[/]")
    regime_df = label_regimes(features)

    # Save regime labels
    regime_path = OUT / "regime_labels.parquet"
    regime_df.to_parquet(regime_path, engine="pyarrow")
    console.print(f"\n  Saved: {regime_path}")

    # 2. Dynamic graph construction
    console.print("\n[bold]Step 2/3: Dynamic Graph Construction[/]")
    snapshots = build_dynamic_graphs(features, regime_df, window=60, step=20, corr_threshold=0.3)
    save_graph_snapshots(snapshots)

    # 3. Visualizations
    if not no_viz:
        console.print("\n[bold]Step 3/3: Visualizations[/]")
        plot_regime_timeline(regime_df, features)
        plot_correlation_heatmap(snapshots, regime_df)
        plot_graph_structure(snapshots)
        plot_graph_density_timeseries(snapshots)
    else:
        console.print("\n[bold]Step 3/3: Skipped (--no-viz)[/]")

    elapsed = time.time() - t0
    console.print(Panel.fit(
        f"[bold green]Complete in {elapsed:.1f}s[/]\n"
        f"Regimes: {regime_df['regime'].nunique()} classes\n"
        f"Graph snapshots: {len(snapshots)}\n"
        f"Date range: {snapshots[0]['date'].date()} to {snapshots[-1]['date'].date()}",
        border_style="green"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime Labeling & Dynamic Graph Construction")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    args = parser.parse_args()
    run(no_viz=args.no_viz)
