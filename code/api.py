"""
Grey-Swan REST API & Dashboard Server
======================================
FastAPI backend serving model results, regime data, and predictions.

Usage:
    python code/api.py                    # start on localhost:8000
    python code/api.py --port 8080        # custom port

Endpoints:
    GET /                          -> Dashboard (static HTML)
    GET /api/overview              -> Project summary stats
    GET /api/regimes               -> Regime labels + timeline data
    GET /api/regimes/timeline      -> Regime transitions over time
    GET /api/regimes/distribution  -> Regime class distribution
    GET /api/predictions           -> Grey-Swan predictions
    GET /api/predictions/latest    -> Most recent predictions
    GET /api/models                -> Baseline model comparison
    GET /api/models/ablation       -> Ablation study results
    GET /api/graphs                -> Graph snapshot metadata
    GET /api/graphs/{date}         -> Single graph snapshot
    GET /api/features/summary      -> Feature statistics
    GET /api/visualizations        -> List of available viz images
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
GRAPH_DIR = DATA / "dynamic_graphs"
VIZ_DIR = DATA / "graph_snapshots"

app = FastAPI(title="Grey-Swan API", version="1.0.0")


def load_csv(name):
    path = DATA / name
    if path.exists():
        return pd.read_csv(path)
    return None


def load_parquet(name):
    path = DATA / name
    if path.exists():
        return pd.read_parquet(path)
    return None


def serve_dashboard():
    html_path = ROOT / "code" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


app.get("/", response_class=HTMLResponse)(lambda: serve_dashboard())


@app.get("/api/overview")
def get_overview():
    master = load_parquet("master_dataset.parquet")
    features = load_parquet("features_dataset.parquet")
    regimes = load_parquet("regime_labels.parquet")
    predictions = load_csv("grey_swan_predictions.csv")
    model_results = load_csv("grey_swan_results.csv")
    baseline_results = load_csv("model_results.csv")

    graph_files = list(GRAPH_DIR.glob("graph_*.npz")) if GRAPH_DIR.exists() else []

    latest_regime = None
    latest_date = None
    if regimes is not None and "regime" in regimes.columns:
        regime_col = regimes["regime"]
        latest_regime = int(regime_col.iloc[-1])
        latest_date = str(regimes.index[-1]) if hasattr(regimes.index, "date") else str(regime_col.index[-1])

    best_prauc = None
    if model_results is not None and "test_extreme_5d_prauc" in model_results.columns:
        best_row = model_results.loc[model_results["test_extreme_5d_prauc"].idxmax()]
        best_prauc = float(best_row["test_extreme_5d_prauc"])

    return {
        "project": "Grey-Swan",
        "description": "Spatiotemporal Graph-Transformer for Extreme Market Regime Detection",
        "master_rows": int(master.shape[0]) if master is not None else 0,
        "master_cols": int(master.shape[1]) if master is not None else 0,
        "feature_count": int(features.shape[1]) if features is not None else 0,
        "graph_snapshots": len(graph_files),
        "current_regime": latest_regime,
        "latest_date": latest_date,
        "best_prauc_5d": best_prauc,
        "has_predictions": predictions is not None,
        "has_baselines": baseline_results is not None,
        "has_grey_swan": model_results is not None,
    }


@app.get("/api/regimes")
def get_regimes():
    df = load_parquet("regime_labels.parquet")
    if df is None:
        return {"error": "No regime data"}
    df = df.reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    cols = ["date", "regime", "vix_level", "term_spread", "drawdown"]
    cols = [c for c in cols if c in df.columns]
    sample = df[cols].tail(500).copy()
    sample["date"] = sample["date"].astype(str)
    records = sample.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                rec[k] = None
    return records


@app.get("/api/regimes/timeline")
def get_regime_timeline():
    df = load_parquet("regime_labels.parquet")
    if df is None:
        return {"error": "No regime data"}
    df = df.reset_index()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").resample("ME")["regime"].agg(lambda x: x.mode()[0] if len(x) > 0 else 0)
    monthly = monthly.reset_index()
    monthly = monthly.tail(60)
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m")
    return {"dates": monthly["date"].tolist(), "regimes": monthly["regime"].tolist()}


@app.get("/api/regimes/distribution")
def get_regime_distribution():
    df = load_parquet("regime_labels.parquet")
    if df is None:
        return {"error": "No regime data"}
    counts = df["regime"].value_counts().sort_index()
    labels = {0: "Normal", 1: "Elevated-Vol", 2: "Stress", 3: "Transition", 4: "Extreme"}
    return {
        "labels": [labels.get(i, str(i)) for i in counts.index],
        "counts": counts.values.tolist(),
        "percentages": (counts / counts.sum() * 100).round(1).tolist(),
    }


@app.get("/api/predictions")
def get_predictions():
    df = load_csv("grey_swan_predictions.csv")
    if df is None:
        return {"error": "No predictions. Run grey_swan_model.py first."}
    return df.tail(200).to_dict(orient="records")


@app.get("/api/predictions/latest")
def get_latest_predictions():
    df = load_csv("grey_swan_predictions.csv")
    if df is None:
        return {"error": "No predictions"}
    last = df.iloc[-1]
    return {
        "regime_pred": int(last.get("regime_pred", 0)),
        "regime_true": int(last.get("regime_true", 0)),
        "extreme_5d_prob": round(float(last.get("extreme_5d_pred", 0)), 4),
        "extreme_10d_prob": round(float(last.get("extreme_10d_pred", 0)), 4),
        "extreme_20d_prob": round(float(last.get("extreme_20d_pred", 0)), 4),
        "extreme_5d_true": int(last.get("extreme_5d_true", 0)),
    }


@app.get("/api/models")
def get_model_comparison():
    df = load_csv("model_results.csv")
    if df is not None:
        df = df.fillna(0)
        for col in df.columns:
            if df[col].dtype in ["float64", "float32"]:
                df[col] = df[col].replace([np.inf, -np.inf], 0).round(4)
        return df.to_dict(orient="records")
    gs = load_csv("grey_swan_results.csv")
    if gs is not None:
        gs = gs.fillna(0)
        for col in gs.columns:
            if gs[col].dtype in ["float64", "float32"]:
                gs[col] = gs[col].replace([np.inf, -np.inf], 0).round(4)
        return gs.to_dict(orient="records")
    return {"error": "No model results"}


@app.get("/api/models/ablation")
def get_ablation():
    df = load_csv("grey_swan_results.csv")
    if df is None:
        return {"error": "No ablation results"}
    cols = ["config", "name", "params", "train_time"]
    metric_cols = [c for c in df.columns if "test_" in c]
    cols.extend(metric_cols)
    cols = [c for c in cols if c in df.columns]
    result = df[cols].copy()
    result = result.fillna(0)
    for col in result.columns:
        if result[col].dtype in ["float64", "float32"]:
            result[col] = result[col].replace([np.inf, -np.inf], 0).round(4)
    return result.to_dict(orient="records")


@app.get("/api/graphs")
def get_graph_index():
    idx_path = GRAPH_DIR / "graph_index.csv"
    if not idx_path.exists():
        return {"error": "No graph data"}
    df = pd.read_csv(idx_path)
    df["date"] = df["date"].astype(str)
    return {
        "total": len(df),
        "avg_edges": round(float(df["n_edges"].mean()), 1),
        "avg_density": round(float(df["density"].mean()), 3),
        "regime_distribution": df["regime"].value_counts().sort_index().to_dict(),
        "snapshots": df.tail(50).to_dict(orient="records"),
    }


@app.get("/api/graphs/{date_str}")
def get_graph_snapshot(date_str: str):
    path = GRAPH_DIR / f"graph_{date_str}.npz"
    if not path.exists():
        return {"error": f"Snapshot {date_str} not found"}
    data = np.load(path, allow_pickle=True)
    edges = data["edges"].tolist() if len(data["edges"]) > 0 else []
    return {
        "date": date_str,
        "nodes": data["nodes"].tolist(),
        "edges": edges,
        "edge_weights": data["edge_weights"].tolist(),
        "n_edges": len(edges),
        "regime": int(data["regime"].item()),
        "node_returns": data["node_returns"].tolist(),
        "node_vol": data["node_vol"].tolist(),
    }


@app.get("/api/features/summary")
def get_feature_summary():
    df = load_parquet("features_dataset.parquet")
    if df is None:
        return {"error": "No feature data"}
    stats = {
        "total_features": int(df.shape[1]),
        "total_rows": int(df.shape[0]),
        "date_range": [str(df.index[0]), str(df.index[-1])],
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "top_categories": {},
    }
    prefix_counts = {}
    for col in df.columns:
        prefix = col.split("_")[0] if "_" in col else col
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    stats["top_categories"] = dict(sorted(prefix_counts.items(), key=lambda x: -x[1])[:10])
    return stats


@app.get("/api/visualizations")
def get_visualizations():
    viz = {}
    if VIZ_DIR.exists():
        for f in sorted(VIZ_DIR.glob("*.png")):
            key = f.stem
            viz[key] = f"/api/visualizations/{key}"
    return viz


@app.get("/api/visualizations/{name}")
def get_visualization(name: str):
    path = VIZ_DIR / f"{name}.png"
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/query")
def query_event(body: dict = Body(...)):
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from text_analyzer import analyze_event
        result = analyze_event(text)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  Grey-Swan API running at http://{args.host}:{args.port}")
    print(f"  Dashboard: http://{args.host}:{args.port}/")
    print(f"  API docs:  http://{args.host}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
