"""
Grey-Swan Feature Engineering
==============================
Computes derived features from the master dataset for regime detection.

Usage:
    python code/feature_engineering.py

Output:
    data/processed/features_dataset.parquet
    data/processed/features_dataset.csv
    data/processed/feature_inventory.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Feature windows ──────────────────────────────────────────────────────
WINDOWS = [5, 10, 20, 60]


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOG RETURNS
# ═══════════════════════════════════════════════════════════════════════════

def compute_log_returns(master: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns for all Close price columns."""
    console.print("  [cyan]Log returns...[/]")
    close_cols = [c for c in master.columns if c.endswith("_close")
                  and not c.endswith("_missing") and not c.endswith("_anomaly")]
    out = pd.DataFrame(index=master.index)
    for col in close_cols:
        series = master[col].replace(0, np.nan)
        out[f"{col}_ret1"] = np.log(series / series.shift(1))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. ROLLING VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_volatility(master: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling annualized volatility for all Close price columns."""
    console.print("  [cyan]Rolling volatility...[/]")
    close_cols = [c for c in master.columns if c.endswith("_close")
                  and not c.endswith("_missing") and not c.endswith("_anomaly")]
    out = pd.DataFrame(index=master.index)
    for col in close_cols:
        series = master[col].replace(0, np.nan)
        daily_ret = np.log(series / series.shift(1))
        for w in WINDOWS:
            out[f"{col}_vol{w}"] = daily_ret.rolling(w, min_periods=max(w // 2, 2)).std() * np.sqrt(252)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. MOMENTUM (rolling returns)
# ═══════════════════════════════════════════════════════════════════════════

def compute_momentum(master: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum as rolling cumulative log returns at multiple horizons."""
    console.print("  [cyan]Momentum...[/]")
    close_cols = [c for c in master.columns if c.endswith("_close")
                  and not c.endswith("_missing") and not c.endswith("_anomaly")]
    out = pd.DataFrame(index=master.index)
    for col in close_cols:
        series = master[col].replace(0, np.nan)
        log_price = np.log(series)
        for w in WINDOWS:
            out[f"{col}_mom{w}"] = log_price - log_price.shift(w)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. YIELD CURVE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_yield_curve(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yield curve spreads and dynamics:
    - Term spread: 10Y - 3M
    - Long-term spread: 30Y - 10Y
    - Spread changes (1d, 5d, 20d)
    - Inversion flag (term spread < 0)
    """
    console.print("  [cyan]Yield curve features...[/]")
    out = pd.DataFrame(index=master.index)

    has_10y = "fred_10y_yield" in master.columns
    has_30y = "fred_30y_yield" in master.columns
    has_3m = "fred_3m_yield" in master.columns

    if has_10y and has_3m:
        out["yc_term_spread"] = master["fred_10y_yield"] - master["fred_3m_yield"]
        out["yc_term_spread_5d"] = out["yc_term_spread"] - out["yc_term_spread"].shift(5)
        out["yc_term_spread_20d"] = out["yc_term_spread"] - out["yc_term_spread"].shift(20)
        out["yc_inverted"] = (out["yc_term_spread"] < 0).astype(int)

    if has_30y and has_10y:
        out["yc_long_spread"] = master["fred_30y_yield"] - master["fred_10y_yield"]
        out["yc_long_spread_5d"] = out["yc_long_spread"] - out["yc_long_spread"].shift(5)

    if has_10y:
        out["yc_10y_5d"] = master["fred_10y_yield"] - master["fred_10y_yield"].shift(5)
        out["yc_10y_20d"] = master["fred_10y_yield"] - master["fred_10y_yield"].shift(20)

    # Treasury yield curve spreads (if available)
    treasury_spread_cols = [c for c in master.columns if c.startswith("treasury_") and "yield" not in c]
    if treasury_spread_cols:
        for c in treasury_spread_cols:
            series = master[c]
            out[f"{c}_ret5"] = series - series.shift(5)
            out[f"{c}_ret20d"] = series - series.shift(20)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. VIX DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_vix_dynamics(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIX-specific features:
    - Level buckets
    - Rate of change
    - High/low regime flags
    - VIX term structure implied from VIX vs SP500 vol
    """
    console.print("  [cyan]VIX dynamics...[/]")
    out = pd.DataFrame(index=master.index)

    # Use Yahoo VIX as primary, CBOE as fallback
    vix_col = "yf_vix_close" if "yf_vix_close" in master.columns else "cboe_vix_close"
    if vix_col not in master.columns:
        return out

    vix = master[vix_col]

    # Level buckets
    out["vix_level"] = vix
    out["vix_high_flag"] = (vix > 30).astype(int)
    out["vix_extreme_flag"] = (vix > 40).astype(int)
    out["vix_low_flag"] = (vix < 15).astype(int)

    # Rate of change
    out["vix_roc_1d"] = vix.pct_change(1)
    out["vix_roc_5d"] = vix.pct_change(5)
    out["vix_roc_20d"] = vix.pct_change(20)

    # VIX regime: 0=low, 1=normal, 2=high, 3=extreme
    regime = pd.Series(1, index=master.index, name="vix_regime")
    regime[vix < 15] = 0
    regime[vix > 30] = 2
    regime[vix > 40] = 3
    out["vix_regime"] = regime

    # VIX momentum (is VIX itself trending?)
    log_vix = np.log(vix.replace(0, np.nan))
    for w in [5, 10, 20]:
        out[f"vix_mom{w}"] = log_vix - log_vix.shift(w)

    # Volatility of volatility
    out["vix_vol_of_vol_20d"] = vix.pct_change().rolling(20, min_periods=10).std()

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6. CROSS-ASSET CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_correlations(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations between key asset classes:
    - SP500 vs VIX (fear gauge)
    - SP500 vs Gold (risk-on vs safe haven)
    - SP500 vs USD/JPY (risk appetite)
    - Gold vs USD/JPY
    - SP500 vs Crude Oil
    - SP500 vs EUR/USD
    """
    console.print("  [cyan]Cross-asset correlations...[/]")
    out = pd.DataFrame(index=master.index)

    # Use log returns if available, otherwise use close price pct_change
    ret_cols = {}
    candidates = {
        "sp500": "yf_sp500_close",
        "vix": "yf_vix_close",
        "gold": "yf_gold_close",
        "usdjpy": "yf_usdjpy_close",
        "crude_oil": "yf_crude_oil_close",
        "eurusd": "yf_eurusd_close",
        "nasdaq100": "yf_nasdaq100_close",
        "bitcoin": "cg_bitcoin_close",
    }

    for key, col in candidates.items():
        ret_col = f"{col}_ret1"
        if ret_col in master.columns:
            ret_cols[key] = master[ret_col]
        elif col in master.columns:
            series = master[col].replace(0, np.nan)
            ret_cols[key] = np.log(series / series.shift(1))

    if len(ret_cols) < 2:
        return out

    # Key correlation pairs
    pairs = [
        ("sp500", "vix"),
        ("sp500", "gold"),
        ("sp500", "usdjpy"),
        ("gold", "usdjpy"),
        ("sp500", "crude_oil"),
        ("sp500", "eurusd"),
        ("sp500", "nasdaq100"),
        ("sp500", "bitcoin"),
        ("gold", "crude_oil"),
    ]

    for a, b in pairs:
        if a in ret_cols and b in ret_cols:
            ra, rb = ret_cols[a], ret_cols[b]
            for w in [20, 60]:
                out[f"corr_{a}_{b}_{w}d"] = ra.rolling(w, min_periods=w // 2).corr(rb)

    # Market breadth: fraction of equity assets with positive 20d return
    equity_ret_cols = [f"yf_{n}_close_ret1" for n in ["sp500", "nasdaq100", "dow_jones", "aapl", "msft", "nvda"]]
    equity_ret_available = [c for c in equity_ret_cols if c in master.columns]
    if equity_ret_available:
        mom20 = pd.DataFrame()
        for c in equity_ret_available:
            base = c.replace("_ret1", "")
            log_p = np.log(master[base].replace(0, np.nan))
            mom20[c] = log_p - log_p.shift(20)
        out["market_breadth_20d"] = (mom20 > 0).sum(axis=1) / len(equity_ret_available)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 7. MACRO & SENTIMENT RATE OF CHANGE
# ═══════════════════════════════════════════════════════════════════════════

def compute_macro_sentiment(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rate-of-change for macro indicators and Google Trends.
    Also compute cross-sentiment features.
    """
    console.print("  [cyan]Macro/sentiment features...[/]")
    out = pd.DataFrame(index=master.index)

    # BLS rate-of-change (monthly data on daily index)
    bls_cols = [c for c in master.columns if c.startswith("bls_") and not c.endswith("_missing")]
    for col in bls_cols:
        series = master[col]
        out[f"{col}_mom1"] = series.pct_change(1)   # 1-month change
        out[f"{col}_mom3"] = series.pct_change(3)   # 3-month change
        out[f"{col}_mom6"] = series.pct_change(6)   # 6-month change
        out[f"{col}_mom12"] = series.pct_change(12)  # 12-month change

    # Google Trends momentum
    trend_cols = [c for c in master.columns if c.startswith("trends_") and not c.endswith("_missing") and not c.endswith("_anomaly")]
    for col in trend_cols:
        series = master[col]
        out[f"{col}_mom1"] = series.pct_change(1)
        out[f"{col}_mom3"] = series.pct_change(3)
        out[f"{col}_mom6"] = series.pct_change(6)

    # Composite sentiment: average of all trends (normalized to 0-100 already)
    if trend_cols:
        trend_data = master[trend_cols]
        out["sentiment_avg"] = trend_data.mean(axis=1)
        out["sentiment_std"] = trend_data.std(axis=1)
        out["sentiment_max"] = trend_data.max(axis=1)

        # Fear composite: average of crash/bankruptcy/recession/sell-off terms
        fear_terms = [c for c in trend_cols if any(kw in c for kw in
                      ["crash", "bankruptcy", "recession", "sell_off", "bank_run", "credit_crisis"])]
        if fear_terms:
            out["fear_composite"] = master[fear_terms].mean(axis=1)
            out["fear_composite_mom3"] = out["fear_composite"].pct_change(3)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 8. TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def compute_technical(master: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for key assets."""
    console.print("  [cyan]Technical indicators...[/]")
    out = pd.DataFrame(index=master.index)

    # For SP500, VIX, Gold, Bitcoin: compute RSI, Bollinger Bands, MACD
    assets = {
        "sp500": "yf_sp500_close",
        "gold": "yf_gold_close",
        "bitcoin": "cg_bitcoin_close",
    }

    for name, col in assets.items():
        if col not in master.columns:
            continue
        series = master[col].replace(0, np.nan)

        # RSI (14-day)
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=7).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0, np.nan)
        out[f"{name}_rsi14"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20-day)
        ma20 = series.rolling(20, min_periods=10).mean()
        std20 = series.rolling(20, min_periods=10).std()
        out[f"{name}_bb_upper"] = ma20 + 2 * std20
        out[f"{name}_bb_lower"] = ma20 - 2 * std20
        out[f"{name}_bb_width"] = (out[f"{name}_bb_upper"] - out[f"{name}_bb_lower"]) / ma20
        out[f"{name}_bb_pctb"] = (series - out[f"{name}_bb_lower"]) / (out[f"{name}_bb_upper"] - out[f"{name}_bb_lower"]).replace(0, np.nan)

        # MACD
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        out[f"{name}_macd"] = ema12 - ema26
        out[f"{name}_macd_signal"] = out[f"{name}_macd"].ewm(span=9, adjust=False).mean()
        out[f"{name}_macd_hist"] = out[f"{name}_macd"] - out[f"{name}_macd_signal"]

        # Price relative to MA
        for w in [20, 50, 200]:
            ma = series.rolling(w, min_periods=w // 2).mean()
            out[f"{name}_price_vs_ma{w}"] = (series - ma) / ma

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 9. CRYPTO-SPECIFIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_crypto(master: pd.DataFrame) -> pd.DataFrame:
    """Compute crypto-specific features."""
    console.print("  [cyan]Crypto features...[/]")
    out = pd.DataFrame(index=master.index)

    crypto_cols = {
        "bitcoin": "cg_bitcoin_close",
        "ethereum": "cg_ethereum_close",
        "solana": "cg_solana_close",
    }

    for name, col in crypto_cols.items():
        if col not in master.columns:
            continue
        series = master[col].replace(0, np.nan)

        # Volatility
        ret = np.log(series / series.shift(1))
        for w in WINDOWS:
            out[f"{name}_vol{w}"] = ret.rolling(w, min_periods=max(w // 2, 2)).std() * np.sqrt(365)

        # BTC dominance proxy: bitcoin returns vs total crypto (if ETH available)
        if "cg_ethereum_close" in master.columns:
            btc_ret = np.log(master["cg_bitcoin_close"].replace(0, np.nan) / master["cg_bitcoin_close"].shift(1))
            eth_ret = np.log(master["cg_ethereum_close"].replace(0, np.nan) / master["cg_ethereum_close"].shift(1))
            out["crypto_btc_eth_spread"] = btc_ret - eth_ret
            out["crypto_btc_eth_spread_20d"] = out["crypto_btc_eth_spread"].rolling(20, min_periods=10).mean()

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 10. RISK-ON / RISK-OFF PROXY
# ═══════════════════════════════════════════════════════════════════════════

def compute_risk_proxy(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite risk-on/risk-off proxies.
    """
    console.print("  [cyan]Risk proxy features...[/]")
    out = pd.DataFrame(index=master.index)

    # Credit stress proxy: VIX high + yield curve inversion + fear sentiment
    components = []

    if "yf_vix_close" in master.columns:
        vix_norm = (master["yf_vix_close"] - master["yf_vix_close"].rolling(252, min_periods=60).mean()) / \
                   master["yf_vix_close"].rolling(252, min_periods=60).std().replace(0, np.nan)
        out["vix_zscore_252"] = vix_norm
        components.append("vix")

    if "yc_term_spread" in pd.DataFrame(index=master.index).columns:
        pass  # computed later

    # Safe haven demand: gold returns when equities down
    if "yf_sp500_close" in master.columns and "yf_gold_close" in master.columns:
        sp_ret = np.log(master["yf_sp500_close"].replace(0, np.nan) / master["yf_sp500_close"].shift(1))
        gold_ret = np.log(master["yf_gold_close"].replace(0, np.nan) / master["yf_gold_close"].shift(1))
        out["safe_haven_demand_20d"] = (gold_ret.rolling(20, min_periods=10).sum()) - \
                                       (sp_ret.rolling(20, min_periods=10).sum())

    # Dollar strength proxy (EUR/USD inverse)
    if "yf_eurusd_close" in master.columns:
        out["dollar_strength"] = -np.log(master["yf_eurusd_close"].replace(0, np.nan) / master["yf_eurusd_close"].shift(1))
        out["dollar_strength_20d"] = out["dollar_strength"].rolling(20, min_periods=10).sum()

    return out


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    console.print(Panel.fit("[bold cyan]Grey-Swan Feature Engineering[/]", border_style="cyan"))

    # Load master dataset
    master_path = OUT / "master_dataset.parquet"
    if not master_path.exists():
        console.print("[red]master_dataset.parquet not found. Run data_pipeline.py first.[/]")
        sys.exit(1)

    console.print(f"\n  Loading master dataset...")
    master = pd.read_parquet(master_path)
    console.print(f"  {master.shape[0]:,} rows x {master.shape[1]} columns")

    # Compute all feature groups
    feature_groups = [
        ("Log Returns",      lambda: compute_log_returns(master)),
        ("Volatility",       lambda: compute_volatility(master)),
        ("Momentum",         lambda: compute_momentum(master)),
        ("Yield Curve",      lambda: compute_yield_curve(master)),
        ("VIX Dynamics",     lambda: compute_vix_dynamics(master)),
        ("Correlations",     lambda: compute_correlations(master)),
        ("Macro/Sentiment",  lambda: compute_macro_sentiment(master)),
        ("Technical",        lambda: compute_technical(master)),
        ("Crypto",           lambda: compute_crypto(master)),
        ("Risk Proxy",       lambda: compute_risk_proxy(master)),
    ]

    all_features = []
    for name, fn in feature_groups:
        console.print(f"\n[bold]{name}[/]")
        try:
            feat = fn()
            n_new = feat.shape[1]
            all_features.append(feat)
            console.print(f"  [green][OK] {n_new} features[/]")
        except Exception as e:
            console.print(f"  [red][FAIL] {e}[/]")

    # Combine all features
    console.print(f"\n[bold]Combining features...[/]")
    features_df = pd.concat(all_features, axis=1)

    # Combine with master
    combined = pd.concat([master, features_df], axis=1)
    combined = combined.sort_index()

    # Remove duplicate columns
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Replace inf with NaN
    combined = combined.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are all NaN
    before_cols = combined.shape[1]
    combined = combined.dropna(axis=1, how="all")
    dropped = before_cols - combined.shape[1]

    final_rows, final_cols = combined.shape
    nan_total = combined.isna().sum().sum()
    nan_pct = nan_total / (final_rows * final_cols) * 100 if final_rows * final_cols > 0 else 0

    console.print(f"  {final_rows:,} rows x {final_cols} features")
    if dropped > 0:
        console.print(f"  Dropped {dropped} all-NaN columns")
    console.print(f"  Overall NaN: {nan_pct:.2f}%")

    # Save
    parquet_path = OUT / "features_dataset.parquet"
    csv_path = OUT / "features_dataset.csv"
    combined.to_parquet(parquet_path, engine="pyarrow")
    combined.to_csv(csv_path)
    console.print(f"\n  [green]Saved:[/] {parquet_path}")
    console.print(f"  [green]Saved:[/] {csv_path}")

    # Feature inventory
    inv_path = OUT / "feature_inventory.txt"
    with open(inv_path, "w") as f:
        f.write("Grey-Swan Feature Inventory\n")
        f.write(f"Generated: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Total features: {final_cols}\n")
        f.write(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}\n")
        f.write(f"Total rows: {final_rows:,}\n\n")

        # Group by prefix
        prefixes = {}
        for col in sorted(combined.columns):
            prefix = col.split("_")[0]
            if col.startswith("yf_"):
                prefix = "yf_" + col.split("_")[1]
            elif col.startswith("fred_"):
                prefix = "fred"
            elif col.startswith("french_"):
                prefix = "french"
            elif col.startswith("treasury_"):
                prefix = "treasury"
            elif col.startswith("bls_"):
                prefix = "bls"
            elif col.startswith("cg_"):
                prefix = "cg_" + col.split("_")[1]
            elif col.startswith("trends_"):
                prefix = "trends"
            elif col.startswith("cboe_"):
                prefix = "cboe"
            prefixes.setdefault(prefix, []).append(col)

        for prefix in sorted(prefixes.keys()):
            cols = prefixes[prefix]
            f.write(f"\n[{prefix}] ({len(cols)} features)\n")
            f.write(f"{'-'*50}\n")
            for c in cols:
                nn = combined[c].notna().sum()
                pct = nn / final_rows * 100
                f.write(f"  {c:<45} {nn:>8,} / {final_rows:,} ({pct:.1f}%)\n")

    console.print(f"  [green]Saved:[/] {inv_path}")

    # Table summary
    table = Table(title="Feature Groups")
    table.add_column("Group", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Example")

    for prefix in sorted(prefixes.keys()):
        cols = prefixes[prefix]
        table.add_row(prefix, str(len(cols)), cols[0])

    console.print(table)

    elapsed = time.time() - t0
    console.print(Panel.fit(
        f"[bold green]Feature engineering complete in {elapsed:.1f}s[/]\n"
        f"Master: {master.shape[0]:,} rows x {master.shape[1]} cols -> "
        f"Combined: {final_rows:,} rows x {final_cols} features",
        border_style="green"
    ))


if __name__ == "__main__":
    run()
