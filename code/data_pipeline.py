"""
Grey-Swan Data Engineering Pipeline
====================================
Cleans, aligns, and merges all raw data into a unified master dataset.

Usage:
    python code/data_pipeline.py              # full pipeline
    python code/data_pipeline.py --skip-viz   # skip data quality report

Output:
    data/processed/master_dataset.parquet
    data/processed/master_dataset.csv
    data/processed/data_quality_report.txt
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ─── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA          # raw data lives in data/<source>/
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ─── US trading calendar (pandas) ────────────────────────────────────────
US_BUSINESS_DAYS = pd.tseries.offsets.BDay()


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOADING FUNCTIONS — one per source
# ═══════════════════════════════════════════════════════════════════════════

def load_yahoo(name: str) -> pd.DataFrame:
    """Load a Yahoo Finance CSV. Returns Date-indexed Close price."""
    path = RAW / "yahoo" / f"{name}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.set_index("Date").sort_index()
    # Drop dividends/splits (useless noise for price analysis)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"yf_{name}_{c.lower()}" for c in df.columns]
    return df


def load_french(name: str) -> pd.DataFrame:
    """Load a Kenneth French factor CSV (handles preamble + blank lines)."""
    path = RAW / "french" / f"{name}.csv"
    lines = path.read_text(encoding="utf-8").splitlines()

    # Find the header line: look for line containing known factor names
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Header line contains known factor keywords
        if any(kw in stripped for kw in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Cannot find header in {name}")

    # Read from header line onwards, skip blank lines
    header_line = lines[header_idx].strip()
    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("Copyright"):
            data_lines.append(stripped)

    # Parse
    from io import StringIO
    csv_text = header_line + "\n" + "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_text))

    # First column is unnamed — it's the date
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").sort_index()

    # Replace missing sentinels
    df = df.replace([-99.99, -999, -99.9], np.nan)

    # Convert from percentage to decimal
    for col in df.columns:
        df[col] = df[col] / 100.0

    df.columns = [f"french_{name}_{c.strip()}" for c in df.columns]
    return df


def load_fred_yields() -> pd.DataFrame:
    """Load FRED yields (10Y, 30Y, 3M) from Yahoo-format CSVs."""
    frames = []
    for name, ticker in [("fred_10y_yield", "10y"), ("fred_30y_yield", "30y"),
                          ("fred_3m_tbill", "3m")]:
        path = RAW / "fred" / f"{name}.csv"
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
        df = df.set_index("Date").sort_index()
        df = df[["Close"]].copy()
        df.columns = [f"fred_{ticker}_yield"]
        frames.append(df)
    return pd.concat(frames, axis=1, sort=True)


def load_treasury_yield_curve() -> pd.DataFrame:
    """Load Treasury yield curve CSV (reverse-chronological, variable columns)."""
    path = RAW / "treasury" / "treasury_yield_curve_full.csv"

    # Read with python engine to handle variable column counts
    import io
    rows = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            # First col is always the date, take only the 10 yield cols
            # (some rows have extra cols for newer tenors)
            date_str = parts[0]
            yields = parts[1:11]  # Take first 10 yield columns
            if len(yields) < 10:
                yields += [np.nan] * (10 - len(yields))
            rows.append([date_str] + yields)

    col_names = ["Date", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr",
                 "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]
    df = pd.DataFrame(rows, columns=col_names)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()

    # Clean column names
    df.columns = [f"treasury_{c.replace(' ', '_').replace('.', '').lower()}" for c in df.columns]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_bls(series_name: str) -> pd.DataFrame:
    """Load a BLS JSON series. Returns monthly CPI or unemployment."""
    path = RAW / "bls" / f"bls_{series_name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    records = raw.get("data", [])
    if not records:
        raise ValueError(f"empty data array in {path.name}")

    rows = []
    for r in records:
        year = int(r["year"])
        period = r["period"]  # "M01" ... "M12"
        if period.startswith("M"):
            month = int(period[1:])
            date = pd.Timestamp(year=year, month=month, day=1)
            val_str = r["value"]
            if val_str == "-" or val_str == "":
                continue
            value = float(val_str)
            rows.append({"Date": date, f"bls_{series_name}": value})

    if not rows:
        raise ValueError(f"no valid rows after parsing {path.name}")

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_cboe_vix() -> pd.DataFrame:
    """Load CBOE VIX historical JSON."""
    path = RAW / "cboe" / "cboe_vix_historical.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("data", [])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "Date"})
    df = df.set_index("Date").sort_index()

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["open", "high", "low", "close"]].copy()
    df.columns = ["cboe_vix_open", "cboe_vix_high", "cboe_vix_low", "cboe_vix_close"]
    return df


def load_coingecko_ohlc(coin: str) -> pd.DataFrame:
    """Load CoinGecko OHLC data (Unix ms arrays)."""
    path = RAW / "coingecko" / f"coingecko_ohlc_{coin}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close"])
    df["Date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
    df = df.set_index("Date").sort_index()
    df = df[["open", "high", "low", "close"]].copy()
    df.columns = [f"cg_{coin}_{c}" for c in df.columns]
    return df


def load_google_trends() -> pd.DataFrame:
    """Load Google Trends historical scores."""
    path = RAW / "trends" / "google_trends_historical.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    frames = []
    for term, series in raw.items():
        dates = pd.to_datetime(list(series.keys()))
        values = list(series.values())
        s = pd.Series(values, index=dates, name=f"trends_{term.replace(' ', '_')}")
        frames.append(s)

    df = pd.concat(frames, axis=1)
    df.index.name = "Date"
    df = df.sort_index()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def clean_dataframe(df: pd.DataFrame, source_name: str) -> dict:
    """Clean a DataFrame: remove duplicates, fix types, log quality stats."""
    stats = {"source": source_name, "raw_rows": len(df)}

    # Drop exact duplicates
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    stats["dupes_removed"] = before - len(df)

    # Convert all columns to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where all values are NaN
    before = len(df)
    df = df.dropna(how="all")
    stats["all_nan_rows"] = before - len(df)

    # Round float precision (Yahoo artifacts)
    for col in df.columns:
        df[col] = df[col].round(8)

    stats["clean_rows"] = len(df)
    return df, stats


# ═══════════════════════════════════════════════════════════════════════════
# 3. MISSING VALUE TREATMENT
# ═══════════════════════════════════════════════════════════════════════════

def treat_missing(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Handle missing values per-source:
    - Price/yield data: forward-fill (market closed = no new data)
    - Factor/score data: linear interpolation then forward-fill
    - Flag remaining NaNs
    """
    stats = {"source": source_name}

    total_cells = df.shape[0] * df.shape[1]
    nan_before = df.isna().sum().sum()
    stats["nan_before"] = int(nan_before)
    stats["nan_pct_before"] = round(nan_before / total_cells * 100, 2) if total_cells > 0 else 0

    # Identify column types
    is_yield = any(kw in source_name for kw in ["fred", "treasury", "yield"])
    is_factor = any(kw in source_name for kw in ["french", "trends"])
    is_price = any(kw in source_name for kw in ["yahoo", "cboe", "coingecko"])

    if is_factor or source_name == "bls":
        # Monthly data — interpolate then ffill
        df = df.interpolate(method="linear", limit_direction="both")
        df = df.ffill()
    else:
        # Daily data — forward-fill (market closures)
        df = df.ffill()
        # For any remaining leading NaNs, back-fill
        df = df.bfill()

    nan_after = df.isna().sum().sum()
    stats["nan_after"] = int(nan_after)
    stats["nan_pct_after"] = round(nan_after / total_cells * 100, 2) if total_cells > 0 else 0

    # Create missing flags for columns that had significant gaps
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    return df, stats


# ═══════════════════════════════════════════════════════════════════════════
# 4. ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_anomalies(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Flag extreme values using rolling z-scores.
    Does NOT remove them -- just adds anomaly flags for downstream use.
    """
    stats = {"source": source_name, "anomalies": 0}

    flag_cols = {}
    for col in df.columns:
        if df[col].dtype != float:
            continue
        roll_mean = df[col].rolling(252, min_periods=60, center=False).mean()
        roll_std = df[col].rolling(252, min_periods=60, center=False).std()
        z = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        anomaly_flag = (z.abs() > 4).astype(int)
        n_anom = int(anomaly_flag.sum())
        stats["anomalies"] += n_anom
        if n_anom > 0:
            flag_cols[f"{col}_anomaly"] = anomaly_flag

    if flag_cols:
        df = pd.concat([df, pd.DataFrame(flag_cols, index=df.index)], axis=1)

    return df, stats


# ═══════════════════════════════════════════════════════════════════════════
# 5. CALENDAR ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def align_to_us_calendar(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Align all data to US business day calendar.
    - Daily data: forward-fill across market holidays/weekends
    - Monthly data: resample to month-end
    """
    is_monthly = any(kw in source_name for kw in ["bls", "trends"])

    if is_monthly:
        # Already monthly — just ensure consistent month-end
        df.index = pd.to_datetime(df.index)
        df = df.resample("MS").first()  # start of month
        df = df.ffill()
    else:
        # Daily: reindex to US business days, forward-fill gaps
        if len(df) == 0:
            return df
        start = df.index.min()
        end = df.index.max()
        all_days = pd.date_range(start, end, freq=US_BUSINESS_DAYS)
        df = df.reindex(all_days)
        df = df.ffill()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 6. NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def z_score_normalize(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Z-score normalize within each column using expanding window.
    Prevents look-ahead bias — each row only uses data up to that point.
    """
    df_norm = pd.DataFrame(index=df.index)

    for col in df.columns:
        if df[col].dtype != float or df[col].std() == 0:
            df_norm[col] = df[col]
            continue
        # Expanding z-score (no look-ahead)
        exp_mean = df[col].expanding(min_periods=60).mean()
        exp_std = df[col].expanding(min_periods=60).std()
        df_norm[col] = (df[col] - exp_mean) / exp_std.replace(0, np.nan)

    df_norm.columns = [f"{c}_zscore" for c in df.columns]
    return df_norm


# ═══════════════════════════════════════════════════════════════════════════
# 7. PIPELINE ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(skip_viz: bool = False):
    """Run the full data engineering pipeline."""
    t0 = time.time()
    console.print(Panel.fit("[bold cyan]Grey-Swan Data Engineering Pipeline[/]", border_style="cyan"))

    all_stats = []

    # ─── Step 1: Load all sources ───────────────────────────────────────
    console.print("\n[bold]Step 1/5: Loading raw data[/]")

    loaders = {
        "yahoo_sp500":     lambda: load_yahoo("sp500"),
        "yahoo_vix":       lambda: load_yahoo("vix"),
        "yahoo_aapl":      lambda: load_yahoo("aapl"),
        "yahoo_msft":      lambda: load_yahoo("msft"),
        "yahoo_nvda":      lambda: load_yahoo("nvda"),
        "yahoo_nasdaq100": lambda: load_yahoo("nasdaq100"),
        "yahoo_dow":       lambda: load_yahoo("dow_jones"),
        "yahoo_gold":      lambda: load_yahoo("gold"),
        "yahoo_crude_oil": lambda: load_yahoo("crude_oil"),
        "yahoo_eurusd":    lambda: load_yahoo("eurusd"),
        "yahoo_usdjpy":    lambda: load_yahoo("usdjpy"),
        "fred_yields":     load_fred_yields,
        "french_3factor":  lambda: load_french("french_3factor"),
        "french_5factor":  lambda: load_french("french_5factor"),
        "french_momentum": lambda: load_french("french_momentum"),
        "treasury_yields": load_treasury_yield_curve,
        "bls_cpi":         lambda: load_bls("cpi_all_items"),
        "bls_unemployment":lambda: load_bls("unemployment_rate"),
        "bls_core_cpi":    lambda: load_bls("core_cpi"),
        "bls_nonfarm":     lambda: load_bls("nonfarm_payrolls"),
        "bls_civilian":    lambda: load_bls("civilian_labor"),
        "cboe_vix":        load_cboe_vix,
        "cg_bitcoin":      lambda: load_coingecko_ohlc("bitcoin"),
        "cg_ethereum":     lambda: load_coingecko_ohlc("ethereum"),
        "cg_solana":       lambda: load_coingecko_ohlc("solana"),
        "cg_binance":      lambda: load_coingecko_ohlc("binancecoin"),
        "cg_tether":       lambda: load_coingecko_ohlc("tether"),
        "google_trends":   load_google_trends,
    }

    loaded = {}
    for name, loader in loaders.items():
        try:
            df = loader()
            loaded[name] = df
            cols = df.shape[1]
            rows = df.shape[0]
            date_range = f"{df.index.min().date()} to {df.index.max().date()}" if len(df) > 0 else "empty"
            console.print(f"  [green][OK] {name:22s}[/]  {rows:>7,} rows x {cols:>3} cols  [{date_range}]")
        except Exception as e:
            console.print(f"  [red][FAIL] {name:22s}[/]  {e}")

    # ─── Step 2: Clean ──────────────────────────────────────────────────
    console.print("\n[bold]Step 2/5: Cleaning[/]")
    cleaned = {}
    for name, df in loaded.items():
        df_clean, stats = clean_dataframe(df, name)
        cleaned[name] = df_clean
        all_stats.append(stats)
        if stats["dupes_removed"] > 0 or stats["all_nan_rows"] > 0:
            console.print(f"  [yellow][~] {name:22s}[/]  dupes={stats['dupes_removed']}, all-NaN rows={stats['all_nan_rows']}")
        else:
            console.print(f"  [green][OK] {name:22s}[/]  clean")

    # ─── Step 3: Missing value treatment ────────────────────────────────
    console.print("\n[bold]Step 3/5: Missing value treatment[/]")
    filled = {}
    for name, df in cleaned.items():
        df_filled, mstats = treat_missing(df, name)
        filled[name] = df_filled
        all_stats.append(mstats)
        pct = mstats["nan_pct_before"]
        pct_after = mstats["nan_pct_after"]
        if pct > 0:
            console.print(f"  [yellow][~] {name:22s}[/]  NaN: {pct:.1f}% -> {pct_after:.1f}%")
        else:
            console.print(f"  [green][OK] {name:22s}[/]  no missing data")

    # ─── Step 4: Calendar alignment ─────────────────────────────────────
    console.print("\n[bold]Step 4/5: Calendar alignment[/]")
    aligned = {}
    for name, df in filled.items():
        df_aligned = align_to_us_calendar(df, name)
        aligned[name] = df_aligned
        console.print(f"  [green][OK] {name:22s}[/]  {len(df_aligned):>7,} rows")

    # ─── Step 5: Merge into master ──────────────────────────────────────
    console.print("\n[bold]Step 5/5: Building master dataset[/]")

    # Separate daily and monthly sources for proper merge strategy
    monthly_sources = ["bls_cpi", "bls_unemployment", "bls_core_cpi",
                       "bls_nonfarm", "bls_civilian", "google_trends"]
    daily_sources = [k for k in aligned if k not in monthly_sources]

    # Start with the broadest daily dataset as the base
    daily_frames = [aligned[k] for k in daily_sources if len(aligned[k]) > 0]
    if daily_frames:
        master = pd.concat(daily_frames, axis=1)
    else:
        master = pd.DataFrame()

    # Merge monthly data — forward-fill to daily frequency
    for name in monthly_sources:
        if name in aligned and len(aligned[name]) > 0:
            monthly_df = aligned[name]
            # Reindex to daily (forward-fill monthly values across days)
            if len(master) > 0:
                monthly_df = monthly_df.reindex(master.index, method="ffill")
            master = pd.concat([master, monthly_df], axis=1)

    # ─── Anomaly detection on master ────────────────────────────────────
    console.print("\n  [bold]Anomaly detection...[/]")
    master, anom_stats = detect_anomalies(master, "master")
    console.print(f"  Found {anom_stats['anomalies']} extreme values flagged")

    # ─── Final stats ────────────────────────────────────────────────────
    master = master.sort_index()
    final_rows, final_cols = master.shape
    date_start = master.index.min().date()
    date_end = master.index.max().date()
    nan_total = master.isna().sum().sum()
    nan_pct = nan_total / (final_rows * final_cols) * 100 if final_rows * final_cols > 0 else 0

    console.print(f"\n  [bold green]Master dataset: {final_rows:,} rows x {final_cols} cols[/]")
    console.print(f"  Date range: {date_start} to {date_end}")
    console.print(f"  Overall NaN: {nan_pct:.2f}%")

    # ─── Save outputs ───────────────────────────────────────────────────
    parquet_path = OUT / "master_dataset.parquet"
    csv_path = OUT / "master_dataset.csv"

    master.to_parquet(parquet_path, engine="pyarrow")
    master.to_csv(csv_path)

    console.print(f"\n  [green]Saved:[/] {parquet_path}")
    console.print(f"  [green]Saved:[/] {csv_path}")

    # ─── Column inventory ───────────────────────────────────────────────
    table = Table(title="Master Dataset Columns", show_lines=True)
    table.add_column("Column", style="cyan")
    table.add_column("Dtype")
    table.add_column("Non-Null")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("Source Group")

    for col in sorted(master.columns):
        if master[col].dtype in [float, int]:
            mn = f"{master[col].min():.4f}" if not np.isnan(master[col].min()) else "-"
            mx = f"{master[col].max():.4f}" if not np.isnan(master[col].max()) else "-"
        else:
            mn, mx = "-", "-"
        nn = f"{master[col].notna().sum():,}"

        # Determine source group
        if col.startswith("yf_"):
            group = "Yahoo Finance"
        elif col.startswith("fred_"):
            group = "FRED"
        elif col.startswith("french_"):
            group = "French Factors"
        elif col.startswith("treasury_"):
            group = "Treasury"
        elif col.startswith("bls_"):
            group = "BLS"
        elif col.startswith("cboe_"):
            group = "CBOE"
        elif col.startswith("cg_"):
            group = "CoinGecko"
        elif col.startswith("trends_"):
            group = "Google Trends"
        else:
            group = "Other"

        table.add_row(col, str(master[col].dtype), nn, mn, mx, group)

    console.print(table)

    # ─── Data quality report ────────────────────────────────────────────
    report_path = OUT / "data_quality_report.txt"
    with open(report_path, "w") as f:
        f.write("Grey-Swan Data Quality Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Master dataset shape: {final_rows} rows x {final_cols} cols\n")
        f.write(f"Date range: {date_start} to {date_end}\n")
        f.write(f"Overall NaN: {nan_pct:.2f}%\n\n")
        f.write(f"{'Column':<45} {'Dtype':<8} {'Non-Null':>10} {'NaN%':>8}\n")
        f.write(f"{'-'*75}\n")
        for col in sorted(master.columns):
            nn = master[col].notna().sum()
            total = len(master)
            pct = (1 - nn / total) * 100 if total > 0 else 0
            f.write(f"{col:<45} {str(master[col].dtype):<8} {nn:>10,} {pct:>7.2f}%\n")

        f.write(f"\n{'='*60}\n")
        f.write("Source loading stats:\n")
        for s in all_stats:
            if "source" in s:
                f.write(f"  {s['source']}: {s.get('raw_rows', s.get('nan_before', 'N/A'))} rows\n")

    console.print(f"  [green]Saved:[/] {report_path}")

    elapsed = time.time() - t0
    console.print(Panel.fit(
        f"[bold green]Pipeline complete in {elapsed:.1f}s[/]\n"
        f"Master dataset: {final_rows:,} rows x {final_cols} cols\n"
        f"Date range: {date_start} to {date_end}",
        border_style="green"
    ))


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grey-Swan Data Engineering Pipeline")
    parser.add_argument("--skip-viz", action="store_true", help="Skip data quality report")
    args = parser.parse_args()
    run_pipeline(skip_viz=args.skip_viz)
