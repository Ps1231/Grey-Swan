"""
Grey-Swan Full Dataset Downloader
==================================
Downloads complete historical data from all 8 sources.
Uses rich for terminal output, tracks file sizes and download times.

Usage:
    python download_data.py              # download everything
    python download_data.py --yahoo      # yahoo only
    python download_data.py --french     # kenneth french only
    python download_data.py --treasury   # treasury only
    python download_data.py --bls        # bls only
    python download_data.py --coingecko  # coingecko only
    python download_data.py --cboe       # cboe only
    python download_data.py --fred       # fred/yfinance only
    python download_data.py --trends     # google trends only
"""

import os
import sys
import json
import time
import zipfile
import io
import urllib.request
import ssl
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, DownloadColumn, TransferSpeedColumn
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CTX = ssl.create_default_context()

# Results tracking
results = []


def fetch(url: str, timeout: int = 60) -> bytes:
    """Fetch raw bytes from URL."""
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=timeout, context=CTX)
    return resp.read()


def fetch_json(url: str, timeout: int = 60) -> dict:
    """Fetch and parse JSON."""
    return json.loads(fetch(url, timeout).decode("utf-8"))


def human_size(n: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def record(source: str, file: str, path: Path, elapsed: float, success: bool, error: str = ""):
    """Record a download result."""
    size = path.stat().st_size if path.exists() and success else 0
    results.append({
        "source": source,
        "file": file,
        "path": str(path),
        "size": size,
        "elapsed": elapsed,
        "success": success,
        "error": error,
    })


def save_text(path: Path, content: str):
    """Write text content to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_bytes(path: Path, content: bytes):
    """Write raw bytes to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def save_json(path: Path, data):
    """Write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════
# 1. YAHOO FINANCE (via yfinance)
# ══════════════════════════════════════════════════════════════════════════

YAHOO_TICKERS = {
    "sp500":      "^GSPC",
    "nasdaq100":  "^NDX",
    "dow_jones":  "^DJI",
    "vix":        "^VIX",
    "crude_oil":  "CL=F",
    "gold":       "GC=F",
    "eurusd":     "EURUSD=X",
    "usdjpy":     "JPY=X",
    "aapl":       "AAPL",
    "msft":       "MSFT",
    "nvda":       "NVDA",
}


def download_yahoo():
    """Download full historical data from Yahoo Finance via yfinance."""
    import yfinance as yf

    source = "Yahoo Finance"
    with console.status(f"[bold cyan]Downloading Yahoo Finance data...") as status:
        for name, ticker in YAHOO_TICKERS.items():
            status.update(f"[bold cyan]Downloading {name} ({ticker})...")
            out = DATA_DIR / "yahoo" / f"{name}.csv"
            t0 = time.time()
            try:
                t = yf.Ticker(ticker)
                df = t.history(period="max")
                df.to_csv(out)
                elapsed = time.time() - t0
                record(source, f"{name}.csv", out, elapsed, True)
                console.print(f"  [green]✓[/] {name:12s} ({ticker:8s}) → {len(df):,} rows, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                record(source, f"{name}.csv", out, elapsed, False, str(e))
                console.print(f"  [red]✗[/] {name:12s} ({ticker:8s}) → {e}")


# ══════════════════════════════════════════════════════════════════════════
# 2. KENNETH FRENCH DATA LIBRARY
# ══════════════════════════════════════════════════════════════════════════

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "3factor":  "F-F_Research_Data_Factors_daily_CSV.zip",
    "5factor":  "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
}


def download_french():
    """Download Kenneth French factor data."""
    source = "Kenneth French"
    for name, fname in FRENCH_FILES.items():
        out = DATA_DIR / "french" / f"french_{name}.csv"
        t0 = time.time()
        try:
            data = fetch(FRENCH_BASE + fname, timeout=60)
            z = zipfile.ZipFile(io.BytesIO(data))
            csv_name = z.namelist()[0]
            content = z.read(csv_name).decode("utf-8", errors="replace")
            save_text(out, content)
            elapsed = time.time() - t0
            lines = content.strip().split("\n")
            record(source, f"french_{name}.csv", out, elapsed, True)
            console.print(f"  [green]✓[/] {name:12s} → {len(lines):,} rows, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            record(source, f"french_{name}.csv", out, elapsed, False, str(e))
            console.print(f"  [red]✗[/] {name:12s} → {e}")


# ══════════════════════════════════════════════════════════════════════════
# 3. TREASURY.GOV YIELD CURVE
# ══════════════════════════════════════════════════════════════════════════

TREASURY_YEARS = list(range(2000, 2027))


def download_treasury():
    """Download Treasury.gov daily yield curve rates for all years."""
    source = "Treasury.gov"
    all_rows = []

    with console.status("[bold cyan]Downloading Treasury.gov yield curve...") as status:
        for year in TREASURY_YEARS:
            status.update(f"[bold cyan]Fetching {year}...")
            url = (
                f"https://home.treasury.gov/resource-center/data-chart-center/"
                f"interest-rates/daily-treasury-rates.csv/{year}/all"
                f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
            )
            try:
                raw = fetch(url, timeout=30).decode("utf-8")
                lines = raw.strip().split("\n")
                if lines and lines[0].startswith("Date"):
                    if not all_rows:
                        all_rows.append(lines[0])  # header
                    all_rows.extend(lines[1:])
                    console.print(f"    [dim]{year}: {len(lines)-1} rows[/]")
                else:
                    console.print(f"    [dim]{year}: no data[/]")
            except Exception as e:
                console.print(f"    [red]✗ {year}: {e}[/]")

    if all_rows:
        out = DATA_DIR / "treasury" / "treasury_yield_curve_full.csv"
        t0 = time.time()
        save_text(out, "\n".join(all_rows) + "\n")
        elapsed = time.time() - t0
        record(source, "treasury_yield_curve_full.csv", out, elapsed, True)
        console.print(f"  [green]✓[/] Full yield curve → {len(all_rows)-1:,} rows, {human_size(out.stat().st_size)}, {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# 4. BLS (Bureau of Labor Statistics)
# ══════════════════════════════════════════════════════════════════════════

BLS_SERIES = {
    "cpi_all_items":       "CUSR0000SA0",
    "unemployment_rate":   "LNS14000000",
    "civilian_labor":      "LNS11000000",
    "nonfarm_payrolls":    "CES0000000001",
    "core_cpi":            "CUSR0000SA0L1E",
}


def download_bls():
    """Download BLS data for all series (2000-present)."""
    source = "BLS"
    series_ids = list(BLS_SERIES.values())
    series_names = list(BLS_SERIES.keys())

    # BLS API allows max 20 years per request; do 2000-2019, 2020-present
    periods = [("2000", "2019"), ("2020", "2026")]

    t0 = time.time()
    for name, sid in BLS_SERIES.items():
        out = DATA_DIR / "bls" / f"bls_{name}.json"
        try:
            all_data = []
            for start, end in periods:
                payload = json.dumps({
                    "seriesid": [sid],
                    "startyear": start,
                    "endyear": end,
                }).encode()
                req = urllib.request.Request(
                    "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                )
                resp = urllib.request.urlopen(req, timeout=30, context=CTX)
                data = json.loads(resp.read().decode())
                series = data.get("Results", {}).get("series", [])
                if series:
                    all_data.extend(series[0].get("data", []))

            save_json(out, {"seriesID": sid, "data": all_data})
            elapsed = time.time() - t0
            record(source, f"bls_{name}.json", out, elapsed, True)
            console.print(f"  [green]✓[/] {name:20s} ({sid}) → {len(all_data):,} data points, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            record(source, f"bls_{name}.json", out, elapsed, False, str(e))
            console.print(f"  [red]✗[/] {name:20s} → {e}")


# ══════════════════════════════════════════════════════════════════════════
# 5. COINGECKO
# ══════════════════════════════════════════════════════════════════════════

COINGECKO_COINS = ["bitcoin", "ethereum", "tether", "binancecoin", "solana"]


def download_coingecko():
    """Download CoinGecko market data, global stats, and OHLC."""
    source = "CoinGecko"
    out_dir = DATA_DIR / "coingecko"

    # Markets endpoint
    t0 = time.time()
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
        data = fetch_json(url)
        out = out_dir / "coingecko_markets.json"
        save_json(out, data)
        elapsed = time.time() - t0
        record(source, "coingecko_markets.json", out, elapsed, True)
        console.print(f"  [green]✓[/] markets (top 100) → {len(data)} coins, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        record(source, "coingecko_markets.json", out, elapsed, False, str(e))
        console.print(f"  [red]✗[/] markets → {e}")

    # Global endpoint
    t0 = time.time()
    try:
        data = fetch_json("https://api.coingecko.com/api/v3/global")
        out = out_dir / "coingecko_global.json"
        save_json(out, data)
        elapsed = time.time() - t0
        record(source, "coingecko_global.json", out, elapsed, True)
        console.print(f"  [green]✓[/] global → {len(data.get('data', {}))} fields, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        record(source, "coingecko_global.json", out, elapsed, False, str(e))
        console.print(f"  [red]✗[/] global → {e}")

    # OHLC for each coin (365-day)
    for coin in COINGECKO_COINS:
        t0 = time.time()
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days=365"
            data = fetch_json(url)
            out = out_dir / f"coingecko_ohlc_{coin}.json"
            save_json(out, data)
            elapsed = time.time() - t0
            record(source, f"coingecko_ohlc_{coin}.json", out, elapsed, True)
            console.print(f"  [green]✓[/] ohlc {coin:12s} → {len(data)} candles, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            record(source, f"coingecko_ohlc_{coin}.json", out, elapsed, False, str(e))
            console.print(f"  [red]✗[/] ohlc {coin:12s} → {e}")


# ══════════════════════════════════════════════════════════════════════════
# 6. CBOE VIX HISTORICAL
# ══════════════════════════════════════════════════════════════════════════

def download_cboe():
    """Download CBOE VIX historical OHLCV."""
    source = "CBOE"
    t0 = time.time()
    try:
        data = fetch_json("https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_VIX.json")
        out = DATA_DIR / "cboe" / "cboe_vix_historical.json"
        save_json(out, data)
        elapsed = time.time() - t0
        entries = data.get("data", data) if isinstance(data, dict) else data
        count = len(entries) if isinstance(entries, list) else 0
        record(source, "cboe_vix_historical.json", out, elapsed, True)
        console.print(f"  [green]✓[/] VIX historical → {count:,} records, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        out = DATA_DIR / "cboe" / "cboe_vix_historical.json"
        record(source, "cboe_vix_historical.json", out, elapsed, False, str(e))
        console.print(f"  [red]✗[/] VIX historical → {e}")


# ══════════════════════════════════════════════════════════════════════════
# 7. FRED (via yfinance for market proxies)
# ══════════════════════════════════════════════════════════════════════════

FRED_TICKERS = {
    "10y_yield":   "^TNX",
    "30y_yield":   "^TYX",
    "3m_tbill":    "^IRX",
}


def download_fred():
    """Download FRED series via yfinance."""
    import yfinance as yf

    source = "FRED"
    for name, ticker in FRED_TICKERS.items():
        out = DATA_DIR / "fred" / f"fred_{name}.csv"
        t0 = time.time()
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="max")
            df.to_csv(out)
            elapsed = time.time() - t0
            record(source, f"fred_{name}.csv", out, elapsed, True)
            console.print(f"  [green]✓[/] {name:12s} ({ticker:8s}) → {len(df):,} rows, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            record(source, f"fred_{name}.csv", out, elapsed, False, str(e))
            console.print(f"  [red]✗[/] {name:12s} ({ticker:8s}) → {e}")


# ══════════════════════════════════════════════════════════════════════════
# 8. GOOGLE TRENDS (via pytrends)
# ══════════════════════════════════════════════════════════════════════════

TREND_TERMS = [
    "recession", "stock market crash", "bankruptcy",
    "inflation", "unemployment", "interest rates",
    "credit crisis", "bank run", "market sell off",
]


def download_trends():
    """Download Google Trends data for risk-related search terms."""
    source = "Google Trends"
    try:
        from pytrends.request import TrendReq
    except ImportError:
        console.print("  [red]✗[/] pytrends not installed. Run: pip install pytrends")
        record(source, "google_trends.json", Path(""), 0, False, "pytrends not installed")
        return

    pytrends = TrendReq(hl="en-US", tz=360)

    # Historical interest (yearly chunks to avoid API limits)
    t0 = time.time()
    try:
        all_data = {}
        for term in TREND_TERMS:
            try:
                pytrends.build_payload([term], timeframe="2015-01-01 2026-01-01")
                df = pytrends.interest_over_time()
                if not df.empty:
                    all_data[term] = df[term].to_dict()
                time.sleep(1)  # rate limit
            except Exception as e:
                console.print(f"    [yellow]warn[/] {term}: {e}")
                time.sleep(2)

        out = DATA_DIR / "trends" / "google_trends_historical.json"
        save_json(out, all_data)
        elapsed = time.time() - t0
        record(source, "google_trends_historical.json", out, elapsed, True)
        console.print(f"  [green]✓[/] historical trends → {len(all_data)} terms, {human_size(out.stat().st_size)}, {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        record(source, "google_trends_historical.json", Path(""), elapsed, False, str(e))
        console.print(f"  [red]✗[/] historical trends → {e}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def print_summary():
    """Print final summary table."""
    console.print()
    table = Table(title="Download Summary", box=box.ROUNDED, show_lines=True)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("File", style="white")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Time", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    total_size = 0
    total_time = 0
    successes = 0

    for r in results:
        status = "[green]✓[/]" if r["success"] else f"[red]✗ {r['error'][:30]}[/]"
        size_str = human_size(r["size"]) if r["success"] else "-"
        time_str = f"{r['elapsed']:.1f}s"
        table.add_row(r["source"], r["file"], size_str, time_str, status)
        if r["success"]:
            total_size += r["size"]
            total_time += r["elapsed"]
            successes += 1

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{successes}/{len(results)} files[/]",
        f"[bold]{human_size(total_size)}[/]",
        f"[bold]{total_time:.1f}s[/]",
        "",
    )

    console.print(table)
    console.print(f"\n[bold green]All data saved to {DATA_DIR.absolute()}[/]\n")


def main():
    parser = argparse.ArgumentParser(description="Grey-Swan Dataset Downloader")
    parser.add_argument("--yahoo",    action="store_true", help="Download Yahoo Finance only")
    parser.add_argument("--french",   action="store_true", help="Download Kenneth French only")
    parser.add_argument("--treasury", action="store_true", help="Download Treasury.gov only")
    parser.add_argument("--bls",      action="store_true", help="Download BLS only")
    parser.add_argument("--coingecko",action="store_true", help="Download CoinGecko only")
    parser.add_argument("--cboe",     action="store_true", help="Download CBOE only")
    parser.add_argument("--fred",     action="store_true", help="Download FRED (via yfinance) only")
    parser.add_argument("--trends",   action="store_true", help="Download Google Trends only")
    args = parser.parse_args()

    run_all = not any([args.yahoo, args.french, args.treasury, args.bls,
                       args.coingecko, args.cboe, args.fred, args.trends])

    console.print(Panel.fit(
        "[bold]Grey-Swan Dataset Downloader[/]\n"
        f"Target: [cyan]{DATA_DIR.absolute()}[/]\n"
        f"Time: [yellow]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
        border_style="blue",
    ))

    t_total = time.time()

    if run_all or args.yahoo:
        console.print("\n[bold blue]1/8 Yahoo Finance[/]")
        download_yahoo()

    if run_all or args.french:
        console.print("\n[bold blue]2/8 Kenneth French[/]")
        download_french()

    if run_all or args.treasury:
        console.print("\n[bold blue]3/8 Treasury.gov[/]")
        download_treasury()

    if run_all or args.bls:
        console.print("\n[bold blue]4/8 Bureau of Labor Statistics[/]")
        download_bls()

    if run_all or args.coingecko:
        console.print("\n[bold blue]5/8 CoinGecko[/]")
        download_coingecko()

    if run_all or args.cboe:
        console.print("\n[bold blue]6/8 CBOE[/]")
        download_cboe()

    if run_all or args.fred:
        console.print("\n[bold blue]7/8 FRED (via yfinance)[/]")
        download_fred()

    if run_all or args.trends:
        console.print("\n[bold blue]8/8 Google Trends[/]")
        download_trends()

    elapsed_total = time.time() - t_total
    console.print(f"\n[bold]Total wall time: {elapsed_total:.1f}s[/]")
    print_summary()


if __name__ == "__main__":
    main()
