# Grey-Swan

## A Spatiotemporal Graph-Transformer Framework for Early Detection of Extreme Financial Market Regime Transitions

---

## 1. Overview

Grey-Swan is a research-grade financial intelligence and risk-monitoring system focused not on ordinary stock-price prediction but on identifying subtle changes in market structure that precede rare, high-impact financial events and estimating their probability, timing, and potential severity. The system models the financial market as a dynamic graph, learns cross-asset relationships under stress, and combines graph neural networks, temporal transformers, and extreme value theory to detect regime transitions before they fully materialize.

---

## 2. Market Regime Taxonomy

| Regime | Description |
|---|---|
| **Normal** | Low volatility, stable correlations, balanced risk appetite |
| **Elevated-Volatility** | Above-average but not extreme volatility; early warning signals present |
| **Stress** | Elevated risk across multiple indicators; systemic pressure building |
| **Transition** | Active regime shift in progress; correlations converging, diversification failing |
| **Extreme / Crisis** | Tail event realized; max drawdown, liquidity withdrawal, panic behavior |

**Forward-Looking Targets:** 5-day, 10-day, and 20-day extreme-event labels defined per the regime thresholds above.

---

## 3. Evaluation Metrics

| Metric | Purpose |
|---|---|
| **PR-AUC** | Primary metric for rare-event detection (handles class imbalance) |
| **ROC-AUC** | Discrimination ability across thresholds |
| **Recall @ Controlled FPR** | Early-warning recall at operationally acceptable false-positive rates |
| **F1 Score** | Precision-recall balance at chosen threshold |
| **Brier Score** | Probability calibration quality |
| **Calibration Error (ECE)** | Reliability of predicted probabilities |
| **Detection Lead Time** | How far in advance warnings are issued before event onset |
| **Max Drawdown Error** | Severity estimation accuracy |
| **Value at Risk (VaR)** | Tail quantile loss estimation |
| **Expected Shortfall (ES)** | Conditional tail risk estimation beyond VaR |

---

## 4. Data Sources

### 4.1 FRED (Federal Reserve Economic Data)

- [S&P 500](https://fred.stlouisfed.org/series/SP500)
- [CBOE Volatility Index (VIX)](https://fred.stlouisfed.org/series/VIXCLS)
- [Market Yield on U.S. Treasury Securities at 10-Year](https://fred.stlouisfed.org/series/DGS10)
- [Market Yield on U.S. Treasury Securities at 2-Year](https://fred.stlouisfed.org/series/DGS2)
- [EUR/USD Exchange Rate](https://fred.stlouisfed.org/series/DEXUSEU)
- [Effective Federal Funds Rate (DFF)](https://fred.stlouisfed.org/series/DFF)
- [High Yield Credit Spread (ICE BofA US High Yield Index OAS)](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)

### 4.2 CBOE VIX Historical Data

- [VIX Daily Price History (1990-Present)](https://www.cboe.com/tradable-products/vix/vix-historical-data)
- [3-Month Volatility Index (VIX3M)](https://www.cboe.com/us/indices/dashboard/vix3m/)
- [VIX Volatility Index (VVIX)](https://www.cboe.com/us/indices/dashboard/vvix/)

### 4.3 Kenneth R. French Data Library

- [Fama/French 3 Factors (Daily)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [Fama/French 5 Factors 2x3 (Daily)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [Momentum Factor Mom (Daily)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

### 4.4 Yahoo Finance Market Data

- **Indices:** [S&P 500 (GSPC)](https://finance.yahoo.com/quote/%5EGSPC/history/), [Nasdaq 100 (NDX)](https://finance.yahoo.com/quote/%5ENDX/history/), [Dow Jones (DJI)](https://finance.yahoo.com/quote/%5EDJI/history/), [VIX (VIX)](https://finance.yahoo.com/quote/%5EVIX/history/)
- **Commodities:** [Crude Oil Futures (CL=F)](https://finance.yahoo.com/quote/QM%3DF/history/), [Gold Futures (GC=F)](https://finance.yahoo.com/quote/MGC%3DF/history/)
- **Foreign Exchange:** [EUR/USD (EURUSD=X)](https://finance.yahoo.com/quote/EURUSD%3DX/history/), [USD/JPY (JPY=X)](https://finance.yahoo.com/quote/JPY%3DX/history/)
- **Equities:** [Apple (AAPL)](https://finance.yahoo.com/quote/AAPL/history/), [Microsoft (MSFT)](https://finance.yahoo.com/quote/MSFT/history/), [NVIDIA (NVDA)](https://finance.yahoo.com/quote/NVDA/history/)

### 4.5 Bureau of Labor Statistics (BLS)

- US CPI, unemployment, non-farm payrolls, labor force participation. Macro surprise features for inflation and employment shocks.
- [BLS Public Data API](https://www.bls.gov/data/)

### 4.6 US Treasury.gov

- Daily Treasury par yield curve rates, auction data, and federal debt statistics. Direct source for yield curve construction without reliance on third-party wrappers.
- [Treasury.gov Daily Yield Curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/)

### 4.7 CoinGecko API

- Free cryptocurrency market data: BTC dominance, total crypto market cap, DeFi TVL, stablecoin flows. Crypto stress often precedes or coincides with broader market stress.
- [CoinGecko API](https://www.coingecko.com/en/api)

### 4.8 Google Trends

- Free proxy for retail investor sentiment, search-driven panic, and attention spikes. Elevated search interest in terms like "recession", "crash", "bankruptcy" correlates with regime transitions.
- [Google Trends](https://trends.google.com/)

### 4.9 Financial News & Sentiment (Future Extension)

- News embeddings and sentiment data to investigate multimodal information in regime detection.

### 4.10 Downloaded Data Headers (Local)

All downloaded data files are stored in `data-headers/` for reference.


### 4.11 SEC EDGAR Datasets

- [Financial Statement and Notes Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets)
- [Form 8-k ](https://www.sec.gov/Archives/edgar/data/1050446/000119312526237907/mstr-20260526.htm)
- [Form 8-K Event Filings](https://sec-api.io/sandbox/latest-form-8-k-filings)

### 4.12 FRED Financial Market Indicators

- [TED Spread](https://fred.stlouisfed.org/series/STLFSI3)
- [St. Louis Fed Financial Stress Index](https://fred.stlouisfed.org/series/STLFSI3)
- [ICE BofA Option-Adjusted Spreads](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
- [3-Month Treasury Bill Secondary Market Rate (DTB3)](https://fred.stlouisfed.org/series/DTB3)
  

**Yahoo Finance (11 tickers)** -- `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`

| File | Ticker |
|---|---|
| `yahoo_sp500.csv` | ^GSPC |
| `yahoo_nasdaq.csv` | ^NDX |
| `yahoo_dow.csv` | ^DJI |
| `yahoo_vix.csv` | ^VIX |
| `yahoo_crude_oil.csv` | CL=F |
| `yahoo_gold.csv` | GC=F |
| `yahoo_eurusd.csv` | EURUSD=X |
| `yahoo_usdjpy.csv` | JPY=X |
| `yahoo_aapl.csv` | AAPL |
| `yahoo_msft.csv` | MSFT |
| `yahoo_nvda.csv` | NVDA |

**Kenneth French Data Library (3 files)** -- `Date, Mkt-RF, SMB, HML, RF` (+ RMW, CMA for 5-Factor)

| File | Description | Rows |
|---|---|---|
| `french_3factor.csv` | Fama/French 3 Factors (Daily, 1926-present) | 26,276 |
| `french_5factor.csv` | Fama/French 5 Factors 2x3 (Daily, 1963-present) | 15,856 |
| `french_momentum.csv` | Momentum Factor Mom (Daily, 1926-present) | 26,187 |

**Treasury.gov Yield Curve** -- `Date, 1Mo, 2Mo, 3Mo, 4Mo, 6Mo, 1Yr, 2Yr, 3Yr, 5Yr, 7Yr, 10Yr, 20Yr, 30Yr`

| File | Rows |
|---|---|
| `treasury_yield_curve.csv` | 250 (daily 2024) |

**BLS (3 series)** -- `year, period, periodName, value`

| File | Series | Description |
|---|---|---|
| `bls_cpi.json` | CUSR0000SA0 | CPI-U All Items |
| `bls_multi.json` | CUSR0000SA0, LNS14000000, LNS11000000 | CPI, Unemployment Rate, Civilian Labor Force |

**CoinGecko (3 endpoints)** -- JSON

| File | Endpoint | Fields |
|---|---|---|
| `coingecko_markets.json` | /coins/markets | current_price, market_cap, volume, high_24h, low_24h, price_change_24h, 26 fields total |
| `coingecko_global.json` | /global | active_cryptocurrencies, total_market_cap, market_cap_percentage, volume |
| `coingecko_ohlc.json` | /coins/bitcoin/ohlc | [timestamp, open, high, low, close] |

**CBOE VIX Historical** -- `date, volume, open, high, low, close`

| File | Description |
|---|---|
| `cboe_vix.json` | VIX daily OHLCV (1990-present) |

**Google Trends** -- `date, recession, stock market crash, bankruptcy, inflation, unemployment, isPartial`

| File | Description |
|---|---|
| `google_trends.csv` | Weekly search interest for risk-related terms |

---

## 5. Data Engineering Pipeline

The reproducible pipeline handles multi-source ingestion and produces a unified time-indexed master dataset:

1. **Collection** -- Pull raw data from all sources via APIs and bulk downloads
2. **Cleaning** -- Remove duplicates, fix formatting, handle malformed records
3. **Missing Value Treatment** -- Forward-fill, interpolation, or flagging per data type
4. **Calendar Alignment** -- Synchronize different trading calendars (US, India, holidays)
5. **Price Adjustments** -- Split/dividend adjustments for equities
6. **Anomaly Detection** -- Statistical outlier flagging at the record level
7. **Normalization** -- Standardization/scaling for model input compatibility
8. **Look-Ahead Prevention** -- Strict point-in-time alignment; no future information leakage
9. **Dataset Versioning** -- Immutable snapshots for reproducibility
10. **Master Dataset** -- Unified, time-indexed table covering all assets and indicators

---

## 6. Feature Engineering

A comprehensive feature set captures the evolving state of the financial system:

| Category | Features |
|---|---|
| **Returns** | Log returns, rolling returns (5/10/20/60-day), momentum |
| **Drawdowns** | Current drawdown, max drawdown over window, drawdown acceleration |
| **Volatility** | Realized volatility, Parkinson, Garman-Klass, volatility acceleration, vol-of-vol |
| **VIX Dynamics** | VIX level, VIX term structure (VIX/VIX3M ratio), VVIX, VIX momentum |
| **Market Breadth** | Advance-decline, % above moving averages, new highs vs new lows |
| **Volume** | Volume anomalies, OBV, volume-price divergence |
| **Sector Dispersion** | Cross-sector return spread, sector correlation breakdown |
| **Correlations** | Rolling cross-asset correlations, correlation change detection, correlation convergence |
| **Lead-Lag** | Cross-asset lead-lag relationships, information transmission delays |
| **Yield Spreads** | 10Y-2Y term spread, credit spreads, yield curve slope |
| **Credit** | High-yield OAS, credit deterioration signals |
| **FX Stress** | DXY proxy, EUR/USD and USD/JPY volatility, currency correlation shifts |
| **Commodities** | Oil shocks, gold safe-haven flows, commodity correlation with equities |
| **Macro** | Fed Funds Rate changes, macro surprise indices, economic momentum |

---

## 7. Dynamic Financial Graph Construction

The financial market is modeled as a time-varying graph:

- **Nodes:** Equities, sector ETFs, indices, bonds, currencies, commodities, volatility instruments, credit instruments, macro-sensitive assets
- **Edges:** Time-varying constructed from rolling correlations, partial correlations, dependency measures, lead-lag relationships, sector co-membership, and other pairwise relationships
- **Key Property:** During stress periods, diversification fails and correlations converge toward 1 -- the graph structure compresses, and the GNN learns to detect this structural collapse as a precursor to extreme events
- **Edge Construction:** Rebuilt at each time step or sliding window to capture evolving relationships

---

## 8. Model Architecture

The proposed architecture combines three components into a unified multi-task system:

### 8.1 Dynamic Graph Neural Network (PyTorch Geometric)

- Encodes cross-asset relationships and their temporal evolution
- Produces a graph-level representation capturing systemic state
- Learns how relationships between assets change under stress

### 8.2 Temporal Transformer (PyTorch)

- Takes sequential GNN outputs as input
- Captures long-range temporal dependencies and regime evolution patterns
- Self-attention mechanism identifies which historical states are most relevant to current conditions

### 8.3 Extreme Value Theory (EVT) Module

- Models the extreme tail of the return/loss distribution specifically
- Uses SciPy/PyTorch for GEV/GPD fitting
- Focuses on tail behavior rather than treating rare events as ordinary classification examples

### 8.4 Multi-Task Prediction Head

Simultaneously predicts:

- **Current market regime** (classification)
- **Extreme event probability** at 5/10/20-day horizons
- **Expected maximum drawdown** (regression)
- **Time-to-event** (survival analysis)

**Output:** Current regime, Grey-Swan Risk Score, 5/10/20-day tail-event probabilities, expected drawdown, confidence estimate, dominant risk factors.

---

## 9. Baseline Models

All baselines are implemented under identical temporal validation conditions:

| Model | Type | Purpose |
|---|---|---|
| Logistic Regression | Linear | Minimum complexity baseline |
| Random Forest | Ensemble | Non-linear tabular baseline |
| XGBoost / LightGBM | Gradient Boosting | Strong tabular baseline |
| LSTM | Recurrent | Sequential modeling baseline |
| TCN | Convolutional | Temporal pattern baseline |
| Standard Transformer | Attention | Temporal attention baseline (no graph) |

---

## 10. Ablation Study

Systematic comparison to isolate the contribution of each component:

| Configuration | Components |
|---|---|
| 1 | Logistic Regression |
| 2 | Random Forest |
| 3 | XGBoost / LightGBM |
| 4 | LSTM |
| 5 | TCN |
| 6 | Standard Transformer |
| 7 | GNN only |
| 8 | GNN + Transformer |
| 9 | GNN + EVT |
| 10 | Transformer + EVT |
| 11 | GNN + Transformer + EVT |
| 12 | GNN + Transformer + EVT + AutoML |

---

## 11. AutoML Layer

An automated research layer searches for strong feature/model/hyperparameter configurations:

| Tool | Role |
|---|---|
| **Optuna** | Hyperparameter and architecture optimization (learning rates, sequence lengths, hidden dims, attention heads, GNN depth, Transformer depth, dropout, batch size, optimizers, loss weights, feature subsets) |
| **AutoGluon / FLAML** | Automated classical/tabular model selection and strong AutoML baselines |
| **Ray Tune** | Distributed hyperparameter and architecture search at scale |
| **MLflow** | Experiment tracking -- datasets, hyperparameters, models, metrics, configs, and results for full reproducibility |

---

## 12. Training and Validation Protocol

- **Strict chronological splits** -- no random train/test splitting
- **Walk-forward validation** -- rolling origin with expanding or sliding training windows
- **No look-ahead bias** -- all features constructed from point-in-time data only
- **Distribution shift testing** -- evaluate on market conditions that differ substantially from training distribution (e.g., 2008 GFC, 2020 COVID, 2022 rate hikes)
- **Progressive complexity** -- test whether the model recognizes unfamiliar combinations of volatility, correlations, macro conditions, and cross-asset stress

---

## 13. Interpretability

The system identifies major factors contributing to each warning:

- **Feature attribution** -- volatility acceleration, credit-spread widening, correlation convergence, currency stress, yield-curve instability, sector synchronization, commodity shocks
- **Graph-level explanations** -- identify which assets and relationships contribute most strongly to systemic risk
- **Attention visualization** -- temporal attention weights reveal which historical regimes the model considers most relevant
- **Risk factor decomposition** -- decompose the Grey-Swan Risk Score into individual contributing factors

---

## 14. Monitoring Dashboard and API

**Dashboard displays:**

- Current market regime
- Overall Grey-Swan Risk Score
- 5-, 10-, and 20-day extreme-event probabilities
- Expected drawdown, VaR, Expected Shortfall
- Model confidence and detection lead time
- Cross-asset stress levels
- Dynamic correlation structure (graph visualization)
- Most influential risk factors

**Grey-Swan Risk API** -- programmatic access to all model outputs for integration with external systems.

---

## 15. Full Pipeline Summary

```
Multi-Source Financial Data Ingestion
    (FRED, CBOE, Yahoo Finance, Kenneth French, BLS, Treasury.gov, CoinGecko, Google Trends)
        |
        v
Cleaning and Versioning Pipeline
    (missing values, calendar alignment, anomaly detection, normalization, look-ahead prevention)
        |
        v
Feature Engineering
    (returns, volatility, VIX dynamics, breadth, correlations, yield curves, FX, commodities, macro)
        |
        v
Dynamic Financial Graph Construction
    (nodes: assets/indices/bonds/FX/commodities/vol | edges: time-varying correlations, lead-lag)
        |
        v
PyTorch / PyTorch-Geometric GNN
    (cross-asset relationship learning, stress detection)
        |
        v
PyTorch Temporal Transformer
    (long-range temporal dependencies, regime evolution)
        |
        v
Multi-Task Regime and Risk Prediction
    (regime, event probability, drawdown, time-to-event)
        |
        v
EVT Tail Modelling (SciPy / PyTorch)
    (extreme tail distribution fitting, VaR, Expected Shortfall)
        |
        v
AutoML Layer
    (Optuna HPO, AutoGluon/FLAML tabular, Ray Tune distributed, MLflow tracking)
        |
        v
Interpretability
    (factor attribution, graph explanations, attention visualization)
        |
        v
Grey-Swan Risk API + Real-Time Research Dashboard
```

---

## 16. Future Extensions

- Financial-news embeddings and sentiment analysis
- Global market contagion modeling
- Additional alternative data sources
- Multimodal information fusion for improved early detection of unseen extreme financial regime transitions
