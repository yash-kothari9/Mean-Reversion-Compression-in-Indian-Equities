# Mean Reversion Compression in Indian Equities — A Cointegration Analysis of Pairs and Baskets

## Abstract

This project investigates whether sector-specific regulatory and policy events in India alter the speed of mean reversion in cointegrated equity pairs and baskets. We construct a universe of 31 stocks from the NSE pharmaceutical and auto ancillary sectors spanning April 2019–April 2024, and identify 10 cointegrated pairs using a dual-filter approach combining the Engle-Granger two-step test and the Johansen trace test, with an Ornstein-Uhlenbeck half-life constraint of 5–126 trading days. A vectorized pairs trading backtest with realistic transaction costs (18 basis points per leg) yields Sharpe ratios ranging from −0.72 to 1.25, with the strongest performance concentrated in pairs exhibiting short half-lives (21–40 days). The core contribution is a rolling half-life regime analysis around eight policy events—including PLI scheme announcements, BS-VI emission norm implementations, and NPPA drug price revisions—which reveals that regulatory shocks compress mean-reversion half-life in 59.1% of cases, temporarily strengthening the cointegrating relationship. Extending the framework to 3-stock Johansen baskets yields 133 cointegrated triplets (9 with rank-2 cointegration), with the top baskets producing Sharpe ratios up to 1.49.

---

## Table of Contents

- [Research Overview](#research-overview)
- [Key Findings](#key-findings)
- [Pipeline Architecture](#pipeline-architecture)
- [Data](#data)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Policy Event Calendar](#policy-event-calendar)
- [Reproducing the Paper](#reproducing-the-paper)
- [Citation](#citation)
- [License](#license)

---

## Research Overview

This study addresses a gap in the pairs trading and cointegration literature: **how do identifiable regulatory shocks alter the speed of mean reversion in cointegrated equity relationships?** While the efficacy of cointegration-based trading in developed markets is well established, the behavior of mean-reversion speed around exogenous policy events remains underexplored. The Indian equity market provides a unique laboratory for this question due to the frequency and sector-specificity of its regulatory interventions.

### Three Contributions

1. **Systematic cointegration screening** — A dual-filter (Engle-Granger + Johansen + half-life tradeability) applied to all within-sector pairs across two NSE sectors
2. **Rolling half-life regime analysis** — Measures the shift in mean-reversion speed around 8 major Indian policy events (2019–2024)
3. **Johansen basket extension** — Extends from pairs to 3-stock baskets using rank-2 cointegration, capturing richer equilibrium structures

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Half-life compression dominates** | Policy events compress mean-reversion half-life in **59.1%** of pair-event cases (26 of 44), indicating temporary strengthening of cointegrating relationships |
| **Short half-life ⟹ high Sharpe** | Pairs with half-lives of 21–40 days achieve Sharpe ratios of 0.44–1.25; those exceeding 66 days produce Sharpe ratios near or below zero |
| **Baskets outperform pairs** | The top 3-stock basket (AUROPHARMA/TORNTPHARM/ABBOTINDIA) achieves Sharpe = 1.47 vs. its constituent pair at 0.53 |
| **PLI schemes compress most** | The PLI Pharma announcement compresses half-life in 7 of 7 pharma pairs with sufficient pre-event data |
| **Auto sector is heterogeneous** | Auto ancillary pairs show more varied regime responses, reflecting differing OEM exposure profiles |
| **Two-phase response** | Several pairs exhibit initial compression followed by expansion, suggesting transient structural convergence then divergence |

---

## Pipeline Architecture

The project is organized as a sequential 5-phase pipeline:

```
Phase 1: nse_data_loader.py       → Download & clean OHLCV data
Phase 2: cointegration_tests.py   → Pair cointegration screening
Phase 3: backtester.py            → Pairs backtest + regime analysis
Phase 4: (absorbed into Phase 3)
Phase 5: basket_extension.py      → 3-stock Johansen baskets
```

Each phase reads the outputs of the previous phase and produces intermediate data files, result summaries, and plots.

---

## Data

### Stock Universe

- **34 NSE-listed stocks** initially (19 pharma + 15 auto ancillary)
- **31 stocks** retained after survivorship and quality filtering
- **3 excluded**: AMARARAJA, LUMAX, MINDAIND (insufficient data)

### Survivorship Bias Controls

1. Manually curated universe with verified listing dates
2. Exclusion of stocks listed on or after the sample start (April 1, 2019)
3. Exclusion of stocks with >10% missing data

### Price Data

- **Source**: Yahoo Finance via `yfinance`
- **Period**: April 1, 2019 – April 1, 2024 (1,304 trading days)
- **Format**: Daily adjusted close prices (split/dividend adjusted)
- **Final matrix**: 1,304 × 31, zero missing values

---

## Methodology

### 1. Cointegration Screening (Dual-Filter)

A pair is classified as cointegrated if **all three** conditions are satisfied:

| Filter | Criterion |
|--------|-----------|
| **Engle-Granger** | Two-step OLS residual ADF test, p-value < 0.05 (both directions tested, lower p-value selected) |
| **Johansen Trace** | Trace statistic exceeds 95% critical value (rank ≥ 1) |
| **Half-Life Tradeability** | 5 ≤ HL ≤ 126 trading days (Ornstein-Uhlenbeck AR(1) estimate) |

### 2. Pairs Trading Backtest

| Parameter | Value |
|-----------|-------|
| Z-score window | 60-day rolling |
| Entry threshold | \|Z\| > 1.5 |
| Exit threshold | \|Z\| < 0.0 |
| Stop-loss | \|Z\| > 3.0 |
| Slippage | 5 bps/leg |
| Market impact | 10 bps/leg |
| Brokerage | 3 bps/leg |
| **Total cost** | **18 bps/leg** (36 bps round-trip) |
| Basket cost multiplier | 1.5× (54 bps round-trip) |

### 3. Rolling Half-Life Regime Analysis

For each policy event date *d_e*, the mean half-life is computed over 30 calendar days before and after the event:

- **Compression** (ΔHL < 0): Spread reverts faster → cointegrating relationship strengthens
- **Expansion** (ΔHL > 0): Spread reverts slower → cointegrating relationship weakens

### 4. Johansen Basket Extension

3-stock baskets are formed from all within-sector triplets. The basket spread is:

```
spread = w1·log(P1) + w2·log(P2) + w3·log(P3)
```

where [w1, w2, w3] is the first cointegrating eigenvector (w1 normalized to 1). Rank-2 cointegration (two independent cointegrating vectors) is also identified.

---

## Results Summary

### Cointegrated Pairs (10 pairs: 7 pharma, 3 auto)

| Pair | Sector | EG p-value | β | Half-Life (days) | Sharpe | Net Return |
|------|--------|-----------|---|-----------------|--------|------------|
| SUNDRMFAST / SUPRAJIT | Auto | 0.0022 | 1.171 | 21.72 | **1.25** | +2.37 |
| TORNTPHARM / ALKEM | Pharma | 0.0056 | 1.137 | 26.40 | 0.37 | +0.50 |
| BHARATFORG / SUNDRMFAST | Auto | 0.0109 | 0.854 | 24.75 | 0.57 | +0.89 |
| TORNTPHARM / ABBOTINDIA | Pharma | 0.0217 | 1.020 | 30.92 | 0.53 | +0.72 |
| ENDURANCE / SUPRAJIT | Auto | 0.0435 | 0.626 | 34.54 | 0.44 | +0.67 |
| DIVISLAB / LAURUSLABS | Pharma | 0.0332 | 0.458 | 40.45 | 0.68 | +0.72 |
| SUNPHARMA / ABBOTINDIA | Pharma | 0.0270 | 1.476 | 66.51 | 0.00 | −0.00 |
| ABBOTINDIA / ZYDUSLIFE | Pharma | 0.0257 | 0.734 | 75.45 | −0.48 | −0.59 |
| DRREDDY / GLENMARK | Pharma | 0.0393 | 0.556 | 81.61 | 0.15 | +0.16 |
| ABBOTINDIA / GLENMARK | Pharma | 0.0148 | 0.466 | 108.47 | **−0.72** | −0.91 |

### Top Cointegrated Baskets (of 133 total)

| Basket | Rank | HL (days) | Sharpe | Net Return |
|--------|------|-----------|--------|------------|
| DRREDDY / ZYDUSLIFE / GRANULES | 1 | 15.05 | **1.49** | +1.34 |
| AUROPHARMA / TORNTPHARM / ABBOTINDIA | 1 | 19.58 | 1.47 | +6.83 |
| TORNTPHARM / ABBOTINDIA / GLENMARK | 2 | 13.89 | 1.36 | +1.58 |
| DIVISLAB / IPCALAB / LAURUSLABS | 2 | 28.07 | 1.27 | +1.30 |
| BHARATFORG / SUNDRMFAST / SUPRAJIT | 1 | 17.15 | 1.24 | +4.20 |
| CIPLA / TORNTPHARM / ABBOTINDIA | 1 | 20.31 | 1.06 | +3.23 |

### Regime Analysis (Half-Life Compression/Expansion)

| Sector | Compression | Expansion | Compression Rate |
|--------|-------------|-----------|-----------------|
| Pharma (pairs) | 20 | 13 | 60.6% |
| Auto (pairs) | 6 | 5 | 54.5% |
| **Total (pairs)** | **26** | **18** | **59.1%** |
| Baskets (148 observations) | Dominant | — | Majority |

---

## Project Structure

```
cointegrated_pairs_across_market_regimes/
│
├── paper.tex                          # IEEE journal-format research paper (LaTeX)
├── Mean Reversion Compression...pdf   # Compiled PDF of the paper
├── README.md                          # This file
│
├── nse_data_loader.py                 # Phase 1: Data download & survivorship filtering
├── cointegration_tests.py             # Phase 2: Pair cointegration screening
├── backtester.py                      # Phase 3: Backtest + rolling half-life regime analysis
├── basket_extension.py                # Phase 5: 3-stock Johansen basket extension
│
├── data/
│   ├── raw_prices.parquet             # Adjusted close price matrix (1304 × 31)
│   ├── raw_ohlcv/                     # Full OHLCV per ticker (31 parquet files)
│   ├── universe_report.csv            # Stock inclusion/exclusion log
│   ├── cointegration_results.csv      # Full results for all 237 pairs
│   ├── cointegrated_pairs.csv         # 10 pairs passing dual-filter
│   ├── cointegrated_baskets.csv       # 133 cointegrated baskets
│   ├── basket_cointegration_results.csv  # Full basket test results
│   ├── spreads/                       # Spread time series per pair (10 parquet)
│   └── basket_spreads/               # Basket spread time series (15 CSV)
│
├── results/
│   ├── backtest_summary.csv           # Per-pair backtest metrics
│   ├── basket_backtest_summary.csv    # Per-basket backtest metrics
│   ├── event_window_analysis.csv      # Half-life shifts around events (pairs)
│   ├── basket_event_window_analysis.csv  # Half-life shifts (baskets)
│   └── rolling_halflife/              # Rolling HL time series (25 files)
│
└── plots/
    ├── pvalue_heatmap_pharma.png       # EG p-value matrix (pharma)
    ├── pvalue_heatmap_auto.png         # EG p-value matrix (auto)
    ├── spread_*.png                    # Spread + Z-score plots (10 pairs)
    ├── backtest_*.png                  # Backtest equity curves (10 pairs)
    ├── rolling_hl_*.png                # Rolling half-life with event markers (10 pairs)
    ├── regime_comparison.png           # Pre/post event bar chart (pairs)
    ├── basket_backtest_*.png           # Basket backtest curves (15 baskets)
    ├── basket_rolling_hl_*.png         # Basket rolling half-life (15 baskets)
    └── basket_regime_comparison.png    # Pre/post event bar chart (baskets)
```

---

## Installation

### Prerequisites

- Python 3.10+ (tested on 3.13)
- LaTeX distribution (for compiling the paper; e.g., TeX Live, MiKTeX)

### Setup

```bash
# Clone the repository
git clone https://github.com/<username>/cointegrated_pairs_across_market_regimes.git
cd cointegrated_pairs_across_market_regimes

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels pyarrow
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Downloading NSE historical OHLCV data |
| `pandas` | Data manipulation and time series handling |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting (heatmaps, equity curves, regime charts) |
| `seaborn` | Statistical visualization (p-value heatmaps) |
| `scipy` | Statistical functions |
| `statsmodels` | OLS, ADF test, Engle-Granger cointegration, Johansen test |
| `pyarrow` | Parquet file I/O |

---

## Usage

### Run the Full Pipeline

Execute the scripts in order. Each script reads outputs from the previous phase:

```bash
# Phase 1: Download data (requires internet connection to Yahoo Finance)
python nse_data_loader.py

# Phase 2: Run cointegration tests on all within-sector pairs
python cointegration_tests.py

# Phase 3: Backtest cointegrated pairs + rolling half-life regime analysis
python backtester.py

# Phase 5: Extend to 3-stock Johansen baskets
python basket_extension.py
```

### Run Individual Phases

Each script is self-contained and can be run independently as long as its input files exist:

- **Phase 2** requires `data/raw_prices.parquet` (from Phase 1)
- **Phase 3** requires `data/cointegrated_pairs.csv` and `data/spreads/` (from Phase 2)
- **Phase 5** requires `data/raw_prices.parquet` and `data/cointegrated_baskets.csv` (from Phases 1 and 5 itself)

### Configuration

Key parameters are defined at the top of each script and can be modified:

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `DATA_START_DATE` | 2019-04-01 | `nse_data_loader.py` | Sample start date |
| `DATA_END_DATE` | 2024-04-01 | `nse_data_loader.py` | Sample end date |
| `EG_PVALUE_THRESH` | 0.05 | `cointegration_tests.py` | Engle-Granger significance threshold |
| `JOHANSEN_CONF` | 0.05 | `cointegration_tests.py` | Johansen confidence level |
| `HL_MIN`, `HL_MAX` | 5, 126 | `cointegration_tests.py` | Half-life tradeability bounds (days) |
| `ZSCORE_WINDOW` | 60 | `backtester.py` | Rolling Z-score window (days) |
| `ENTRY_Z` | 1.5 | `backtester.py` | Z-score entry threshold |
| `EXIT_Z` | 0.0 | `backtester.py` | Z-score exit threshold |
| `STOP_Z` | 3.0 | `backtester.py` | Z-score stop-loss threshold |
| `COST_BPS` | 18 | `backtester.py` | Per-leg transaction cost (bps) |
| `EVENT_WINDOW_DAYS` | 30 | `backtester.py` | Event window width (calendar days) |
| `TOP_N_BASKETS` | 15 | `basket_extension.py` | Number of top baskets to backtest |

---

## Policy Event Calendar

Eight sector-specific policy events are analyzed:

| Date | Event | Sector |
|------|-------|--------|
| 2020-03-21 | PLI Pharma scheme announced | Pharma |
| 2020-11-10 | API Bulk Drug Park scheme notified | Pharma |
| 2021-02-01 | Union Budget 2021 — health allocation | Pharma |
| 2022-06-15 | NPPA drug price revision order | Pharma |
| 2023-03-28 | PLI pharma tranche 2 disbursement | Pharma |
| 2021-09-15 | PLI Auto Components scheme | Auto |
| 2022-04-01 | BS-VI Phase 2 norms effective | Auto |
| 2023-01-13 | EV PLI incentive revised | Auto |

---
