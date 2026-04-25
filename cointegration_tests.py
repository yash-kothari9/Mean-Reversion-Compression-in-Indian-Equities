"""
Phase 2 — Cointegration Tests & Half-Life Estimation
=====================================================
Runs Engle-Granger and Johansen cointegration tests across all
within-sector pairs in the NSE pharma and auto ancillary universe.

For each pair that passes both tests, computes:
  - Cointegrating vector (hedge ratio)
  - Spread (residual series)
  - Half-life of mean reversion via Ornstein-Uhlenbeck fit
  - Z-score of current spread

Outputs:
  data/cointegration_results.csv   — full results table, all pairs
  data/cointegrated_pairs.csv      — pairs that passed both tests
  data/spreads/                    — spread time series per passing pair
  plots/pvalue_heatmap.png         — p-value heatmap across universe
  plots/spread_<A>_<B>.png         — spread + Z-score for each valid pair

Requirements:
    pip install statsmodels pandas numpy matplotlib seaborn pyarrow
"""

import os
import warnings
import itertools
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

EG_PVALUE_THRESHOLD  = 0.05   # Engle-Granger significance level
JO_CONFIDENCE        = 0.05   # Johansen confidence level (0.05 = 95%)
MIN_HALFLIFE_DAYS    = 5      # Ignore pairs with half-life < 5 days (too noisy)
MAX_HALFLIFE_DAYS    = 126    # Ignore pairs with half-life > 126 days (~6 months)
ZSCORE_WINDOW        = 60     # Rolling window for Z-score normalisation
DATA_DIR             = "data"
PLOT_DIR             = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/spreads", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =============================================================================
# STEP 1: Load price data
# =============================================================================

prices = pd.read_parquet(f"{DATA_DIR}/raw_prices.parquet")
report = pd.read_csv(f"{DATA_DIR}/universe_report.csv")
sector_map = dict(zip(report["ticker"], report["sector"]))

pharma_tickers = [t for t in prices.columns if sector_map.get(t) == "pharma"]
auto_tickers   = [t for t in prices.columns if sector_map.get(t) == "auto_ancillary"]

log.info(f"Loaded price matrix: {prices.shape}")
log.info(f"Pharma tickers: {len(pharma_tickers)}")
log.info(f"Auto ancillary tickers: {len(auto_tickers)}\n")

# Use log prices — standard practice for cointegration on equity prices
# Log prices are more likely to be I(1) and the spread is interpretable
# as a log price ratio (equivalent to a pairs spread in returns space)
log_prices = np.log(prices)

# =============================================================================
# STEP 2: ADF unit root pre-check
# Individual series should be I(1) — non-stationary in levels,
# stationary in first differences — for cointegration to be meaningful
# =============================================================================

def adf_test(series: pd.Series) -> tuple[float, bool]:
    """
    Returns (p-value, is_nonstationary).
    We WANT p > 0.05 here — series should NOT be stationary in levels.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    return p_value, p_value > 0.05   # True = non-stationary = good for coint

log.info("Running ADF unit root tests on all tickers...")
adf_results = {}
excluded_stationary = []

for ticker in log_prices.columns:
    pval, is_nonstationary = adf_test(log_prices[ticker])
    adf_results[ticker] = pval
    if not is_nonstationary:
        excluded_stationary.append(ticker)
        log.warning(f"  {ticker}: ADF p={pval:.4f} — appears stationary, "
                    f"will exclude from cointegration tests")

valid_tickers = [t for t in log_prices.columns if t not in excluded_stationary]
pharma_valid  = [t for t in pharma_tickers  if t in valid_tickers]
auto_valid    = [t for t in auto_tickers    if t in valid_tickers]

log.info(f"Valid for cointegration: {len(pharma_valid)} pharma, "
         f"{len(auto_valid)} auto ancillary\n")

# =============================================================================
# STEP 3: Engle-Granger two-step cointegration test
# =============================================================================

def engle_granger_test(y: pd.Series, x: pd.Series) -> dict:
    """
    Engle-Granger two-step method:
      Step 1: Regress y on x (OLS) to get the cointegrating residual
      Step 2: ADF test on the residual — if stationary, pair is cointegrated

    Also estimates hedge ratio (beta) and intercept.
    Note: EG is sensitive to variable ordering (y vs x), so we run both
    directions and take the more significant result.
    """
    def _run(y, x):
        x_c = add_constant(x)
        model = OLS(y, x_c).fit()
        beta  = model.params.iloc[1]
        alpha = model.params.iloc[0]
        resid = model.resid
        _, pval, _ = coint(y, x)   # statsmodels coint = EG test
        return pval, beta, alpha, resid

    pval_yx, beta_yx, alpha_yx, resid_yx = _run(y, x)
    pval_xy, beta_xy, alpha_xy, resid_xy = _run(x, y)

    # Take direction with lower p-value (more evidence of cointegration)
    if pval_yx <= pval_xy:
        return {
            "eg_pvalue": pval_yx, "eg_direction": f"{y.name}~{x.name}",
            "beta": beta_yx, "alpha": alpha_yx, "spread": resid_yx
        }
    else:
        # Invert beta so spread is always defined as y - beta*x
        return {
            "eg_pvalue": pval_xy, "eg_direction": f"{x.name}~{y.name}",
            "beta": 1.0 / beta_xy, "alpha": -alpha_xy / beta_xy,
            "spread": resid_xy * (-1.0 / beta_xy)
        }

# =============================================================================
# STEP 4: Johansen test
# =============================================================================

def johansen_test(y: pd.Series, x: pd.Series,
                  confidence: float = 0.05) -> dict:
    """
    Johansen trace test for cointegration rank.
    Unlike EG, Johansen is symmetric (order-invariant) and directly
    gives the cointegrating vector.

    confidence=0.05 uses the 95% critical values.
    Returns whether rank >= 1 (at least one cointegrating relationship).
    """
    data = pd.concat([y, x], axis=1).dropna()
    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        # Trace statistic vs critical values
        # result.cvt columns: [90%, 95%, 99%]
        cv_col = {0.10: 0, 0.05: 1, 0.01: 2}[confidence]
        trace_stat = result.lr1[0]       # Rank 0 test statistic
        trace_cv   = result.cvt[0, cv_col]
        cointegrated = trace_stat > trace_cv

        # Cointegrating vector (first eigenvector)
        coint_vector = result.evec[:, 0]

        return {
            "jo_trace_stat": trace_stat,
            "jo_trace_cv":   trace_cv,
            "jo_cointegrated": cointegrated,
            "jo_coint_vector": coint_vector
        }
    except Exception as e:
        return {
            "jo_trace_stat": np.nan, "jo_trace_cv": np.nan,
            "jo_cointegrated": False, "jo_coint_vector": None
        }

# =============================================================================
# STEP 5: Half-life of mean reversion (Ornstein-Uhlenbeck)
# =============================================================================

def compute_halflife(spread: pd.Series) -> float:
    """
    Fit an AR(1) to the spread: Δspread_t = λ * spread_{t-1} + ε_t
    The mean reversion speed is λ, and the half-life is:
        HL = -log(2) / log(1 + λ)  [in trading days]

    A negative λ confirms mean reversion (spread pulled back to zero).
    Half-life is the expected time for the spread to revert halfway.
    """
    spread = spread.dropna()
    spread_lag   = spread.shift(1).dropna()
    spread_diff  = spread.diff().dropna()

    # Align
    idx = spread_lag.index.intersection(spread_diff.index)
    spread_lag  = spread_lag.loc[idx]
    spread_diff = spread_diff.loc[idx]

    x = add_constant(spread_lag)
    model = OLS(spread_diff, x).fit()
    lam = model.params.iloc[1]

    if lam >= 0:
        return np.nan   # No mean reversion

    half_life = -np.log(2) / np.log(1 + lam)
    return half_life

# =============================================================================
# STEP 6: Run all within-sector pairs
# =============================================================================

def run_pair_tests(tickers: list, sector: str) -> pd.DataFrame:
    pairs = list(itertools.combinations(tickers, 2))
    log.info(f"[{sector}] Testing {len(pairs)} pairs...")

    results = []

    for i, (t1, t2) in enumerate(pairs):
        y = log_prices[t1]
        x = log_prices[t2]

        # Engle-Granger
        eg = engle_granger_test(y, x)

        # Johansen
        jo = johansen_test(y, x, confidence=JO_CONFIDENCE)

        # Half-life on EG spread
        hl = compute_halflife(eg["spread"])

        results.append({
            "ticker_1":        t1,
            "ticker_2":        t2,
            "sector":          sector,
            "eg_pvalue":       round(eg["eg_pvalue"], 6),
            "eg_direction":    eg["eg_direction"],
            "eg_significant":  eg["eg_pvalue"] < EG_PVALUE_THRESHOLD,
            "beta":            round(eg["beta"], 6),
            "jo_trace_stat":   round(jo["jo_trace_stat"], 4)
                               if not np.isnan(jo["jo_trace_stat"]) else np.nan,
            "jo_trace_cv":     round(jo["jo_trace_cv"], 4)
                               if not np.isnan(jo["jo_trace_cv"]) else np.nan,
            "jo_significant":  jo["jo_cointegrated"],
            "both_significant": eg["eg_pvalue"] < EG_PVALUE_THRESHOLD
                                and jo["jo_cointegrated"],
            "half_life_days":  round(hl, 2) if not np.isnan(hl) else np.nan,
            "hl_tradeable":    (not np.isnan(hl))
                               and MIN_HALFLIFE_DAYS <= hl <= MAX_HALFLIFE_DAYS,
            "spread_series":   eg["spread"]
        })

        if (i + 1) % 20 == 0:
            log.info(f"  {i+1}/{len(pairs)} pairs tested...")

    return pd.DataFrame(results)


pharma_results = run_pair_tests(pharma_valid, "pharma")
auto_results   = run_pair_tests(auto_valid,   "auto_ancillary")
all_results    = pd.concat([pharma_results, auto_results], ignore_index=True)

log.info(f"\nPharma:        {len(pharma_results)} pairs tested")
log.info(f"Auto ancillary: {len(auto_results)} pairs tested")

# =============================================================================
# STEP 7: Filter to cointegrated pairs
# =============================================================================

cointegrated = all_results[
    all_results["both_significant"] & all_results["hl_tradeable"]
].copy().sort_values("eg_pvalue").reset_index(drop=True)

log.info(f"\nPairs passing both EG + Johansen + half-life filter: "
         f"{len(cointegrated)}")

# =============================================================================
# STEP 8: Save spread series for passing pairs
# =============================================================================

for _, row in cointegrated.iterrows():
    label = f"{row['ticker_1'].replace('.NS','')}_{row['ticker_2'].replace('.NS','')}"
    spread_df = row["spread_series"].to_frame(name="spread")
    spread_df.to_parquet(f"{DATA_DIR}/spreads/{label}.parquet")

# Drop spread_series column before saving CSV (not serialisable)
save_cols = [c for c in all_results.columns if c != "spread_series"]
all_results[save_cols].to_csv(f"{DATA_DIR}/cointegration_results.csv", index=False)
cointegrated[save_cols].to_csv(f"{DATA_DIR}/cointegrated_pairs.csv", index=False)

log.info(f"Saved: {DATA_DIR}/cointegration_results.csv")
log.info(f"Saved: {DATA_DIR}/cointegrated_pairs.csv")

# =============================================================================
# STEP 9: P-value heatmaps (one per sector)
# =============================================================================

def plot_pvalue_heatmap(results_df: pd.DataFrame, tickers: list,
                        sector: str, filename: str):
    n = len(tickers)
    matrix = pd.DataFrame(np.ones((n, n)), index=tickers, columns=tickers)

    for _, row in results_df.iterrows():
        t1, t2 = row["ticker_1"], row["ticker_2"]
        if t1 in matrix.index and t2 in matrix.columns:
            matrix.loc[t1, t2] = row["eg_pvalue"]
            matrix.loc[t2, t1] = row["eg_pvalue"]

    # Clean up labels
    clean = lambda t: t.replace(".NS", "")
    matrix.index   = [clean(t) for t in matrix.index]
    matrix.columns = [clean(t) for t in matrix.columns]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    mask = np.eye(n, dtype=bool)   # Mask diagonal
    sns.heatmap(
        matrix, mask=mask, ax=ax,
        cmap="RdYlGn_r",
        vmin=0, vmax=0.20,
        annot=True, fmt=".3f",
        annot_kws={"size": 7},
        linewidths=0.3, linecolor="#E8E6DF",
        cbar_kws={"label": "EG p-value  (green = more cointegrated)"}
    )

    # Highlight significant cells
    for i in range(n):
        for j in range(n):
            if i != j:
                val = matrix.iloc[i, j]
                if val < EG_PVALUE_THRESHOLD:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        fill=False, edgecolor="#1D9E75",
                        linewidth=1.8, zorder=3
                    ))

    ax.set_title(
        f"{sector.replace('_', ' ').title()} — Engle-Granger p-value Matrix\n"
        f"Green border = significant at {EG_PVALUE_THRESHOLD*100:.0f}% level",
        fontsize=13, fontweight="medium", pad=14, color="#2C2C2A"
    )
    ax.tick_params(axis="both", labelsize=8, colors="#5F5E5A")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved heatmap: {filename}")


plot_pvalue_heatmap(
    pharma_results[save_cols], pharma_valid, "pharma",
    f"{PLOT_DIR}/pvalue_heatmap_pharma.png"
)
plot_pvalue_heatmap(
    auto_results[save_cols], auto_valid, "auto_ancillary",
    f"{PLOT_DIR}/pvalue_heatmap_auto.png"
)

# =============================================================================
# STEP 10: Spread plots for each cointegrated pair
# =============================================================================

def plot_spread(row: pd.Series, log_prices: pd.DataFrame,
                zscore_window: int, output_path: str):
    t1, t2    = row["ticker_1"], row["ticker_2"]
    spread    = row["spread_series"]
    half_life = row["half_life_days"]
    beta      = row["beta"]

    # Rolling Z-score
    roll_mean = spread.rolling(zscore_window).mean()
    roll_std  = spread.rolling(zscore_window).std()
    zscore    = (spread - roll_mean) / roll_std

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#FAFAF8")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                            height_ratios=[1.2, 1.2, 0.8])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    BLUE   = "#185FA5"
    GREEN  = "#1D9E75"
    AMBER  = "#BA7517"
    RED    = "#993556"

    def style(ax, title):
        ax.set_facecolor("#FAFAF8")
        ax.set_title(title, fontsize=11, fontweight="medium",
                     color="#2C2C2A", pad=8)
        ax.tick_params(colors="#888780", labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#D3D1C7")
        ax.grid(axis="y", color="#E8E6DF", linewidth=0.5, linestyle="--")

    label1 = t1.replace(".NS","")
    label2 = t2.replace(".NS","")

    # --- Panel 1: Log price series (normalised to 100) ---
    p1 = np.exp(log_prices[t1]) / np.exp(log_prices[t1].iloc[0]) * 100
    p2 = np.exp(log_prices[t2]) / np.exp(log_prices[t2].iloc[0]) * 100
    ax1.plot(p1.index, p1.values, color=BLUE,  linewidth=1.2,
             label=label1, alpha=0.9)
    ax1.plot(p2.index, p2.values, color=AMBER, linewidth=1.2,
             label=label2, alpha=0.9)
    style(ax1, f"Normalised Price Series  (rebased to 100)")
    ax1.legend(fontsize=8, framealpha=0.6)
    ax1.set_ylabel("Price (rebased)", fontsize=9, color="#5F5E5A")

    # --- Panel 2: Spread with mean and ±1σ bands ---
    ax2.plot(spread.index, spread.values, color=GREEN, linewidth=1.0,
             alpha=0.85, label="Spread")
    ax2.plot(roll_mean.index, roll_mean.values, color="#2C2C2A",
             linewidth=1.2, linestyle="--", alpha=0.6, label=f"{zscore_window}d mean")
    ax2.fill_between(spread.index,
                     roll_mean + roll_std, roll_mean - roll_std,
                     alpha=0.12, color=GREEN, label="±1σ band")
    style(ax2, f"Spread  (β={beta:.3f})  |  Half-life: {half_life:.1f} days")
    ax2.legend(fontsize=8, framealpha=0.6)
    ax2.set_ylabel("log-price spread", fontsize=9, color="#5F5E5A")

    # --- Panel 3: Z-score with entry/exit thresholds ---
    ax3.plot(zscore.index, zscore.values, color=BLUE, linewidth=0.9, alpha=0.85)
    ax3.axhline( 1.5, color=RED,   linewidth=1.0, linestyle="--", alpha=0.7,
                label="±1.5σ entry")
    ax3.axhline(-1.5, color=RED,   linewidth=1.0, linestyle="--", alpha=0.7)
    ax3.axhline( 3.0, color="#888780", linewidth=0.8, linestyle=":",
                alpha=0.6, label="±3σ stop")
    ax3.axhline(-3.0, color="#888780", linewidth=0.8, linestyle=":", alpha=0.6)
    ax3.axhline( 0.0, color="#2C2C2A", linewidth=0.8, alpha=0.4)
    ax3.fill_between(zscore.index, 1.5, zscore.values,
                     where=(zscore > 1.5),  alpha=0.15, color=RED)
    ax3.fill_between(zscore.index, zscore.values, -1.5,
                     where=(zscore < -1.5), alpha=0.15, color=GREEN)
    style(ax3, f"Rolling Z-score  ({zscore_window}d window)")
    ax3.legend(fontsize=8, framealpha=0.6, loc="upper right")
    ax3.set_ylabel("Z-score", fontsize=9, color="#5F5E5A")
    ax3.set_ylim(-5, 5)

    fig.suptitle(
        f"Cointegration Analysis: {label1} / {label2}  "
        f"|  EG p={row['eg_pvalue']:.4f}  "
        f"|  HL={half_life:.1f}d",
        fontsize=13, fontweight="medium", color="#2C2C2A", y=1.01
    )

    fig.text(0.5, -0.01,
             f"EG p-value: {row['eg_pvalue']:.5f}  |  "
             f"Johansen trace: {row['jo_trace_stat']:.2f} > CV {row['jo_trace_cv']:.2f}  |  "
             f"β={beta:.4f}  |  Half-life={half_life:.1f} trading days",
             ha="center", fontsize=8, color="#888780")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


log.info(f"\nGenerating spread plots for {len(cointegrated)} cointegrated pairs...")
for _, row in cointegrated.iterrows():
    label = (f"{row['ticker_1'].replace('.NS','')}_"
             f"{row['ticker_2'].replace('.NS','')}")
    plot_spread(row, log_prices, ZSCORE_WINDOW,
                f"{PLOT_DIR}/spread_{label}.png")
    log.info(f"  Saved: plots/spread_{label}.png")

# =============================================================================
# STEP 11: Summary printout
# =============================================================================

pharma_coint = cointegrated[cointegrated["sector"] == "pharma"]
auto_coint   = cointegrated[cointegrated["sector"] == "auto_ancillary"]

print(f"\n{'='*70}")
print(f"  Cointegration Results Summary")
print(f"{'='*70}")
print(f"\n  Significance thresholds:")
print(f"    Engle-Granger p-value  < {EG_PVALUE_THRESHOLD}")
print(f"    Johansen trace test    95% confidence")
print(f"    Half-life              {MIN_HALFLIFE_DAYS}–{MAX_HALFLIFE_DAYS} trading days")

print(f"\n  {'Pair':<45} {'EG p':>8} {'HL (days)':>10} {'β':>8}")
print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*8}")

for sector_name, subset in [("Pharma", pharma_coint),
                              ("Auto Ancillary", auto_coint)]:
    if subset.empty:
        continue
    print(f"\n  [{sector_name}]")
    for _, row in subset.iterrows():
        pair = (f"{row['ticker_1'].replace('.NS','')} / "
                f"{row['ticker_2'].replace('.NS','')}")
        print(f"  {pair:<45} {row['eg_pvalue']:>8.5f} "
              f"{row['half_life_days']:>10.1f} {row['beta']:>8.4f}")

print(f"\n  Total cointegrated pairs: {len(cointegrated)}")
print(f"  Pharma: {len(pharma_coint)}  |  Auto ancillary: {len(auto_coint)}")
print(f"\n  Outputs:")
print(f"    {DATA_DIR}/cointegration_results.csv  — all {len(all_results)} pairs")
print(f"    {DATA_DIR}/cointegrated_pairs.csv     — {len(cointegrated)} passing pairs")
print(f"    {PLOT_DIR}/pvalue_heatmap_pharma.png")
print(f"    {PLOT_DIR}/pvalue_heatmap_auto.png")
print(f"    {PLOT_DIR}/spread_*.png               — {len(cointegrated)} spread plots")
print(f"{'='*70}\n")
