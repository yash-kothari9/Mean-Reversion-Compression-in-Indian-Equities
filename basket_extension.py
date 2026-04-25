"""
Phase 5 — Johansen Basket Extension
=====================================
Extends from pairs to 3-stock baskets using Johansen rank-2 cointegration.
Tests all within-sector triplets, finds baskets with at least one
cointegrating vector, and runs the same backtester + rolling half-life
regime analysis as Phase 3.

Why this matters:
  Pairs cointegration assumes a single cointegrating relationship between
  two series. The Johansen test on a 3-stock basket can detect up to 2
  independent cointegrating vectors (rank 2), meaning the basket has a
  richer long-run equilibrium structure — more realistic for sector stocks
  driven by common fundamentals (API costs, regulatory environment, OEM demand).

  The basket spread is defined as:
      spread = w1*log(P1) + w2*log(P2) + w3*log(P3)
  where [w1, w2, w3] is the first cointegrating eigenvector, normalised
  so w1 = 1. This spread is mean-reverting by construction.

Outputs:
  data/basket_cointegration_results.csv
  data/cointegrated_baskets.csv
  data/basket_spreads/
  plots/basket_backtest_<label>.png
  plots/basket_rolling_hl_<label>.png
  plots/basket_regime_comparison.png
  results/basket_backtest_summary.csv
  results/basket_event_window_analysis.csv

Requirements:
    pip install statsmodels pandas numpy matplotlib pyarrow
"""

import os
import warnings
import logging
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
JO_CONFIDENCE     = 0.05
MIN_HALFLIFE_DAYS = 5
MAX_HALFLIFE_DAYS = 126
ENTRY_ZSCORE      = 1.5
EXIT_ZSCORE       = 0.0
STOP_ZSCORE       = 3.0
ZSCORE_WINDOW     = 60
HL_WINDOW         = 60
EVENT_WINDOW      = 30
TOTAL_COST_BPS    = 18        # 5 slip + 10 impact + 3 brokerage
ROUND_TRIP_COST   = 2 * TOTAL_COST_BPS / 10_000

DATA_DIR    = "data"
RESULTS_DIR = "results"
PLOT_DIR    = "plots"

os.makedirs(f"{DATA_DIR}/basket_spreads", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BG = "#FAFAF8"; BLUE = "#185FA5"; GREEN = "#1D9E75"
AMBER = "#BA7517"; RED = "#993556"; GRAY = "#888780"

# =============================================================================
# EVENT CALENDAR
# =============================================================================
EVENTS = [
    {"date": "2020-03-21", "label": "PLI Pharma scheme announced",    "sector": "pharma"},
    {"date": "2020-11-10", "label": "API Bulk Drug Park scheme",       "sector": "pharma"},
    {"date": "2021-02-01", "label": "Union Budget 2021 health alloc",  "sector": "pharma"},
    {"date": "2022-06-15", "label": "NPPA drug price revision",        "sector": "pharma"},
    {"date": "2023-03-28", "label": "PLI pharma tranche 2",            "sector": "pharma"},
    {"date": "2021-09-15", "label": "PLI Auto Components scheme",      "sector": "auto_ancillary"},
    {"date": "2022-04-01", "label": "BS-VI Phase 2 norms effective",   "sector": "auto_ancillary"},
    {"date": "2023-01-13", "label": "EV PLI incentive revised",        "sector": "auto_ancillary"},
]
events_df = pd.DataFrame(EVENTS)
events_df["date"] = pd.to_datetime(events_df["date"])

# =============================================================================
# LOAD DATA
# =============================================================================
prices     = pd.read_parquet(f"{DATA_DIR}/raw_prices.parquet")
log_prices = np.log(prices)
report     = pd.read_csv(f"{DATA_DIR}/universe_report.csv")
sector_map = dict(zip(report["ticker"], report["sector"]))

pharma_tickers = [t for t in log_prices.columns if sector_map.get(t) == "pharma"]
auto_tickers   = [t for t in log_prices.columns if sector_map.get(t) == "auto_ancillary"]
log.info(f"Pharma: {len(pharma_tickers)} | Auto: {len(auto_tickers)}\n")

# =============================================================================
# CORE: JOHANSEN BASKET TEST
# =============================================================================
def johansen_basket_test(t1, t2, t3, log_prices, confidence=0.05):
    """
    Johansen trace test on a 3-variable system.

    Two null hypotheses:
      H0: rank=0  -> rejected means at least 1 cointegrating vector
      H0: rank<=1 -> rejected means at least 2 cointegrating vectors (rank 2)

    We require rank >= 1. Rank 2 is stronger — two independent long-run
    equilibria exist among the three stocks.

    The first eigenvector (largest eigenvalue) gives the cointegrating weights,
    normalised so the first weight = 1 (standard convention).
    """
    data = log_prices[[t1, t2, t3]].dropna()
    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
    except Exception as e:
        return {"rank": 0, "cointegrated": False, "weights": None, "error": str(e)}

    cv_col      = {0.10: 0, 0.05: 1, 0.01: 2}[confidence]
    trace_stats = result.lr1
    trace_cvs   = result.cvt[:, cv_col]

    rank = 0
    if trace_stats[0] > trace_cvs[0]:
        rank = 1
    if len(trace_stats) > 1 and trace_stats[1] > trace_cvs[1]:
        rank = 2

    evec    = result.evec[:, 0]
    weights = evec / evec[0]   # normalise so w[0] = 1

    return {
        "rank":         rank,
        "cointegrated": rank >= 1,
        "rank2":        rank == 2,
        "trace_stat_0": round(float(trace_stats[0]), 4),
        "trace_cv_0":   round(float(trace_cvs[0]),   4),
        "trace_stat_1": round(float(trace_stats[1]), 4) if len(trace_stats) > 1 else np.nan,
        "trace_cv_1":   round(float(trace_cvs[1]),   4) if len(trace_cvs)   > 1 else np.nan,
        "weights":      weights,
        "error":        None,
    }


def compute_basket_spread(t1, t2, t3, weights, log_prices):
    data = log_prices[[t1, t2, t3]].dropna()
    return weights[0]*data[t1] + weights[1]*data[t2] + weights[2]*data[t3]


def compute_halflife(spread):
    spread = spread.dropna()
    lag    = spread.shift(1).dropna()
    diff   = spread.diff().dropna()
    idx    = lag.index.intersection(diff.index)
    try:
        lam = OLS(diff.loc[idx], add_constant(lag.loc[idx])).fit().params.iloc[1]
        return -np.log(2) / np.log(1 + lam) if lam < 0 else np.nan
    except Exception:
        return np.nan

# =============================================================================
# RUN ALL WITHIN-SECTOR TRIPLETS
# =============================================================================
def run_basket_tests(tickers, sector):
    triplets = list(itertools.combinations(tickers, 3))
    log.info(f"[{sector}] Testing {len(triplets)} triplets...")
    results = []
    for i, (t1, t2, t3) in enumerate(triplets):
        jo = johansen_basket_test(t1, t2, t3, log_prices, JO_CONFIDENCE)
        hl = np.nan
        if jo["cointegrated"] and jo["weights"] is not None:
            spread = compute_basket_spread(t1, t2, t3, jo["weights"], log_prices)
            hl     = compute_halflife(spread)
        w = jo["weights"] if jo["weights"] is not None else [np.nan, np.nan, np.nan]
        results.append({
            "ticker_1": t1, "ticker_2": t2, "ticker_3": t3, "sector": sector,
            "johansen_rank":  jo["rank"],
            "cointegrated":   jo["cointegrated"],
            "rank2":          jo.get("rank2", False),
            "trace_stat_0":   jo.get("trace_stat_0", np.nan),
            "trace_cv_0":     jo.get("trace_cv_0",   np.nan),
            "trace_stat_1":   jo.get("trace_stat_1", np.nan),
            "trace_cv_1":     jo.get("trace_cv_1",   np.nan),
            "w1": round(float(w[0]), 6),
            "w2": round(float(w[1]), 6),
            "w3": round(float(w[2]), 6),
            "half_life_days": round(hl, 2) if not np.isnan(hl) else np.nan,
            "hl_tradeable":   (not np.isnan(hl)) and MIN_HALFLIFE_DAYS <= hl <= MAX_HALFLIFE_DAYS,
        })
        if (i + 1) % 100 == 0:
            log.info(f"  {i+1}/{len(triplets)} done...")
    return pd.DataFrame(results)


pharma_res = run_basket_tests(pharma_tickers, "pharma")
auto_res   = run_basket_tests(auto_tickers,   "auto_ancillary")
all_res    = pd.concat([pharma_res, auto_res], ignore_index=True)

cointegrated_baskets = all_res[
    all_res["cointegrated"] & all_res["hl_tradeable"]
].copy().sort_values(
    ["rank2", "trace_stat_0"], ascending=[False, False]
).reset_index(drop=True)

log.info(f"\nCointegrated baskets: {len(cointegrated_baskets)}  "
         f"(rank-2: {int(cointegrated_baskets['rank2'].sum())})\n")

all_res.to_csv(f"{DATA_DIR}/basket_cointegration_results.csv", index=False)
cointegrated_baskets.to_csv(f"{DATA_DIR}/cointegrated_baskets.csv", index=False)

# =============================================================================
# BACKTESTER + ROLLING HALF-LIFE + EVENT WINDOWS
# =============================================================================
def compute_zscore(spread, window):
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()


def compute_halflife_rolling(spread, window):
    hl = {}
    for i in range(window, len(spread)):
        s   = spread.iloc[i - window: i]
        lag = s.shift(1).dropna()
        dif = s.diff().dropna()
        idx = lag.index.intersection(dif.index)
        try:
            lam = OLS(dif.loc[idx], add_constant(lag.loc[idx])).fit().params.iloc[1]
            hl[spread.index[i]] = -np.log(2) / np.log(1 + lam) if lam < 0 else np.nan
        except Exception:
            hl[spread.index[i]] = np.nan
    return pd.Series(hl)


def backtest_basket(spread):
    zscore   = compute_zscore(spread, ZSCORE_WINDOW)
    position = pd.Series(0, index=zscore.index, dtype=float)
    pos = 0
    for i in range(1, len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            position.iloc[i] = 0
            continue
        if pos == 0:
            if z > ENTRY_ZSCORE:    pos = -1
            elif z < -ENTRY_ZSCORE: pos =  1
        elif pos == 1:
            if z > -EXIT_ZSCORE or abs(z) > STOP_ZSCORE: pos = 0
        elif pos == -1:
            if z < EXIT_ZSCORE  or abs(z) > STOP_ZSCORE: pos = 0
        position.iloc[i] = pos

    spread_ret = spread.diff()
    strat_ret  = position.shift(1) * spread_ret
    trades     = position.diff().abs()
    cost_daily = trades * ROUND_TRIP_COST / 2 * 1.5   # 3 legs vs 2
    strat_net  = strat_ret - cost_daily
    cum        = strat_ret.cumsum()
    cum_net    = strat_net.cumsum()
    drawdown   = cum_net - cum_net.cummax()
    daily      = strat_net.dropna()
    sharpe     = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    n_trades   = int(trades.sum() / 2)
    active     = strat_net[position.shift(1) != 0]
    wins, losses = (active > 0).sum(), (active < 0).sum()
    hit_rate   = wins / (wins + losses) if (wins + losses) > 0 else np.nan

    return {
        "metrics": {
            "sharpe":    round(sharpe, 4),
            "total_ret": round(cum_net.iloc[-1], 6),
            "gross_ret": round(cum.iloc[-1], 6),
            "max_dd":    round(drawdown.min(), 6),
            "n_trades":  n_trades,
            "hit_rate":  round(hit_rate, 4),
            "cost_drag": round(cum.iloc[-1] - cum_net.iloc[-1], 6),
        },
        "series": {
            "spread": spread, "zscore": zscore, "position": position,
            "cum_pnl": cum, "cum_pnl_net": cum_net, "drawdown": drawdown,
        }
    }


def event_window_analysis(rolling_hl, events, sector, window):
    rows = []
    for _, ev in events[events["sector"] == sector].iterrows():
        d    = ev["date"]
        pre  = rolling_hl.loc[
            (rolling_hl.index >= d - pd.Timedelta(days=window)) &
            (rolling_hl.index <  d)].dropna()
        post = rolling_hl.loc[
            (rolling_hl.index >  d) &
            (rolling_hl.index <= d + pd.Timedelta(days=window))].dropna()
        if len(pre) < 5 or len(post) < 5:
            continue
        pm, pom = pre.mean(), post.mean()
        chg = pom - pm
        rows.append({
            "event_date":    d.date(),
            "event_label":   ev["label"],
            "pre_hl_mean":   round(pm,  2),
            "post_hl_mean":  round(pom, 2),
            "hl_change":     round(chg, 2),
            "hl_pct_change": round(chg / pm * 100, 2),
            "direction":     "compress" if chg < 0 else "expand",
            "n_pre":         len(pre),
            "n_post":        len(post),
        })
    return pd.DataFrame(rows)

# =============================================================================
# PLOTTING
# =============================================================================
def style_ax(ax, title, ylabel=None):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=10, fontweight="medium", color="#2C2C2A", pad=7)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.grid(axis="y", color="#E8E6DF", linewidth=0.5, linestyle="--")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#5F5E5A")


def plot_basket_backtest(label, names, series, metrics, rank, weights):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5,
                            height_ratios=[1, 1.2, 0.7, 0.7])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    axes[0].plot(series["zscore"].index, series["zscore"].values,
                 color=BLUE, linewidth=0.8, alpha=0.85)
    axes[0].axhline( ENTRY_ZSCORE, color=RED,   linewidth=0.9, linestyle="--", alpha=0.7)
    axes[0].axhline(-ENTRY_ZSCORE, color=GREEN,  linewidth=0.9, linestyle="--", alpha=0.7)
    axes[0].axhline(0, color="#2C2C2A", linewidth=0.6, alpha=0.3)
    axes[0].fill_between(series["position"].index, -5, 5,
                         where=(series["position"] == 1),  alpha=0.08, color=GREEN)
    axes[0].fill_between(series["position"].index, -5, 5,
                         where=(series["position"] == -1), alpha=0.08, color=RED)
    axes[0].set_ylim(-5, 5)
    style_ax(axes[0], "Basket Z-score  (green=long, red=short)", "Z-score")

    axes[1].plot(series["cum_pnl"].index, series["cum_pnl"].values,
                 color=GRAY, linewidth=0.9, linestyle="--", alpha=0.7, label="Gross")
    axes[1].plot(series["cum_pnl_net"].index, series["cum_pnl_net"].values,
                 color=BLUE, linewidth=1.2, label="Net (after costs)")
    axes[1].axhline(0, color="#2C2C2A", linewidth=0.6, alpha=0.3)
    axes[1].legend(fontsize=8, framealpha=0.6)
    style_ax(axes[1], "Cumulative P&L", "Cum. P&L")

    axes[2].fill_between(series["drawdown"].index,
                         series["drawdown"].values, 0, color=RED, alpha=0.35)
    axes[2].plot(series["drawdown"].index, series["drawdown"].values,
                 color=RED, linewidth=0.8)
    style_ax(axes[2], f"Drawdown  (max: {metrics['max_dd']:.4f})", "Drawdown")

    axes[3].fill_between(series["position"].index, series["position"].values, 0,
                         where=(series["position"] > 0),  color=GREEN, alpha=0.5, label="Long")
    axes[3].fill_between(series["position"].index, series["position"].values, 0,
                         where=(series["position"] < 0),  color=RED,   alpha=0.5, label="Short")
    axes[3].set_ylim(-1.5, 1.5)
    axes[3].legend(fontsize=8, framealpha=0.6)
    style_ax(axes[3], "Position", "")

    w_str = f"({weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f})"
    fig.suptitle(
        f"Basket: {' / '.join(names)}  |  Rank={rank}  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  Hit: {metrics['hit_rate']*100:.1f}%",
        fontsize=11, fontweight="medium", color="#2C2C2A", y=1.01)
    fig.text(0.5, -0.01,
             f"Weights {w_str}  |  Net: {metrics['total_ret']:.4f}  |  "
             f"DD: {metrics['max_dd']:.4f}  |  Trades: {metrics['n_trades']}",
             ha="center", fontsize=7.5, color=GRAY)
    plt.savefig(f"{PLOT_DIR}/basket_backtest_{label}.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_basket_rolling_hl(label, names, rolling_hl, sector, static_hl, rank):
    relevant = events_df[events_df["sector"] == sector]
    fig, ax  = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    ax.plot(rolling_hl.index, rolling_hl.values, color=BLUE, linewidth=1.1,
            alpha=0.85, label=f"Rolling {HL_WINDOW}d HL")
    ax.axhline(static_hl, color=GRAY, linewidth=1.0, linestyle="--",
               alpha=0.6, label=f"Full-sample HL: {static_hl:.1f}d")
    ax.axhline(126, color=RED,   linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(5,   color=GREEN, linewidth=0.8, linestyle=":", alpha=0.5)

    ev_colors = [AMBER, RED, GREEN, "#9B59B6", "#E67E22"]
    for i, (_, ev) in enumerate(relevant.iterrows()):
        c   = ev_colors[i % len(ev_colors)]
        ax.axvline(ev["date"], color=c, linewidth=1.4, linestyle="-.", alpha=0.75)
        idx = min(rolling_hl.index.searchsorted(ev["date"]), len(rolling_hl) - 1)
        yval = rolling_hl.iloc[idx] if len(rolling_hl) > 0 else static_hl
        if np.isnan(yval):
            yval = static_hl
        ax.annotate(ev["label"][:28], xy=(ev["date"], yval),
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=7, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6))

    ymax = rolling_hl.dropna().max() if not rolling_hl.dropna().empty else 150
    ax.set_ylim(0, min(ymax * 1.4 + 20, 250))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D3D1C7")
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.grid(axis="y", color="#E8E6DF", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Half-life (days)", fontsize=9, color="#5F5E5A")
    ax.set_title(f"Basket Rolling HL: {' / '.join(names)}  (rank={rank})",
                 fontsize=11, fontweight="medium", color="#2C2C2A", pad=8)
    ax.legend(fontsize=8, framealpha=0.6, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/basket_rolling_hl_{label}.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_basket_regime_comparison(ew_df):
    if ew_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    for ax, sector in zip(axes, ["pharma", "auto_ancillary"]):
        subset = ew_df[ew_df["sector"] == sector].reset_index(drop=True)
        ax.set_facecolor(BG)
        if subset.empty:
            ax.text(0.5, 0.5, "No event data", ha="center",
                    transform=ax.transAxes, color=GRAY)
            ax.set_title(sector.replace("_", " ").title(), fontsize=11)
            continue
        x = np.arange(len(subset)); width = 0.35
        ax.bar(x - width/2, subset["pre_hl_mean"],
               width, color=BLUE, alpha=0.75, label="Pre-event")
        for i, (_, row) in enumerate(subset.iterrows()):
            c = GREEN if row["direction"] == "compress" else RED
            ax.bar(i + width/2, row["post_hl_mean"], width, color=c, alpha=0.8)
            ax.annotate(f"{row['hl_pct_change']:+.1f}%",
                        xy=(i + width/2, row["post_hl_mean"]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color=c, fontweight="medium")
        xlabels = [f"{r['event_label'][:18]}\n{r.get('basket_label','')}"
                   for _, r in subset.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=6, rotation=30, ha="right")
        ax.set_ylabel("Mean half-life (days)", fontsize=9, color="#5F5E5A")
        ax.set_title(f"{sector.replace('_',' ').title()} Baskets\nHL Pre vs Post (Green=compress)",
                     fontsize=10, fontweight="medium", color="#2C2C2A")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#D3D1C7")
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.grid(axis="y", color="#E8E6DF", linewidth=0.5, linestyle="--")
        handles = [
            mpatches.Patch(color=BLUE,  alpha=0.75, label="Pre-event"),
            mpatches.Patch(color=GREEN, alpha=0.8,  label="Post (compress)"),
            mpatches.Patch(color=RED,   alpha=0.8,  label="Post (expand)"),
        ]
        ax.legend(handles=handles, fontsize=7, framealpha=0.6)
    fig.suptitle("Basket Extension — Regime Analysis: Half-Life Before vs After Policy Events",
                 fontsize=12, fontweight="medium", color="#2C2C2A", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/basket_regime_comparison.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"Saved: {PLOT_DIR}/basket_regime_comparison.png")

# =============================================================================
# MAIN LOOP — top 15 baskets by Johansen trace statistic
# =============================================================================
if cointegrated_baskets.empty:
    log.warning("No cointegrated baskets found. Try relaxing half-life bounds.")
else:
    backtest_summary = []
    all_ew           = []
    top_baskets      = cointegrated_baskets.head(15)
    log.info(f"Running backtest for top {len(top_baskets)} baskets...\n")

    for _, row in top_baskets.iterrows():
        t1, t2, t3 = row["ticker_1"], row["ticker_2"], row["ticker_3"]
        sector      = row["sector"]
        weights     = np.array([row["w1"], row["w2"], row["w3"]])
        static_hl   = row["half_life_days"]
        rank        = int(row["johansen_rank"])
        names       = [x.replace(".NS", "") for x in [t1, t2, t3]]
        label       = "__".join(names)

        log.info(f"Basket: {' / '.join(names)}  (rank={rank}, HL={static_hl:.1f}d)")

        spread = compute_basket_spread(t1, t2, t3, weights, log_prices)
        spread.to_csv(f"{DATA_DIR}/basket_spreads/{label}.csv", header=True)

        bt = backtest_basket(spread)
        plot_basket_backtest(label, names, bt["series"], bt["metrics"], rank, weights)

        rolling_hl = compute_halflife_rolling(spread, HL_WINDOW)
        rolling_hl.to_csv(
            f"{RESULTS_DIR}/rolling_halflife/basket_{label}.csv", header=True)
        plot_basket_rolling_hl(label, names, rolling_hl, sector, static_hl, rank)

        ew = event_window_analysis(rolling_hl, events_df, sector, EVENT_WINDOW)
        if not ew.empty:
            ew["basket_label"] = "/".join(names[:2]) + f"\n/{names[2]}"
            ew["ticker_1"] = t1; ew["ticker_2"] = t2
            ew["ticker_3"] = t3; ew["sector"]   = sector
            all_ew.append(ew)

        backtest_summary.append({
            "ticker_1": t1, "ticker_2": t2, "ticker_3": t3,
            "sector": sector, "johansen_rank": rank, "static_hl": static_hl,
            "w1": round(float(weights[0]), 4),
            "w2": round(float(weights[1]), 4),
            "w3": round(float(weights[2]), 4),
            **bt["metrics"]
        })
        log.info(f"  Sharpe:{bt['metrics']['sharpe']:.2f}  "
                 f"Hit:{bt['metrics']['hit_rate']*100:.1f}%  "
                 f"Trades:{bt['metrics']['n_trades']}")

    summary_df = pd.DataFrame(backtest_summary)
    summary_df.to_csv(f"{RESULTS_DIR}/basket_backtest_summary.csv", index=False)

    if all_ew:
        ew_df = pd.concat(all_ew, ignore_index=True)
        ew_df.to_csv(f"{RESULTS_DIR}/basket_event_window_analysis.csv", index=False)
        plot_basket_regime_comparison(ew_df)
    else:
        ew_df = pd.DataFrame()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_triplets = len(pharma_res) + len(auto_res)
    n_coint  = len(cointegrated_baskets)
    n_rank2  = int(cointegrated_baskets["rank2"].sum())

    print(f"\n{'='*76}")
    print(f"  Phase 5 — Johansen Basket Extension Summary")
    print(f"{'='*76}")
    print(f"\n  Triplets tested:       {total_triplets}")
    print(f"  Cointegrated baskets:  {n_coint}  "
          f"({n_coint/total_triplets*100:.1f}% hit rate)")
    print(f"  of which rank-2:       {n_rank2}")
    print(f"  Pharma:        "
          f"  {(cointegrated_baskets['sector']=='pharma').sum()} baskets")
    print(f"  Auto ancillary:"
          f"  {(cointegrated_baskets['sector']=='auto_ancillary').sum()} baskets")

    print(f"\n  {'Basket':<50} {'Rank':>5} {'HL':>6} "
          f"{'Sharpe':>7} {'Hit%':>6} {'Net Ret':>9}")
    print(f"  {'-'*50} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*9}")
    for _, r in summary_df.sort_values("sharpe", ascending=False).iterrows():
        ns = [r["ticker_1"].replace(".NS",""),
              r["ticker_2"].replace(".NS",""),
              r["ticker_3"].replace(".NS","")]
        print(f"  {' / '.join(ns):<50} {int(r['johansen_rank']):>5} "
              f"{r['static_hl']:>6.1f} {r['sharpe']:>7.2f} "
              f"{r['hit_rate']*100:>5.1f}% {r['total_ret']:>+9.4f}")

    if not ew_df.empty:
        comp = (ew_df["direction"] == "compress").sum()
        exp  = (ew_df["direction"] == "expand").sum()
        print(f"\n  Event windows — "
              f"Compression: {comp}/{comp+exp}  |  Expansion: {exp}/{comp+exp}")

    print(f"\n  Plots:   {PLOT_DIR}/basket_*.png")
    print(f"  Results: {RESULTS_DIR}/basket_backtest_summary.csv")
    print(f"{'='*76}\n")
