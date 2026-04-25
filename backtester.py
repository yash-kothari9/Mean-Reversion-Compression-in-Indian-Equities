"""
Phase 3 — Backtester & Rolling Half-Life Regime Analysis
=========================================================
Runs a vectorized pairs trading backtest on all cointegrated pairs
from Phase 2, with a realistic cost model.

The research layer: re-estimates the Ornstein-Uhlenbeck half-life on
a rolling 60-day window and measures whether it shifts around key
Indian pharma and auto ancillary policy event dates.

This is the core research contribution of the project.

Outputs:
  results/backtest_summary.csv         — per-pair performance metrics
  results/rolling_halflife/            — rolling HL time series per pair
  plots/backtest_equity_<pair>.png     — equity curve + drawdown per pair
  plots/rolling_hl_<pair>.png          — rolling HL with event markers
  plots/regime_comparison.png          — pre vs post event bar chart
  results/event_window_analysis.csv    — half-life shifts around events

Requirements:
    pip install pandas numpy matplotlib scipy statsmodels pyarrow
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

ENTRY_ZSCORE    = 1.5
EXIT_ZSCORE     = 0.0
STOP_ZSCORE     = 3.0
ZSCORE_WINDOW   = 60

SLIPPAGE_BPS    = 5
IMPACT_BPS      = 10
BROKERAGE_BPS   = 3
TOTAL_COST_BPS  = SLIPPAGE_BPS + IMPACT_BPS + BROKERAGE_BPS
ROUND_TRIP_COST = 2 * TOTAL_COST_BPS / 10_000

HL_WINDOW       = 60
EVENT_WINDOW    = 30

DATA_DIR    = "data"
RESULTS_DIR = "results"
PLOT_DIR    = "plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/rolling_halflife", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

BG    = "#FAFAF8"
BLUE  = "#185FA5"
GREEN = "#1D9E75"
AMBER = "#BA7517"
RED   = "#993556"
GRAY  = "#888780"

# =============================================================================
# POLICY EVENT CALENDAR
# =============================================================================

EVENTS = [
    {"date": "2020-03-21", "label": "PLI Pharma scheme announced",
     "sector": "pharma"},
    {"date": "2020-11-10", "label": "API Bulk Drug Park scheme notified",
     "sector": "pharma"},
    {"date": "2021-02-01", "label": "Union Budget 2021 — health allocation",
     "sector": "pharma"},
    {"date": "2022-06-15", "label": "NPPA drug price revision order",
     "sector": "pharma"},
    {"date": "2023-03-28", "label": "PLI pharma tranche 2 disbursement",
     "sector": "pharma"},
    {"date": "2021-09-15", "label": "PLI Auto Components scheme",
     "sector": "auto_ancillary"},
    {"date": "2022-04-01", "label": "BS-VI Phase 2 norms effective",
     "sector": "auto_ancillary"},
    {"date": "2023-01-13", "label": "EV PLI incentive revised",
     "sector": "auto_ancillary"},
]

events_df = pd.DataFrame(EVENTS)
events_df["date"] = pd.to_datetime(events_df["date"])

# =============================================================================
# LOAD DATA
# =============================================================================

prices     = pd.read_parquet(f"{DATA_DIR}/raw_prices.parquet")
log_prices = np.log(prices)
pairs_df   = pd.read_csv(f"{DATA_DIR}/cointegrated_pairs.csv")

log.info(f"Loaded {len(pairs_df)} cointegrated pairs\n")

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_spread(log_prices, t1, t2, beta):
    return log_prices[t1] - beta * log_prices[t2]


def compute_zscore(spread, window):
    mu  = spread.rolling(window).mean()
    sig = spread.rolling(window).std()
    return (spread - mu) / sig


def compute_halflife_rolling(spread, window):
    """Re-estimate OU half-life on a rolling window."""
    hl_values = {}
    for i in range(window, len(spread)):
        s   = spread.iloc[i - window: i]
        lag = s.shift(1).dropna()
        dif = s.diff().dropna()
        idx = lag.index.intersection(dif.index)
        try:
            lam = OLS(dif.loc[idx], add_constant(lag.loc[idx])).fit().params.iloc[1]
            hl  = -np.log(2) / np.log(1 + lam) if lam < 0 else np.nan
        except Exception:
            hl = np.nan
        hl_values[spread.index[i]] = hl
    return pd.Series(hl_values)


def backtest_pair(t1, t2, beta, log_prices):
    spread = compute_spread(log_prices, t1, t2, beta)
    zscore = compute_zscore(spread, ZSCORE_WINDOW)

    position = pd.Series(0, index=zscore.index, dtype=float)
    pos = 0

    for i in range(1, len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            position.iloc[i] = 0
            continue
        if pos == 0:
            if z > ENTRY_ZSCORE:
                pos = -1
            elif z < -ENTRY_ZSCORE:
                pos = 1
        elif pos == 1:
            if z > -EXIT_ZSCORE or abs(z) > STOP_ZSCORE:
                pos = 0
        elif pos == -1:
            if z < EXIT_ZSCORE or abs(z) > STOP_ZSCORE:
                pos = 0
        position.iloc[i] = pos

    spread_ret    = spread.diff()
    strat_ret     = position.shift(1) * spread_ret
    trades        = position.diff().abs()
    cost_daily    = trades * ROUND_TRIP_COST / 2
    strat_ret_net = strat_ret - cost_daily

    cum_pnl     = strat_ret.cumsum()
    cum_pnl_net = strat_ret_net.cumsum()
    drawdown    = cum_pnl_net - cum_pnl_net.cummax()

    daily_ret = strat_ret_net.dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                 if daily_ret.std() > 0 else np.nan)
    n_trades  = int(trades.sum() / 2)
    active    = strat_ret_net[position.shift(1) != 0]
    wins      = (active > 0).sum()
    losses    = (active < 0).sum()
    hit_rate  = wins / (wins + losses) if (wins + losses) > 0 else np.nan

    return {
        "metrics": {
            "sharpe":    round(sharpe, 4),
            "total_ret": round(cum_pnl_net.iloc[-1], 6),
            "gross_ret": round(cum_pnl.iloc[-1], 6),
            "max_dd":    round(drawdown.min(), 6),
            "n_trades":  n_trades,
            "hit_rate":  round(hit_rate, 4),
            "cost_drag": round(cum_pnl.iloc[-1] - cum_pnl_net.iloc[-1], 6),
        },
        "series": {
            "spread":      spread,
            "zscore":      zscore,
            "position":    position,
            "cum_pnl":     cum_pnl,
            "cum_pnl_net": cum_pnl_net,
            "drawdown":    drawdown,
        }
    }


def event_window_analysis(rolling_hl, events, sector, window):
    relevant = events[events["sector"] == sector]
    rows = []
    for _, ev in relevant.iterrows():
        d = ev["date"]
        pre  = rolling_hl.loc[(rolling_hl.index >= d - pd.Timedelta(days=window)) &
                               (rolling_hl.index <  d)].dropna()
        post = rolling_hl.loc[(rolling_hl.index >  d) &
                               (rolling_hl.index <= d + pd.Timedelta(days=window))].dropna()
        if len(pre) < 5 or len(post) < 5:
            continue
        pre_m, post_m = pre.mean(), post.mean()
        chg   = post_m - pre_m
        pct   = chg / pre_m * 100
        rows.append({
            "event_date":    d.date(),
            "event_label":   ev["label"],
            "pre_hl_mean":   round(pre_m, 2),
            "post_hl_mean":  round(post_m, 2),
            "hl_change":     round(chg, 2),
            "hl_pct_change": round(pct, 2),
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
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#D3D1C7")
    ax.grid(axis="y", color="#E8E6DF", linewidth=0.5, linestyle="--")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#5F5E5A")


def plot_backtest(label, t1, t2, series, metrics):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5,
                            height_ratios=[1, 1.2, 0.7, 0.7])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # Z-score
    axes[0].plot(series["zscore"].index, series["zscore"].values,
                 color=BLUE, linewidth=0.8, alpha=0.85)
    axes[0].axhline( ENTRY_ZSCORE, color=RED,   linewidth=0.9,
                    linestyle="--", alpha=0.7)
    axes[0].axhline(-ENTRY_ZSCORE, color=GREEN,  linewidth=0.9,
                    linestyle="--", alpha=0.7)
    axes[0].axhline(0, color="#2C2C2A", linewidth=0.6, alpha=0.3)
    axes[0].fill_between(series["position"].index, -5, 5,
                         where=(series["position"] == 1),
                         alpha=0.08, color=GREEN)
    axes[0].fill_between(series["position"].index, -5, 5,
                         where=(series["position"] == -1),
                         alpha=0.08, color=RED)
    axes[0].set_ylim(-5, 5)
    style_ax(axes[0], "Z-score  (green = long spread, red = short)", "Z-score")

    # Equity curves
    axes[1].plot(series["cum_pnl"].index, series["cum_pnl"].values,
                 color=GRAY, linewidth=0.9, linestyle="--",
                 alpha=0.7, label="Gross P&L")
    axes[1].plot(series["cum_pnl_net"].index, series["cum_pnl_net"].values,
                 color=BLUE, linewidth=1.2, label="Net P&L (after costs)")
    axes[1].axhline(0, color="#2C2C2A", linewidth=0.6, alpha=0.3)
    axes[1].legend(fontsize=8, framealpha=0.6)
    style_ax(axes[1], "Cumulative P&L  (log-spread units)", "Cum. P&L")

    # Drawdown
    axes[2].fill_between(series["drawdown"].index,
                         series["drawdown"].values, 0,
                         color=RED, alpha=0.35)
    axes[2].plot(series["drawdown"].index, series["drawdown"].values,
                 color=RED, linewidth=0.8)
    style_ax(axes[2], f"Drawdown  (max: {metrics['max_dd']:.4f})", "Drawdown")

    # Position
    axes[3].fill_between(series["position"].index,
                         series["position"].values, 0,
                         where=(series["position"] > 0),
                         color=GREEN, alpha=0.5, label="Long spread")
    axes[3].fill_between(series["position"].index,
                         series["position"].values, 0,
                         where=(series["position"] < 0),
                         color=RED, alpha=0.5, label="Short spread")
    axes[3].set_ylim(-1.5, 1.5)
    axes[3].legend(fontsize=8, framealpha=0.6, loc="upper right")
    style_ax(axes[3], "Position", "")

    fig.suptitle(
        f"Backtest: {t1.replace('.NS','')} / {t2.replace('.NS','')}  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  "
        f"Hit rate: {metrics['hit_rate']*100:.1f}%  |  "
        f"Trades: {metrics['n_trades']}",
        fontsize=12, fontweight="medium", color="#2C2C2A", y=1.01
    )
    fig.text(0.5, -0.01,
             f"Net return: {metrics['total_ret']:.4f}  |  "
             f"Cost drag: {metrics['cost_drag']:.4f}  |  "
             f"Max DD: {metrics['max_dd']:.4f}  |  "
             f"Cost model: {TOTAL_COST_BPS} bps/leg",
             ha="center", fontsize=8, color=GRAY)

    plt.savefig(f"{PLOT_DIR}/backtest_{label}.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def plot_rolling_halflife(label, t1, t2, rolling_hl, sector, static_hl):
    relevant = events_df[events_df["sector"] == sector]
    fig, ax  = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(rolling_hl.index, rolling_hl.values,
            color=BLUE, linewidth=1.1, alpha=0.85,
            label=f"Rolling {HL_WINDOW}d half-life")
    ax.axhline(static_hl, color=GRAY, linewidth=1.0,
               linestyle="--", alpha=0.6,
               label=f"Full-sample HL: {static_hl:.1f}d")
    ax.axhline(126, color=RED, linewidth=0.8,
               linestyle=":", alpha=0.5, label="126d bound")
    ax.axhline(5, color=GREEN, linewidth=0.8,
               linestyle=":", alpha=0.5, label="5d bound")

    ev_colors = [AMBER, RED, GREEN, "#9B59B6", "#E67E22"]
    for i, (_, ev) in enumerate(relevant.iterrows()):
        c = ev_colors[i % len(ev_colors)]
        ax.axvline(ev["date"], color=c, linewidth=1.4,
                   linestyle="-.", alpha=0.75)
        # Find nearest rolling HL value to annotate
        idx = rolling_hl.index.searchsorted(ev["date"])
        idx = min(idx, len(rolling_hl) - 1)
        yval = rolling_hl.iloc[idx] if len(rolling_hl) > 0 else 50
        if np.isnan(yval):
            yval = static_hl
        ax.annotate(
            ev["label"][:28],
            xy=(ev["date"], yval),
            xytext=(8, 12), textcoords="offset points",
            fontsize=7, color=c,
            arrowprops=dict(arrowstyle="-", color=c, lw=0.6)
        )

    ymax = rolling_hl.dropna().max() if not rolling_hl.dropna().empty else 150
    ax.set_ylim(0, min(ymax * 1.4 + 20, 250))
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#D3D1C7")
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.grid(axis="y", color="#E8E6DF", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Half-life (trading days)", fontsize=9, color="#5F5E5A")
    ax.set_title(
        f"Rolling Half-Life of Mean Reversion: "
        f"{t1.replace('.NS','')} / {t2.replace('.NS','')}",
        fontsize=11, fontweight="medium", color="#2C2C2A", pad=8
    )
    ax.legend(fontsize=8, framealpha=0.6, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/rolling_hl_{label}.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def plot_regime_comparison(ew_df):
    if ew_df.empty:
        log.warning("No event window data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    for ax, sector in zip(axes, ["pharma", "auto_ancillary"]):
        subset = ew_df[ew_df["sector"] == sector].reset_index(drop=True)
        ax.set_facecolor(BG)

        if subset.empty:
            ax.text(0.5, 0.5, "No event data for this sector",
                    ha="center", transform=ax.transAxes, color=GRAY)
            ax.set_title(sector.replace("_", " ").title(), fontsize=11)
            continue

        x     = np.arange(len(subset))
        width = 0.35

        ax.bar(x - width/2, subset["pre_hl_mean"],
               width, color=BLUE, alpha=0.75, label="Pre-event HL")

        for i, (_, row) in enumerate(subset.iterrows()):
            c = GREEN if row["direction"] == "compress" else RED
            bar = ax.bar(i + width/2, row["post_hl_mean"],
                         width, color=c, alpha=0.8)
            ax.annotate(
                f"{row['hl_pct_change']:+.1f}%",
                xy=(i + width/2, row["post_hl_mean"]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=7, color=c, fontweight="medium"
            )

        xlabels = [
            f"{r['event_label'][:20]}\n"
            f"{r['ticker_1'].replace('.NS','')}/"
            f"{r['ticker_2'].replace('.NS','')}"
            for _, r in subset.iterrows()
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=6.5, rotation=25, ha="right")
        ax.set_ylabel("Mean half-life (days)", fontsize=9, color="#5F5E5A")
        ax.set_title(
            f"{sector.replace('_',' ').title()} — HL Pre vs Post Event\n"
            "(Green post = compression, Red = expansion)",
            fontsize=10, fontweight="medium", color="#2C2C2A"
        )
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#D3D1C7")
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.grid(axis="y", color="#E8E6DF", linewidth=0.5, linestyle="--")

        handles = [
            mpatches.Patch(color=BLUE,  alpha=0.75, label="Pre-event"),
            mpatches.Patch(color=GREEN, alpha=0.8,  label="Post-event (compress)"),
            mpatches.Patch(color=RED,   alpha=0.8,  label="Post-event (expand)"),
        ]
        ax.legend(handles=handles, fontsize=7, framealpha=0.6)

    fig.suptitle(
        "Core Research Finding: Does Policy Shock Alter Mean Reversion Speed?\n"
        "Half-Life Before vs After Key Regulatory Events",
        fontsize=12, fontweight="medium", color="#2C2C2A", y=1.02
    )
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/regime_comparison.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved: {PLOT_DIR}/regime_comparison.png")

# =============================================================================
# MAIN LOOP
# =============================================================================

backtest_summary  = []
all_event_windows = []

for _, pair in pairs_df.iterrows():
    t1, t2    = pair["ticker_1"], pair["ticker_2"]
    beta      = pair["beta"]
    sector    = pair["sector"]
    static_hl = pair["half_life_days"]
    label     = f"{t1.replace('.NS','')}__{t2.replace('.NS','')}"

    if t1 not in log_prices.columns or t2 not in log_prices.columns:
        log.warning(f"  Skipping {label} — ticker missing")
        continue

    log.info(f"Processing  {t1.replace('.NS','')} / {t2.replace('.NS','')}")

    bt = backtest_pair(t1, t2, beta, log_prices)
    backtest_summary.append({
        "ticker_1": t1, "ticker_2": t2, "sector": sector,
        "eg_pvalue": pair["eg_pvalue"], "static_hl": static_hl,
        **bt["metrics"]
    })
    plot_backtest(label, t1, t2, bt["series"], bt["metrics"])
    log.info(f"  Sharpe: {bt['metrics']['sharpe']:.2f}  "
             f"Hit: {bt['metrics']['hit_rate']*100:.1f}%  "
             f"Trades: {bt['metrics']['n_trades']}")

    spread     = compute_spread(log_prices, t1, t2, beta)
    rolling_hl = compute_halflife_rolling(spread, HL_WINDOW)
    rolling_hl.to_csv(f"{RESULTS_DIR}/rolling_halflife/{label}.csv",
                      header=True)
    plot_rolling_halflife(label, t1, t2, rolling_hl, sector, static_hl)

    ew = event_window_analysis(rolling_hl, events_df, sector, EVENT_WINDOW)
    if not ew.empty:
        ew["ticker_1"] = t1
        ew["ticker_2"] = t2
        ew["sector"]   = sector
        all_event_windows.append(ew)

    log.info(f"  Event windows: {len(ew)} computed")

# =============================================================================
# SAVE & PRINT
# =============================================================================

summary_df = pd.DataFrame(backtest_summary)
summary_df.to_csv(f"{RESULTS_DIR}/backtest_summary.csv", index=False)

if all_event_windows:
    ew_df = pd.concat(all_event_windows, ignore_index=True)
    ew_df.to_csv(f"{RESULTS_DIR}/event_window_analysis.csv", index=False)
    plot_regime_comparison(ew_df)
else:
    ew_df = pd.DataFrame()
    log.warning("No event windows computed — check event dates vs data range")

print(f"\n{'='*74}")
print(f"  Phase 3 — Backtest & Regime Analysis Summary")
print(f"{'='*74}")
print(f"\n  Cost model: {TOTAL_COST_BPS} bps/leg  "
      f"({SLIPPAGE_BPS} slip + {IMPACT_BPS} impact + {BROKERAGE_BPS} brokerage)")
print(f"  Entry |Z|>{ENTRY_ZSCORE}  Exit |Z|<{EXIT_ZSCORE}  Stop |Z|>{STOP_ZSCORE}\n")

print(f"  {'Pair':<42} {'Sharpe':>7} {'Hit%':>6} {'Trades':>7} "
      f"{'Net Ret':>9} {'Cost Drag':>10} {'Max DD':>8}")
print(f"  {'-'*42} {'-'*7} {'-'*6} {'-'*7} {'-'*9} {'-'*10} {'-'*8}")

for _, r in summary_df.sort_values("sharpe", ascending=False).iterrows():
    pair = (f"{r['ticker_1'].replace('.NS','')} / "
            f"{r['ticker_2'].replace('.NS','')}")
    print(f"  {pair:<42} {r['sharpe']:>7.2f} "
          f"{r['hit_rate']*100:>5.1f}% {r['n_trades']:>7} "
          f"{r['total_ret']:>+9.4f} {r['cost_drag']:>+10.4f} "
          f"{r['max_dd']:>+8.4f}")

if not ew_df.empty:
    print(f"\n  Event Window Analysis ({EVENT_WINDOW}d pre/post):")
    print(f"  {'Event':<32} {'Pair':<28} "
          f"{'Pre HL':>7} {'Post HL':>8} {'Chg%':>7} {'Dir':>8}")
    print(f"  {'-'*32} {'-'*28} {'-'*7} {'-'*8} {'-'*7} {'-'*8}")
    for _, r in ew_df.iterrows():
        pair  = (f"{r['ticker_1'].replace('.NS','')}/"
                 f"{r['ticker_2'].replace('.NS','')}")
        print(f"  {r['event_label'][:32]:<32} {pair:<28} "
              f"{r['pre_hl_mean']:>7.1f} {r['post_hl_mean']:>8.1f} "
              f"{r['hl_pct_change']:>+7.1f}% {r['direction']:>8}")

    compress = (ew_df["direction"] == "compress").sum()
    expand   = (ew_df["direction"] == "expand").sum()
    total    = compress + expand
    print(f"\n  Compression: {compress}/{total}  |  Expansion: {expand}/{total}")

    if total > 0:
        pct = compress / total * 100
        if pct > 60:
            finding = (f"Policy events tend to COMPRESS mean reversion "
                       f"({pct:.0f}% of cases) — regulatory shocks "
                       f"temporarily strengthen the cointegrating relationship")
        elif pct < 40:
            finding = (f"Policy events tend to EXPAND mean reversion "
                       f"({100-pct:.0f}% of cases) — regulatory shocks "
                       f"temporarily disrupt the cointegrating relationship")
        else:
            finding = ("Mixed evidence — no dominant directional effect. "
                       "Sector and event type may matter more than the shock itself.")
        print(f"\n  Research finding:\n  {finding}")

print(f"\n  Plots: {PLOT_DIR}/")
print(f"  Data:  {RESULTS_DIR}/")
print(f"{'='*74}\n")
