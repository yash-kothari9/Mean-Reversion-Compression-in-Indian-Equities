"""
NSE Sector Data Loader — Pharma & Auto Ancillary
=================================================
Downloads 5 years of daily OHLCV data for NSE-listed stocks
in the pharma and auto ancillary sectors via yfinance.

Survivorship bias is handled by:
  1. Using a manually curated universe with known listing dates
  2. Dropping any stock that was NOT listed before DATA_START_DATE
  3. Dropping any stock with >10% missing data over the period
  4. Logging every exclusion so you can report it transparently

Output:
  data/raw_prices.parquet    — adjusted close prices (wide format)
  data/raw_ohlcv/            — full OHLCV per ticker (parquet)
  data/universe_report.csv   — inclusion/exclusion log

Requirements:
    pip install yfinance pandas pyarrow
"""

import os
import time
import warnings
import logging
from datetime import datetime, date

import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_START_DATE   = "2019-04-01"   # 5 years back from ~April 2024
DATA_END_DATE     = "2024-04-01"   # Adjust to today if running live
LISTING_CUTOFF    = date(2019, 4, 1)  # Stock must have listed before this date
MAX_MISSING_PCT   = 0.10           # Drop ticker if >10% of trading days are NaN
DOWNLOAD_DELAY    = 0.5            # Seconds between API calls (be a good citizen)
OUTPUT_DIR        = "data"

# =============================================================================
# STOCK UNIVERSE WITH LISTING DATES
# Format: "TICKER.NS": (listing_date, "company_name")
#
# Listing dates are approximate BSE/NSE IPO dates.
# This is the survivorship bias fix — we explicitly record when each stock
# entered the market so we never backfill prices that didn't exist yet.
# =============================================================================

PHARMA_UNIVERSE = {
    # Large-cap, liquid, pre-2015 listings — core universe
    "SUNPHARMA.NS":  (date(1994, 3,  1),  "Sun Pharmaceutical"),
    "CIPLA.NS":      (date(1995, 2,  1),  "Cipla"),
    "DRREDDY.NS":    (date(1986, 6,  1),  "Dr. Reddy's Laboratories"),
    "LUPIN.NS":      (date(2001, 6,  1),  "Lupin"),
    "AUROPHARMA.NS": (date(2000, 7,  1),  "Aurobindo Pharma"),
    "DIVISLAB.NS":   (date(2003, 3,  1),  "Divi's Laboratories"),
    "TORNTPHARM.NS": (date(2002, 11, 1),  "Torrent Pharmaceuticals"),
    "ALKEM.NS":      (date(2015, 12, 23), "Alkem Laboratories"),
    "IPCALAB.NS":    (date(1995, 3,  1),  "Ipca Laboratories"),
    "BIOCON.NS":     (date(2004, 4,  7),  "Biocon"),
    "ABBOTINDIA.NS": (date(2010, 1,  1),  "Abbott India"),
    "GLAXO.NS":      (date(2001, 1,  1),  "GSK Pharma India"),
    "PFIZER.NS":     (date(1996, 1,  1),  "Pfizer India"),
    "ZYDUSLIFE.NS":  (date(2010, 3,  18), "Zydus Lifesciences"),
    "SANOFI.NS":     (date(2009, 1,  1),  "Sanofi India"),

    # Mid-cap, listed before cutoff
    "LAURUSLABS.NS": (date(2016, 12, 19), "Laurus Labs"),   # Dec 2016 IPO
    "GRANULES.NS":   (date(2004, 9,  1),  "Granules India"),
    "NATCOPHARM.NS": (date(1995, 1,  1),  "Natco Pharma"),
    "GLENMARK.NS":   (date(1999, 12, 1),  "Glenmark Pharmaceuticals"),

    # Intentional exclusions (listed AFTER cutoff — survivorship bias test cases)
    # "GLAND.NS":    (date(2020, 11, 20), "Gland Pharma"),   # IPO Nov 2020 -> EXCLUDED
    # "NUVAMA.NS":   (date(2022, 1,  1),  "Nuvama"),         # Too recent -> EXCLUDED
}

AUTO_ANCILLARY_UNIVERSE = {
    # Core liquid names
    "BOSCHLTD.NS":     (date(1996, 1,  1),  "Bosch India"),
    "MOTHERSON.NS":    (date(1993, 1,  1),  "Samvardhana Motherson"),
    "BHARATFORG.NS":   (date(1998, 1,  1),  "Bharat Forge"),
    "BALKRISIND.NS":   (date(2002, 1,  1),  "Balkrishna Industries"),
    "SCHAEFFLER.NS":   (date(2013, 10, 1),  "Schaeffler India"),
    "MINDAIND.NS":     (date(1997, 1,  1),  "Minda Industries"),
    "SUNDRMFAST.NS":   (date(1995, 1,  1),  "Sundram Fasteners"),
    "ENDURANCE.NS":    (date(2016, 10, 18), "Endurance Technologies"),
    "TIINDIA.NS":      (date(2016, 9,  14), "Tube Investments of India"),
    "EXIDEIND.NS":     (date(1997, 1,  1),  "Exide Industries"),
    "AMARARAJA.NS":    (date(1992, 1,  1),  "Amara Raja Batteries"),
    "MINDACORP.NS":    (date(2004, 1,  1),  "Minda Corporation"),
    "SUPRAJIT.NS":     (date(1999, 1,  1),  "Suprajit Engineering"),
    "GABRIEL.NS":      (date(1999, 1,  1),  "Gabriel India"),
    "LUMAX.NS":        (date(2011, 1,  1),  "Lumax Industries"),

    # Intentional exclusions (listed after cutoff)
    # "SONA.NS":       (date(2021, 6, 14), "Sona BLW Precision"), # IPO June 2021 -> EXCLUDED
    # "CRAFTSMAN.NS":  (date(2021, 3, 25), "Craftsman Auto"),      # IPO Mar 2021 -> EXCLUDED
}

# =============================================================================
# SETUP
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/raw_ohlcv", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =============================================================================
# STEP 1: Apply survivorship bias filter BEFORE downloading
# =============================================================================

def apply_survivorship_filter(universe: dict, cutoff: date, label: str) -> dict:
    """
    Remove any stock whose listing date is on or after LISTING_CUTOFF.
    These stocks did not exist at the start of our backtest period —
    including them would introduce survivorship bias because we only know
    about them because they survived to get listed.
    """
    included = {}
    excluded = []

    for ticker, (listing_date, name) in universe.items():
        if listing_date < cutoff:
            included[ticker] = (listing_date, name)
        else:
            excluded.append({
                "ticker": ticker,
                "name": name,
                "listing_date": listing_date,
                "reason": f"Listed {listing_date} >= cutoff {cutoff}"
            })

    log.info(f"[{label}] Survivorship filter: {len(included)} included, "
             f"{len(excluded)} excluded")
    for ex in excluded:
        log.info(f"  EXCLUDED  {ex['ticker']:20s}  {ex['reason']}")

    return included, excluded

pharma_filtered, pharma_excluded = apply_survivorship_filter(
    PHARMA_UNIVERSE, LISTING_CUTOFF, "PHARMA"
)
auto_filtered, auto_excluded = apply_survivorship_filter(
    AUTO_ANCILLARY_UNIVERSE, LISTING_CUTOFF, "AUTO"
)

all_tickers = {**pharma_filtered, **auto_filtered}
log.info(f"\nTotal tickers to download: {len(all_tickers)}\n")

# =============================================================================
# STEP 2: Download OHLCV data with error handling
# =============================================================================

def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download adjusted OHLCV for a single ticker.
    Returns None on failure rather than raising — lets us continue the loop.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,   # Adjusts for splits and dividends
            progress=False
        )
        if df.empty:
            return None

        # yfinance sometimes returns MultiIndex columns — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    except Exception as e:
        log.warning(f"  Download error for {ticker}: {e}")
        return None


raw_data   = {}   # ticker -> full OHLCV DataFrame
close_data = {}   # ticker -> Close Series (for building the wide matrix)
report     = []   # inclusion/exclusion log

for ticker, (listing_date, name) in all_tickers.items():
    sector = "pharma" if ticker in pharma_filtered else "auto_ancillary"
    log.info(f"Downloading  {ticker:22s}  ({name})")

    df = download_ticker(ticker, DATA_START_DATE, DATA_END_DATE)
    time.sleep(DOWNLOAD_DELAY)

    # --- Quality checks ---
    if df is None or len(df) < 100:
        reason = "Empty or <100 rows returned"
        log.warning(f"  SKIP  {ticker}  — {reason}")
        report.append({
            "ticker": ticker, "name": name, "sector": sector,
            "listing_date": listing_date, "status": "excluded",
            "rows": 0, "missing_pct": None, "reason": reason
        })
        continue

    missing_pct = df["Close"].isna().mean()
    if missing_pct > MAX_MISSING_PCT:
        reason = f"Missing data: {missing_pct*100:.1f}% > {MAX_MISSING_PCT*100:.0f}% threshold"
        log.warning(f"  SKIP  {ticker}  — {reason}")
        report.append({
            "ticker": ticker, "name": name, "sector": sector,
            "listing_date": listing_date, "status": "excluded",
            "rows": len(df), "missing_pct": round(missing_pct, 4),
            "reason": reason
        })
        continue

    # --- Zero/negative price check ---
    if (df["Close"] <= 0).any():
        df = df[df["Close"] > 0]
        log.info(f"  Removed {(df['Close'] <= 0).sum()} zero/negative price rows for {ticker}")

    # --- Forward-fill short gaps (≤2 days, e.g. NSE holidays not in yfinance calendar) ---
    df["Close"] = df["Close"].ffill(limit=2)

    # --- Save individual OHLCV ---
    df.to_parquet(f"{OUTPUT_DIR}/raw_ohlcv/{ticker.replace('.', '_')}.parquet")

    raw_data[ticker]   = df
    close_data[ticker] = df["Close"].rename(ticker)

    report.append({
        "ticker": ticker, "name": name, "sector": sector,
        "listing_date": listing_date, "status": "included",
        "rows": len(df), "missing_pct": round(missing_pct, 4),
        "reason": "passed all checks"
    })
    log.info(f"  OK    {len(df)} rows, {missing_pct*100:.2f}% missing")

# Add pre-filtered survivorship exclusions to the report
for ex in pharma_excluded + auto_excluded:
    sector = "pharma" if ex["ticker"] in PHARMA_UNIVERSE else "auto_ancillary"
    report.append({
        "ticker": ex["ticker"], "name": ex["name"], "sector": sector,
        "listing_date": ex["listing_date"], "status": "excluded_survivorship",
        "rows": None, "missing_pct": None, "reason": ex["reason"]
    })

# =============================================================================
# STEP 3: Build the wide adjusted-close price matrix
# =============================================================================

if not close_data:
    raise RuntimeError("No data downloaded. Check your internet connection or tickers.")

price_matrix = pd.concat(close_data.values(), axis=1)
price_matrix.index = pd.to_datetime(price_matrix.index)
price_matrix = price_matrix.sort_index()

# Align all series to the same trading calendar (NSE)
# Some tickers have different holiday calendars — take the union of dates
# and forward-fill short gaps created by the alignment
price_matrix = price_matrix.resample("B").last()
price_matrix = price_matrix.ffill(limit=3)

log.info(f"\nPrice matrix shape: {price_matrix.shape}  "
         f"({price_matrix.shape[1]} tickers x {price_matrix.shape[0]} days)")

# =============================================================================
# STEP 4: Save outputs
# =============================================================================

# Wide price matrix
price_matrix.to_parquet(f"{OUTPUT_DIR}/raw_prices.parquet")
log.info(f"Saved:  {OUTPUT_DIR}/raw_prices.parquet")

# Universe report
report_df = pd.DataFrame(report).sort_values(["sector", "status", "ticker"])
report_df.to_csv(f"{OUTPUT_DIR}/universe_report.csv", index=False)
log.info(f"Saved:  {OUTPUT_DIR}/universe_report.csv")

# =============================================================================
# STEP 5: Summary printout
# =============================================================================

included = report_df[report_df["status"] == "included"]
excluded = report_df[report_df["status"] != "included"]

pharma_in   = included[included["sector"] == "pharma"]
auto_in     = included[included["sector"] == "auto_ancillary"]

print(f"\n{'='*60}")
print(f"  Data Download Summary")
print(f"{'='*60}")
print(f"\n  Pharma universe:        {len(pharma_in)} stocks included")
for _, row in pharma_in.iterrows():
    print(f"    {row['ticker']:22s}  {row['rows']:5d} rows  "
          f"{row['missing_pct']*100:.1f}% missing")

print(f"\n  Auto ancillary universe: {len(auto_in)} stocks included")
for _, row in auto_in.iterrows():
    print(f"    {row['ticker']:22s}  {row['rows']:5d} rows  "
          f"{row['missing_pct']*100:.1f}% missing")

print(f"\n  Excluded (all reasons):  {len(excluded)} stocks")
for _, row in excluded.iterrows():
    print(f"    {row['ticker']:22s}  {row['reason']}")

print(f"\n  Price matrix:  {price_matrix.shape[0]} dates  x  "
      f"{price_matrix.shape[1]} tickers")
print(f"  Date range:    {price_matrix.index[0].date()} to "
      f"{price_matrix.index[-1].date()}")
print(f"\n  All outputs saved to ./{OUTPUT_DIR}/")
print(f"{'='*60}\n")


# =============================================================================
# STEP 6: Load helper — use this in Phase 2 (cointegration tests)
# =============================================================================

def load_prices(sector: str = "all") -> pd.DataFrame:
    """
    Convenience loader for downstream notebooks.

    Usage:
        from nse_data_loader import load_prices
        prices = load_prices("pharma")          # pharma only
        prices = load_prices("auto_ancillary")  # auto only
        prices = load_prices("all")             # both sectors

    Returns a DataFrame of adjusted close prices, dates as index.
    """
    df = pd.read_parquet(f"{OUTPUT_DIR}/raw_prices.parquet")
    report = pd.read_csv(f"{OUTPUT_DIR}/universe_report.csv")

    if sector == "all":
        tickers = report[report["status"] == "included"]["ticker"].tolist()
    else:
        tickers = report[
            (report["status"] == "included") & (report["sector"] == sector)
        ]["ticker"].tolist()

    available = [t for t in tickers if t in df.columns]
    return df[available].dropna(how="all")


def get_sector_map() -> dict:
    """
    Returns {ticker: sector} for all included tickers.
    Useful for labelling pairs in Phase 2.
    """
    report = pd.read_csv(f"{OUTPUT_DIR}/universe_report.csv")
    included = report[report["status"] == "included"]
    return dict(zip(included["ticker"], included["sector"]))


if __name__ == "__main__":
    # Quick sanity check
    prices = load_prices("pharma")
    log.info(f"Pharma prices loaded: {prices.shape}")
    print(prices.tail(3))
