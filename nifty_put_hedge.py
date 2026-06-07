"""
Nifty Put Hedge Module

Fetches historical NIFTY ATM Put option prices using Dhan's Rolling Options API
(endpoint: /charts/rollingoption).  Supports last 5 years of data; dates before
that are ignored (no options data available).

Design decisions (per user spec):
  - ATM only, no strike offset
  - Weekly expiry only (NIFTY weekly puts)
  - Delta-neutral lot sizing  (portfolio_value / nifty_spot / (atm_delta * lot_size))
  - Option ticker naming includes strike + expiry: NIFTY25000PE03JUL2025
  - No local data cache needed – fetch is done chunk-by-chunk and results are
    cached in memory during a backtest run (DataFrame stored in portfolio_engine)
  - Dhan API has 5 years of rolling options data available
"""

import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────
NIFTY_STRIKE_STEP   = 50          # NIFTY strikes are multiples of 50
DATA_AVAILABLE_FROM = date(2020, 1, 1)   # Dhan rolling options data goes back ~5 years
ATM_PUT_DELTA       = 0.5         # Magnitude of delta for ATM put (theoretical: N(-d1) ≈ 0.5)
DHAN_ROLLING_OPTIONS_URL = "https://api.dhan.co/v2/charts/rollingoption"

# NIFTY lot size: 75 from Jan 2025, 50 before
LOT_SIZE_CHANGES = [(date(2025, 1, 1), 75)]
DEFAULT_LOT_SIZE = 50


# ─── Lot Size ─────────────────────────────────────────────────────────────────

def get_nifty_lot_size(as_of_date) -> int:
    """Return NIFTY F&O lot size for a given date."""
    if isinstance(as_of_date, pd.Timestamp):
        as_of_date = as_of_date.date()
    for change_date, lot_size in sorted(LOT_SIZE_CHANGES, reverse=True):
        if as_of_date >= change_date:
            return lot_size
    return DEFAULT_LOT_SIZE


# ─── Strike & Expiry Utilities ────────────────────────────────────────────────

def get_atm_strike(nifty_spot: float) -> int:
    """Round NIFTY spot to nearest 50 to get ATM strike."""
    return int(round(nifty_spot / NIFTY_STRIKE_STEP) * NIFTY_STRIKE_STEP)


def get_next_expiry(from_dt, expiry_type="WEEKLY") -> date:
    """
    Get next Thursday (NIFTY weekly expiry) or last Thursday of month (MONTHLY expiry) after from_dt.
    """
    if isinstance(from_dt, (datetime, pd.Timestamp)):
        from_dt = from_dt.date() if hasattr(from_dt, 'date') else date.fromisoformat(str(from_dt)[:10])

    if expiry_type == "MONTHLY":
        import calendar
        def get_last_thu(y, m):
            last_day = calendar.monthrange(y, m)[1]
            last_dt = date(y, m, last_day)
            offset = (last_dt.weekday() - 3) % 7
            return last_dt - timedelta(days=offset)
        
        last_thu = get_last_thu(from_dt.year, from_dt.month)
        if last_thu <= from_dt:
            next_m = from_dt.month + 1 if from_dt.month < 12 else 1
            next_y = from_dt.year if from_dt.month < 12 else from_dt.year + 1
            last_thu = get_last_thu(next_y, next_m)
        return last_thu
    else:
        weekday = from_dt.weekday()      # Mon=0, Thu=3, Sun=6
        days_until_thu = (3 - weekday) % 7
        if days_until_thu == 0:
            days_until_thu = 7           # Already Thursday – roll to next week
        return from_dt + timedelta(days=days_until_thu)



def get_option_ticker_name(strike: int, expiry_date) -> str:
    """
    Return canonical option ticker name used in tradebook.
    Format: NIFTY{strike}PE{DD}{MON}{YYYY}  e.g. NIFTY25000PE03JUL2025
    """
    if isinstance(expiry_date, (datetime, pd.Timestamp)):
        expiry_date = expiry_date.date()
    return f"NIFTY{strike}PE{expiry_date.strftime('%d%b%Y').upper()}"


# ─── Delta-Neutral Lot Calculation ───────────────────────────────────────────

def delta_neutral_lots(
    portfolio_value: float,
    nifty_spot: float,
    as_of_date=None,
    hedge_ratio: float = 1.0,
    beta: float = 1.0,
    atm_delta: float = ATM_PUT_DELTA,
) -> int:
    """
    Calculate number of NIFTY Put lots for a delta-neutral hedge.

    Logic:
        portfolio_delta (NIFTY units) = portfolio_value × beta / nifty_spot
        hedge_delta_needed            = portfolio_delta × hedge_ratio
        lots_needed                   = hedge_delta_needed / (atm_delta × lot_size)

    Args:
        portfolio_value:  Total portfolio value in ₹
        nifty_spot:       Current NIFTY index level
        as_of_date:       Date for determining lot size
        hedge_ratio:      1.0 = full delta neutral, 0.5 = half hedge
        beta:             Portfolio beta vs NIFTY (default 1.0)
        atm_delta:        Delta magnitude of ATM put (default 0.5)

    Returns:
        Integer number of lots (≥ 1 if portfolio_value > 0, else 0)
    """
    if nifty_spot <= 0 or portfolio_value <= 0:
        return 0

    lot_size = get_nifty_lot_size(as_of_date or date.today())

    portfolio_nifty_units = (portfolio_value * beta) / nifty_spot
    hedge_nifty_units     = portfolio_nifty_units * hedge_ratio
    lots                  = hedge_nifty_units / (atm_delta * lot_size)

    return max(1, math.floor(lots))


# ─── NIFTY Spot Lookup ────────────────────────────────────────────────────────

def get_nifty_spot_on_date(dt: pd.Timestamp, put_df: pd.DataFrame = None) -> float:
    """
    Get NIFTY spot level on a given date.

    Priority:
      1. 'Spot' column in put_df (from Dhan API response)
      2. YFinance ^NSEI download
    """
    # From put_df Spot column (most accurate – comes from the same API response)
    if put_df is not None and not put_df.empty and 'Spot' in put_df.columns:
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        if dt in put_df.index:
            val = put_df.loc[dt, 'Spot']
            if pd.notna(val) and float(val) > 0:
                return float(val)
        # asof fallback
        try:
            nearest = put_df.index.asof(dt)
            if not pd.isna(nearest) and abs((dt - nearest).days) <= 5:
                val = put_df.loc[nearest, 'Spot']
                if pd.notna(val) and float(val) > 0:
                    return float(val)
        except Exception:
            pass

    return 0.0


# ─── Put Premium Lookup ───────────────────────────────────────────────────────

def get_put_premium_on_date(dt: pd.Timestamp, put_df: pd.DataFrame) -> float:
    """
    Look up ATM Put close premium for a given date.
    Falls back to nearest available date within ±5 days.
    Returns 0.0 if unavailable.
    """
    if put_df is None or put_df.empty:
        return 0.0
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)

    if dt in put_df.index:
        val = put_df.loc[dt, 'Close']
        return float(val) if pd.notna(val) else 0.0

    try:
        nearest = put_df.index.asof(dt)
        if not pd.isna(nearest) and abs((dt - nearest).days) <= 5:
            val = put_df.loc[nearest, 'Close']
            return float(val) if pd.notna(val) else 0.0
    except Exception:
        pass

    return 0.0



# ─── Dhan Rolling Options API ─────────────────────────────────────────────────

def _parse_rolling_response(response_data: dict) -> pd.DataFrame:
    """Parse Dhan rolling options API response into daily OHLCIV DataFrame.
    Response structure: {"data": {"pe": {"open":[], "close":[], ... "timestamp":[]}}}
    """
    data = response_data.get("data", {})
    if not data:
        print("[PUT HEDGE] API response has no 'data' key")
        return pd.DataFrame()

    # The response nests under 'pe' for PUT options
    pe_data = data.get("pe", {})
    if not pe_data:
        print("[PUT HEDGE] API response has no 'pe' key — full response keys:", list(data.keys()))
        return pd.DataFrame()

    timestamps = pe_data.get("timestamp", [])
    if not timestamps:
        print("[PUT HEDGE] No timestamps in API response")
        return pd.DataFrame()

    records = []
    for i, ts in enumerate(timestamps):
        try:
            dt = datetime.fromtimestamp(int(ts))
        except Exception:
            continue

        def _get(key, default=0):
            arr = pe_data.get(key, [])
            return arr[i] if i < len(arr) else default

        records.append({
            "Date":   dt,
            "Open":   float(_get("open")),
            "High":   float(_get("high")),
            "Low":    float(_get("low")),
            "Close":  float(_get("close")),
            "Volume": int(_get("volume")),
            "OI":     int(_get("oi")),
            "IV":     float(_get("iv")),
            "Spot":   float(_get("spot")),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("Date")
    df.index = pd.to_datetime(df.index)

    # Resample minute data -> daily OHLCV
    daily = df.resample("D").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
        "OI":     "last",
        "IV":     "last",
        "Spot":   "last",
    }).dropna(subset=["Close"])

    daily = daily[daily["Close"] > 0]

    # Add ATM strike column from Spot
    daily["Strike"] = daily["Spot"].apply(lambda s: get_atm_strike(s) if s > 0 else 0)
    return daily


def _fetch_chunk(dhan_client, from_dt: date, to_dt: date, expiry_type="WEEKLY") -> pd.DataFrame:
    """Fetch one <=30-day chunk of NIFTY ATM put data."""
    # Map our expiry_type to Dhan's expiryFlag (WEEK/MONTH) and expiryCode
    expiry_flag = "WEEK" if expiry_type == "WEEKLY" else "MONTH"
    expiry_code = 1  # nearest expiry

    # Try SDK method first
    if dhan_client is not None:
        try:
            resp = dhan_client.rolling_options_data(
                exchangeSegment="NSE_FNO",
                interval="1",
                securityId="13",
                instrument="OPTIDX",
                expiryFlag=expiry_flag,
                expiryCode=expiry_code,
                strike="ATM",
                drvOptionType="PUT",
                requiredData=["open", "high", "low", "close", "iv", "volume", "oi", "spot"],
                fromDate=from_dt.strftime("%Y-%m-%d"),
                toDate=to_dt.strftime("%Y-%m-%d"),
            )
            if isinstance(resp, dict) and resp.get("status") == "success":
                return _parse_rolling_response(resp)
            elif isinstance(resp, dict) and "data" in resp:
                return _parse_rolling_response(resp)
        except Exception as e:
            print(f"[PUT HEDGE] SDK call failed ({e}), trying direct REST...")

    # Direct REST fallback
    return _fetch_chunk_rest(from_dt, to_dt, expiry_type=expiry_type)


def _fetch_chunk_rest(from_dt: date, to_dt: date, expiry_type="WEEKLY") -> pd.DataFrame:
    """Direct REST call to Dhan Rolling Options endpoint with correct payload."""
    try:
        from config import get_saved_credentials
        creds = get_saved_credentials()
        cid   = creds.get("client_id", "")
        token = creds.get("access_token", "")
    except Exception:
        return pd.DataFrame()

    if not cid or not token:
        print("[PUT HEDGE] No Dhan credentials found for REST call")
        return pd.DataFrame()

    # Map to Dhan API values
    expiry_flag = "WEEK" if expiry_type == "WEEKLY" else "MONTH"

    headers = {
        "access-token": token,
        "client-id":    cid,
        "Content-Type": "application/json",
        "Accept":       "application/json",
    }
    payload = {
        "exchangeSegment": "NSE_FNO",
        "interval":        "1",
        "securityId":      "13",      # NIFTY security ID
        "instrument":      "OPTIDX",
        "expiryFlag":      expiry_flag,
        "expiryCode":      1,         # nearest expiry
        "strike":          "ATM",
        "drvOptionType":   "PUT",
        "requiredData":    ["open", "high", "low", "close", "iv", "volume", "oi", "spot"],
        "fromDate":        from_dt.strftime("%Y-%m-%d"),
        "toDate":          to_dt.strftime("%Y-%m-%d"),
    }

    try:
        print(f"[PUT HEDGE] REST call: {from_dt} -> {to_dt} | expiryFlag={expiry_flag}")
        resp = requests.post(DHAN_ROLLING_OPTIONS_URL, json=payload, headers=headers, timeout=30)
        print(f"[PUT HEDGE] REST status: {resp.status_code}")
        if resp.status_code == 200:
            return _parse_rolling_response(resp.json())
        else:
            print(f"[PUT HEDGE] REST error {resp.status_code}: {resp.text[:400]}")
    except Exception as e:
        print(f"[PUT HEDGE] REST request failed: {e}")

    return pd.DataFrame()


def fetch_rolling_options_data(
    from_date,
    to_date,
    expiry_type: str = "WEEKLY",
    delay_seconds: float = 0.6,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch NIFTY ATM Weekly Put data from Dhan Rolling Options API.

    - Chunks the date range into 30-day windows
    - Only fetches for dates >= DATA_AVAILABLE_FROM (Jan 2020)
    - Caches nothing – caller is responsible for persistence

    Args:
        from_date:         Start date
        to_date:           End date
        delay_seconds:     Pause between API calls (rate limiting)
        progress_callback: Optional callable(current, total, label)

    Returns:
        Daily DataFrame (DatetimeIndex) with Open/High/Low/Close/Volume/OI/IV/Spot/Strike
    """
    if isinstance(from_date, (datetime, pd.Timestamp)):
        from_date = from_date.date()
    if isinstance(to_date, (datetime, pd.Timestamp)):
        to_date = to_date.date()

    # Clamp to available data range
    from_date = max(from_date, DATA_AVAILABLE_FROM)
    if from_date > to_date:
        print(f"[PUT HEDGE] No data before {DATA_AVAILABLE_FROM} – skipping.")
        return pd.DataFrame()

    # Get Dhan client
    dhan_client = None
    try:
        from config import get_dhan_client, validate_credentials
        validate_credentials()
        dhan_client = get_dhan_client()
    except Exception as e:
        print(f"[PUT HEDGE] Dhan client unavailable ({e}). Will try direct REST.")

    # Build 30-day chunks
    chunks = []
    cur = from_date
    while cur <= to_date:
        end = min(cur + timedelta(days=29), to_date)
        chunks.append((cur, end))
        cur = end + timedelta(days=1)

    print(f"[PUT HEDGE] Fetching {len(chunks)} chunks ({from_date} → {to_date})...")

    all_frames = []
    for i, (cf, ct) in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, len(chunks), f"Chunk {i+1}/{len(chunks)}: {cf} → {ct}")

        chunk_df = _fetch_chunk(dhan_client, cf, ct, expiry_type=expiry_type)
        if chunk_df is not None and not chunk_df.empty:
            all_frames.append(chunk_df)
        else:
            print(f"[PUT HEDGE]   No data for chunk {cf} → {ct}")

        if i < len(chunks) - 1:
            time.sleep(delay_seconds)

    if not all_frames:
        print("[PUT HEDGE] No data from Dhan API.")
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"[PUT HEDGE] Total {len(combined)} trading days fetched.")
    return combined


# ─── Master Loader ────────────────────────────────────────────────────────────

def load_or_build_hedge_data(from_date, to_date, expiry_type="WEEKLY", **kwargs):
    """
    Fetch hedge data exclusively from Dhan API.
    Raises RuntimeError if no data is returned — the VIX/Black-Scholes fallback
    has been removed to prevent biased backtesting.
    """
    from datetime import date as _date
    if isinstance(from_date, (datetime, pd.Timestamp)):
        from_date = from_date.date()
    if isinstance(to_date, (datetime, pd.Timestamp)):
        to_date = to_date.date()

    eff_from = max(from_date, DATA_AVAILABLE_FROM)
    if eff_from > to_date:
        raise RuntimeError(
            "Backtest date range is entirely before " + str(DATA_AVAILABLE_FROM) + ". "
            "Dhan options data is only available from Jan 2020 onwards."
        )

    api_df = fetch_rolling_options_data(eff_from, to_date, expiry_type=expiry_type)
    if not api_df.empty:
        return api_df

    raise RuntimeError(
        "Dhan API returned no options data. "
        "Please authenticate with Dhan before running a Put Hedge backtest. "
        "Go to Settings > Dhan Auth and log in first."
    )

