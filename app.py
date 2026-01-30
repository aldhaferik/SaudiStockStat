# app.py
from __future__ import annotations

import os
import json
import math
import time
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# 0) APP + CORS
# =========================================================
app = FastAPI(title="Saudi Valuator Pro", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 1) CONFIG
# =========================================================
DEFAULT_HISTORY_PERIOD = "10y"     # used for Yahoo fallback requests
DEFAULT_INTERVAL = "1d"
TRADING_DAYS = 252

# Your keys (also read env vars if set)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "0LR5JLOBSLOA6Z0A")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "ed240f406bab4225ac6e0a98be553aa2")

# Caching
CACHE_DB_PATH = os.getenv("SVP_CACHE_DB", "svp_cache.sqlite3")
CACHE_TTL_SECONDS = int(os.getenv("SVP_CACHE_TTL_SECONDS", str(6 * 3600)))  # 6 hours default

# Backtest / prediction defaults
PREDICTION_HORIZON_DAYS_DEFAULT = 63  # ~3 months
MIN_TRAIN_DAYS = 600                 # require enough history for walk-forward
WALK_FORWARD_TEST_STEP = 21          # monthly-ish
FEATURE_LOOKBACKS = [5, 21, 63, 126] # days

# DCF defaults (used ONLY when inputs are missing; always disclosed)
DCF_FORECAST_YEARS = 5
TERMINAL_GROWTH_DEFAULT = 0.03       # disclosed; can be overridden
TAX_RATE_DEFAULT = 0.20              # disclosed; can be overridden
WACC_DEFAULT = 0.10                  # disclosed; can be overridden
SHARES_OUTSTANDING_FALLBACK = None   # if missing, DCF per-share may be impossible


# =========================================================
# 2) SQLITE CACHE (RAW + PARSED SNAPSHOTS)
# =========================================================
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(CACHE_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            created_utc INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    return con

def cache_get(cache_key: str, ttl_seconds: int = CACHE_TTL_SECONDS) -> Optional[Dict[str, Any]]:
    con = _db()
    try:
        row = con.execute("SELECT created_utc, payload_json FROM cache WHERE cache_key = ?", (cache_key,)).fetchone()
        if not row:
            return None
        created_utc, payload_json = row
        if int(time.time()) - int(created_utc) > ttl_seconds:
            return None
        return json.loads(payload_json)
    finally:
        con.close()

def cache_set(cache_key: str, payload: Dict[str, Any]) -> None:
    con = _db()
    try:
        con.execute(
            "INSERT OR REPLACE INTO cache(cache_key, created_utc, payload_json) VALUES(?,?,?)",
            (cache_key, int(time.time()), json.dumps(payload, default=str)),
        )
        con.commit()
    finally:
        con.close()


# =========================================================
# 3) UTILITIES
# =========================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            if np.isnan(x):
                return None
            return float(x)
        s = str(x).strip()
        if s.lower() in ("nan", "none", "", "null"):
            return None
        return float(s)
    except Exception:
        return None

def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if np.isnan(v) else v
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="records")
    return x

def df_clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns: date, open, high, low, close, adj_close, volume
    Ensure date is UTC-normalized, sorted, no duplicates.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
    d = df.copy()
    if "Date" in d.columns:
        d = d.rename(columns={"Date": "date"})
    if "date" not in d.columns:
        d = d.reset_index().rename(columns={"index": "date"})
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    d = d.drop_duplicates(subset=["date"], keep="last")

    rename_map = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "AdjClose": "adj_close", "Volume": "volume",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "adj_close": "adj_close", "volume": "volume",
    }
    d = d.rename(columns=rename_map)
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c not in d.columns:
            d[c] = np.nan

    d = d[["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()

    # If adj_close missing but close exists, use close
    if d["adj_close"].isna().all() and not d["close"].isna().all():
        d["adj_close"] = d["close"]

    return d.reset_index(drop=True)

def log_return(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return np.log(s / s.shift(1))

def robust_zscore(x: pd.Series) -> pd.Series:
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if mad is None or mad == 0 or np.isnan(mad):
        return (x - x.mean(skipna=True)) / (x.std(skipna=True) + 1e-12)
    return (x - med) / (1.4826 * mad + 1e-12)


# =========================================================
# 4) DATA LOADERS (YAHOO primary, TwelveData/AlphaVantage fallback)
# =========================================================
def fetch_prices_yahoo(ticker: str, period: str = DEFAULT_HISTORY_PERIOD, interval: str = DEFAULT_INTERVAL) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Uses yfinance (installed via requirements). If not available, raises ImportError.
    """
    import yfinance as yf  # local import

    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=False, actions=True)
    meta = {"source": "yahoo", "period": period, "interval": interval}
    if hist is None or hist.empty:
        return pd.DataFrame(), meta
    hist = hist.reset_index()
    return df_clean_prices(hist), meta

def fetch_prices_twelvedata(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Twelve Data time_series endpoint.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
        "outputsize": 5000,
    }
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end

    r = requests.get(url, params=params, timeout=30)
    meta = {"source": "twelvedata", "http_status": r.status_code}
    if r.status_code != 200:
        return pd.DataFrame(), {**meta, "error": r.text[:500]}
    j = r.json()
    if "values" not in j or not isinstance(j["values"], list):
        return pd.DataFrame(), {**meta, "error": j}
    rows = []
    for v in j["values"]:
        rows.append(
            {
                "date": v.get("datetime"),
                "open": safe_float(v.get("open")),
                "high": safe_float(v.get("high")),
                "low": safe_float(v.get("low")),
                "close": safe_float(v.get("close")),
                "adj_close": safe_float(v.get("close")),
                "volume": safe_float(v.get("volume")),
            }
        )
    df = pd.DataFrame(rows)
    return df_clean_prices(df), meta

def fetch_prices_alphavantage(ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_KEY,
        "outputsize": "full",
    }
    r = requests.get(url, params=params, timeout=30)
    meta = {"source": "alphavantage", "http_status": r.status_code}
    if r.status_code != 200:
        return pd.DataFrame(), {**meta, "error": r.text[:500]}
    j = r.json()
    ts = j.get("Time Series (Daily)")
    if not isinstance(ts, dict):
        return pd.DataFrame(), {**meta, "error": j}
    rows = []
    for dt_str, v in ts.items():
        rows.append(
            {
                "date": dt_str,
                "open": safe_float(v.get("1. open")),
                "high": safe_float(v.get("2. high")),
                "low": safe_float(v.get("3. low")),
                "close": safe_float(v.get("4. close")),
                "adj_close": safe_float(v.get("5. adjusted close")),
                "volume": safe_float(v.get("6. volume")),
            }
        )
    df = pd.DataFrame(rows)
    return df_clean_prices(df), meta

def get_prices_with_fallback(ticker: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "prices": [ ... ],
        "meta": {...},
        "quality": {...},
        "warnings": [...]
      }
    """
    cache_key = f"prices::{ticker}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    warnings: List[str] = []
    sources_tried: List[Dict[str, Any]] = []

    # 1) Yahoo
    df = pd.DataFrame()
    meta = {}
    try:
        df, meta = fetch_prices_yahoo(ticker)
        sources_tried.append(meta)
    except Exception as e:
        warnings.append(f"Yahoo fetch failed: {type(e).__name__}: {str(e)[:200]}")
        sources_tried.append({"source": "yahoo", "error": f"{type(e).__name__}: {str(e)[:200]}"})

    # If Yahoo empty -> Twelve Data
    if df.empty:
        df2, meta2 = fetch_prices_twelvedata(ticker)
        sources_tried.append(meta2)
        if not df2.empty:
            df = df2
            meta = meta2
        else:
            warnings.append("Twelve Data returned empty or error.")

    # If still empty -> Alpha Vantage
    if df.empty:
        df3, meta3 = fetch_prices_alphavantage(ticker)
        sources_tried.append(meta3)
        if not df3.empty:
            df = df3
            meta = meta3
        else:
            warnings.append("Alpha Vantage returned empty or error.")

    # Quality checks
    quality = {
        "n_rows": int(len(df)),
        "start": (df["date"].min().isoformat() if not df.empty else None),
        "end": (df["date"].max().isoformat() if not df.empty else None),
        "missing_adj_close_pct": (float(df["adj_close"].isna().mean()) if not df.empty else 1.0),
        "missing_volume_pct": (float(df["volume"].isna().mean()) if not df.empty else 1.0),
    }

    payload = {
        "asof_utc": utc_now_iso(),
        "ticker": ticker,
        "meta": {"selected_source": meta, "sources_tried": sources_tried},
        "quality": quality,
        "warnings": warnings,
        "prices": [ {k: to_jsonable(v) for k, v in row.items()} for row in df.to_dict(orient="records") ],
    }
    cache_set(cache_key, payload)
    return payload

def fetch_fundamentals_yahoo(ticker: str) -> Dict[str, Any]:
    """
    Uses yfinance fundamentals. For many Tadawul tickers, data may be partial.
    We do NOT invent. Missing stays missing.
    """
    import yfinance as yf  # local import

    t = yf.Ticker(ticker)

    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    def df_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []
        d = df.copy()
        # yfinance returns columns as dates
        d.columns = [str(c) for c in d.columns]
        d.index = [str(i) for i in d.index]
        d = d.reset_index().rename(columns={"index": "line_item"})
        return json.loads(d.to_json(orient="records"))

    fin = {}
    try:
        fin = {
            "financials": df_to_records(t.financials),
            "balance_sheet": df_to_records(t.balance_sheet),
            "cashflow": df_to_records(t.cashflow),
            "income_stmt": df_to_records(getattr(t, "income_stmt", None)),
            "quarterly_financials": df_to_records(getattr(t, "quarterly_financials", None)),
            "quarterly_balance_sheet": df_to_records(getattr(t, "quarterly_balance_sheet", None)),
            "quarterly_cashflow": df_to_records(getattr(t, "quarterly_cashflow", None)),
        }
    except Exception:
        fin = {k: [] for k in ["financials","balance_sheet","cashflow","income_stmt","quarterly_financials","quarterly_balance_sheet","quarterly_cashflow"]}

    out = {
        "source": "yahoo",
        "ticker": ticker,
        "asof_utc": utc_now_iso(),
        "info": info,
        "statements": fin,
    }
    return out

def get_fundamentals(ticker: str) -> Dict[str, Any]:
    cache_key = f"fundamentals::{ticker}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    warnings = []
    data = {}
    try:
        data = fetch_fundamentals_yahoo(ticker)
    except Exception as e:
        warnings.append(f"Fundamentals fetch failed: {type(e).__name__}: {str(e)[:200]}")
        data = {"source": "yahoo", "ticker": ticker, "asof_utc": utc_now_iso(), "info": {}, "statements": {}}

    payload = {"asof_utc": utc_now_iso(), "ticker": ticker, "warnings": warnings, "fundamentals": data}
    cache_set(cache_key, payload)
    return payload


# =========================================================
# 5) MULTIPLES ENGINE
# =========================================================
def extract_key_info(fund: Dict[str, Any]) -> Dict[str, Any]:
    info = (fund.get("fundamentals", {}) or {}).get("info", {}) or {}
    # Values often available in yfinance info
    keys = [
        "marketCap", "enterpriseValue", "sharesOutstanding",
        "trailingPE", "forwardPE", "priceToBook",
        "totalRevenue", "ebitda", "grossMargins", "operatingMargins", "profitMargins",
        "beta", "currency", "shortName", "longName", "sector", "industry",
    ]
    out = {}
    for k in keys:
        out[k] = info.get(k, None)
    return out

def compute_multiples_from_info(price: float, key_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes multiples using available fields (no invention).
    """
    mcap = safe_float(key_info.get("marketCap"))
    ev = safe_float(key_info.get("enterpriseValue"))
    shares = safe_float(key_info.get("sharesOutstanding"))
    revenue = safe_float(key_info.get("totalRevenue"))
    ebitda = safe_float(key_info.get("ebitda"))
    pb = safe_float(key_info.get("priceToBook"))
    tpe = safe_float(key_info.get("trailingPE"))
    fpe = safe_float(key_info.get("forwardPE"))

    out = {
        "price": price,
        "market_cap": mcap,
        "enterprise_value": ev,
        "shares_outstanding": shares,
        "trailing_pe": tpe,
        "forward_pe": fpe,
        "price_to_book": pb,
        "ev_to_ebitda": None,
        "price_to_sales": None,
    }

    if ev is not None and ebitda is not None and ebitda != 0:
        out["ev_to_ebitda"] = ev / ebitda
    if mcap is not None and revenue is not None and revenue != 0:
        out["price_to_sales"] = mcap / revenue

    return out

PEERS_MULTIPLES_CSV = os.environ.get("PEERS_MULTIPLES_CSV", "fundamentals/peers_multiples.csv")

def load_peer_multiples_csv(path: str = PEERS_MULTIPLES_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing peer multiples file: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns or "sector" not in df.columns:
        raise ValueError("CSV must contain: ticker, sector")
    for c in ["pe","pb","ps","ev_ebitda"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()
    return df

def zscore(value: float, series: pd.Series) -> float:
    v = safe_float(value)
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not np.isfinite(v) or len(s) < 5:
        return np.nan
    sd = float(s.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return (v - float(s.mean())) / sd

def compute_peer_multiples_zscores(ticker, sector, company_multiples, peers_df,
                                   used_multiples=("pe","pb","ps","ev_ebitda")):
    if not sector:
        return {"available": False, "warnings": ["Missing sector"]}
    sdf = peers_df[peers_df["sector"] == sector]
    if sdf.empty:
        return {"available": False, "warnings": [f"No peers for sector {sector}"]}
    z = {}
    used = []
    for m in used_multiples:
        if m in sdf.columns:
            val = safe_float(company_multiples.get(m))
            zv = zscore(val, sdf[m])
            if np.isfinite(zv):
                z[m] = float(zv)
                used.append(m)
    comp = np.mean(list(z.values())) if z else np.nan
    return {
        "available": bool(z),
        "sector": sector,
        "peer_count": int(len(sdf)),
        "z_by_multiple": z,
        "used_multiples": used,
        "zscore_composite": None if not np.isfinite(comp) else float(comp),
        "warnings": [],
    }

# =========================================================
# 6) DCF ENGINE (FCFF) WITH FULL DISCLOSURE OF ASSUMPTIONS
# =========================================================
@dataclass
class DCFInputs:
    # Core
    revenue: Optional[float]
    ebit_margin: Optional[float]
    tax_rate: float
    reinvestment_rate: Optional[float]   # fraction of after-tax EBIT reinvested
    wacc: float
    forecast_years: int
    terminal_growth: float
    net_debt: Optional[float]
    shares_outstanding: Optional[float]

    # Modeling distributions / ranges (optional; used for Monte Carlo)
    revenue_growth_mu: float
    revenue_growth_sigma: float
    ebit_margin_mu: float
    ebit_margin_sigma: float
    reinvest_mu: float
    reinvest_sigma: float
    wacc_mu: float
    wacc_sigma: float
    terminal_g_mu: float
    terminal_g_sigma: float

def dcf_inputs_from_fundamentals(fund_payload: Dict[str, Any], multiples: Dict[str, Any]) -> Tuple[DCFInputs, List[str]]:
    warnings = []
    key = extract_key_info(fund_payload)

    revenue = safe_float(key.get("totalRevenue"))
    ebitda = safe_float(key.get("ebitda"))
    # EBIT margin often missing; approximate EBIT margin from operatingMargins if exists (not fabricated, it is an info field)
    op_margin = safe_float(key.get("operatingMargins"))
    ebit_margin = op_margin

    # Net debt approximation: EV - MarketCap (only if both present)
    ev = safe_float(key.get("enterpriseValue"))
    mcap = safe_float(key.get("marketCap"))
    net_debt = None
    if ev is not None and mcap is not None:
        net_debt = ev - mcap

    shares = safe_float(key.get("sharesOutstanding"))
    if shares is None and multiples.get("shares_outstanding") is not None:
        shares = safe_float(multiples.get("shares_outstanding"))

    # If revenue missing, DCF can’t run meaningfully
    if revenue is None:
        warnings.append("DCF missing revenue (totalRevenue not available from free feed). DCF may be unavailable.")

    # If EBIT margin missing, we can’t compute after-tax EBIT reliably
    if ebit_margin is None:
        warnings.append("DCF missing operating margin (operatingMargins not available). DCF may be unreliable/unavailable.")

    # Reinvestment rate is not reliably available from free feeds; treat as parameter with disclosure.
    reinvestment_rate = None
    warnings.append("Reinvestment rate is not reliably available from free feeds; modeled as a parameter distribution (disclosed).")

    # WACC likewise; without a clean Saudi rf/ERP feed, treat as parameter distribution (disclosed).
    warnings.append("WACC components (rf/ERP/country risk) are not reliably available for free; WACC is modeled as a parameter distribution (disclosed).")

    inp = DCFInputs(
        revenue=revenue,
        ebit_margin=ebit_margin,
        tax_rate=TAX_RATE_DEFAULT,
        reinvestment_rate=reinvestment_rate,
        wacc=WACC_DEFAULT,
        forecast_years=DCF_FORECAST_YEARS,
        terminal_growth=TERMINAL_GROWTH_DEFAULT,
        net_debt=net_debt,
        shares_outstanding=shares if shares is not None else SHARES_OUTSTANDING_FALLBACK,

        # Distributions (these are modeling priors; user can override via query later)
        revenue_growth_mu=0.06,
        revenue_growth_sigma=0.05,
        ebit_margin_mu=(ebit_margin if ebit_margin is not None else 0.15),
        ebit_margin_sigma=0.05,
        reinvest_mu=0.35,
        reinvest_sigma=0.15,
        wacc_mu=WACC_DEFAULT,
        wacc_sigma=0.03,
        terminal_g_mu=TERMINAL_GROWTH_DEFAULT,
        terminal_g_sigma=0.01,
    )
    return inp, warnings

def dcf_monte_carlo(inputs: DCFInputs, n: int = 2000, seed: int = 7) -> Dict[str, Any]:
    """
    FCFF DCF:
      EBIT = Revenue * EBIT_margin
      NOPAT = EBIT*(1-tax)
      Reinvestment = NOPAT*reinvestment_rate
      FCFF = NOPAT - Reinvestment
      PV = sum(FCFF_t / (1+WACC)^t) + Terminal / (1+WACC)^N
      Terminal = FCFF_N*(1+g) / (WACC - g)  (requires WACC>g)
      Equity = EV - net_debt (net_debt can be negative)
      Intrinsic per share = Equity / shares
    """
    out = {"available": True, "warnings": [], "assumptions": asdict(inputs)}

    if inputs.revenue is None or inputs.ebit_margin is None:
        out["available"] = False
        out["warnings"].append("DCF not available because revenue or EBIT margin is missing from the free data feed.")
        return out

    if inputs.shares_outstanding is None or inputs.shares_outstanding == 0:
        out["available"] = False
        out["warnings"].append("DCF not available per-share because shares outstanding is missing from the free data feed.")
        return out

    rng = np.random.default_rng(seed)

    # Sample parameters
    g = rng.normal(inputs.revenue_growth_mu, inputs.revenue_growth_sigma, size=n)
    m = rng.normal(inputs.ebit_margin_mu, inputs.ebit_margin_sigma, size=n)
    r = rng.normal(inputs.reinvest_mu, inputs.reinvest_sigma, size=n)
    w = rng.normal(inputs.wacc_mu, inputs.wacc_sigma, size=n)
    tg = rng.normal(inputs.terminal_g_mu, inputs.terminal_g_sigma, size=n)

    # Clip to sensible bounds (these are mathematical constraints, not “shortcuts”)
    g = np.clip(g, -0.20, 0.25)
    m = np.clip(m, -0.10, 0.60)
    r = np.clip(r, 0.00, 0.90)
    w = np.clip(w, 0.02, 0.25)
    tg = np.clip(tg, -0.01, 0.06)

    N = inputs.forecast_years
    rev0 = float(inputs.revenue)
    tax = float(inputs.tax_rate)
    net_debt = float(inputs.net_debt) if inputs.net_debt is not None else 0.0
    shares = float(inputs.shares_outstanding)

    per_share_vals = np.full(n, np.nan, dtype=float)
    ev_vals = np.full(n, np.nan, dtype=float)
    bad_terminal = 0

    for i in range(n):
        wi = w[i]
        tgi = tg[i]
        if wi <= tgi + 1e-6:
            bad_terminal += 1
            continue

        rev = rev0
        pv = 0.0
        fcff_N = None

        for t in range(1, N + 1):
            rev = rev * (1.0 + g[i])
            ebit = rev * m[i]
            nopat = ebit * (1.0 - tax)
            reinv = nopat * r[i]
            fcff = nopat - reinv
            pv += fcff / ((1.0 + wi) ** t)
            if t == N:
                fcff_N = fcff

        terminal = (fcff_N * (1.0 + tgi)) / (wi - tgi)
        ev = pv + terminal / ((1.0 + wi) ** N)
        equity = ev - net_debt
        per_share = equity / shares
        ev_vals[i] = ev
        per_share_vals[i] = per_share

    valid = np.isfinite(per_share_vals)
    if valid.sum() < max(50, int(0.05 * n)):
        out["available"] = False
        out["warnings"].append("DCF sampling produced too few valid terminal values (WACC <= g in many draws).")
        out["warnings"].append(f"Invalid terminal draws: {bad_terminal}/{n}")
        return out

    vals = per_share_vals[valid]
    out["warnings"].append(f"Invalid terminal draws filtered: {bad_terminal}/{n}")
    out["distribution"] = {
        "n": int(valid.sum()),
        "p05": float(np.nanpercentile(vals, 5)),
        "p25": float(np.nanpercentile(vals, 25)),
        "p50": float(np.nanpercentile(vals, 50)),
        "p75": float(np.nanpercentile(vals, 75)),
        "p95": float(np.nanpercentile(vals, 95)),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
    }
    # Keep a small sample for audit (not the full 2000 to avoid huge payloads)
    sample_idx = np.linspace(0, len(vals) - 1, num=min(200, len(vals))).astype(int)
    out["sample_values_per_share"] = [float(vals[j]) for j in sample_idx]
    return out


# =========================================================
# 7) PREDICTION ENGINE (WALK-FORWARD, BASELINE + REGULARIZED MODEL)
# =========================================================
def build_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features computed strictly from price/volume (always available if price data exists).
    No leakage: features at time t use data <= t only.
    """
    d = prices_df.copy()
    d = d.sort_values("date").reset_index(drop=True)
    px = d["adj_close"].astype(float)

    d["ret_1"] = log_return(px)
    d["ret_5"] = log_return(px).rolling(5).sum()
    d["ret_21"] = log_return(px).rolling(21).sum()

    d["vol_21"] = d["ret_1"].rolling(21).std()
    d["vol_63"] = d["ret_1"].rolling(63).std()

    d["ma_21"] = px.rolling(21).mean()
    d["ma_63"] = px.rolling(63).mean()
    d["ma_gap_21"] = (px / d["ma_21"]) - 1.0
    d["ma_gap_63"] = (px / d["ma_63"]) - 1.0

    # Volume features
    vol = d["volume"].astype(float)
    d["vol_z_63"] = robust_zscore(vol.rolling(63).mean())

    # RSI-like (simple)
    delta = px.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-12)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Clean
    feature_cols = ["ret_1", "ret_5", "ret_21", "vol_21", "vol_63", "ma_gap_21", "ma_gap_63", "vol_z_63", "rsi_14"]
    d[feature_cols] = d[feature_cols].replace([np.inf, -np.inf], np.nan)
    return d

def walk_forward_backtest(prices_df: pd.DataFrame, horizon_days: int = PREDICTION_HORIZON_DAYS_DEFAULT) -> Dict[str, Any]:
    """
    Predict future horizon return: log(P_{t+h}/P_t).
    Models:
      - Baseline: predict 0 (random walk in log-return space)
      - Ridge regression on engineered features
    Strict walk-forward evaluation.
    """
    out = {"available": True, "warnings": [], "horizon_days": int(horizon_days)}

    if prices_df is None or prices_df.empty or len(prices_df) < MIN_TRAIN_DAYS:
        out["available"] = False
        out["warnings"].append(f"Not enough price history for backtest. Need >= {MIN_TRAIN_DAYS} rows.")
        out["n_rows"] = int(0 if prices_df is None else len(prices_df))
        return out

    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except Exception as e:
        out["available"] = False
        out["warnings"].append("scikit-learn is required for the prediction engine. Install it via requirements.txt.")
        out["warnings"].append(f"Import error: {type(e).__name__}: {str(e)[:200]}")
        return out

    d = build_features(prices_df)
    d = d.dropna(subset=["adj_close"]).copy()
    d = d.sort_values("date").reset_index(drop=True)

    px = d["adj_close"].astype(float)
    future = np.log(px.shift(-horizon_days) / px)
    d["target_h"] = future

    feature_cols = ["ret_1", "ret_5", "ret_21", "vol_21", "vol_63", "ma_gap_21", "ma_gap_63", "vol_z_63", "rsi_14"]
    d_model = d.dropna(subset=feature_cols + ["target_h"]).copy()
    if len(d_model) < MIN_TRAIN_DAYS // 2:
        out["available"] = False
        out["warnings"].append("After feature/target alignment, too few rows remain for backtest.")
        out["n_rows_effective"] = int(len(d_model))
        return out

    X = d_model[feature_cols].values
    y = d_model["target_h"].values
    dates = d_model["date"].tolist()

    # Walk-forward splits
    n = len(d_model)
    start_train = max(0, n - (TRADING_DAYS * 3))  # keep enough training; still walk-forward
    train_end = max(MIN_TRAIN_DAYS, start_train)

    preds_baseline = []
    preds_ridge = []
    actuals = []
    eval_dates = []

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=5.0, random_state=0)),
    ])

    step = WALK_FORWARD_TEST_STEP
    # Ensure we don't run into target leakage; d_model already aligned properly.
    for test_start in range(train_end, n - 1, step):
        train_X = X[:test_start]
        train_y = y[:test_start]

        test_end = min(test_start + step, n)
        test_X = X[test_start:test_end]
        test_y = y[test_start:test_end]

        if len(train_y) < MIN_TRAIN_DAYS:
            continue

        model.fit(train_X, train_y)

        pred_r = model.predict(test_X)
        pred_b = np.zeros_like(test_y)  # baseline = 0 expected log-return

        preds_ridge.extend(pred_r.tolist())
        preds_baseline.extend(pred_b.tolist())
        actuals.extend(test_y.tolist())
        eval_dates.extend(dates[test_start:test_end])

    if len(actuals) < 30:
        out["available"] = False
        out["warnings"].append("Backtest produced too few evaluation points. Increase history or reduce constraints.")
        out["n_eval"] = int(len(actuals))
        return out

    actuals = np.array(actuals, dtype=float)
    preds_ridge = np.array(preds_ridge, dtype=float)
    preds_baseline = np.array(preds_baseline, dtype=float)

    def metrics(pred: np.ndarray, act: np.ndarray) -> Dict[str, Any]:
        mae = float(np.mean(np.abs(pred - act)))
        rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
        da = float(np.mean((pred > 0) == (act > 0)))
        return {"mae": mae, "rmse": rmse, "directional_accuracy": da}

    out["n_eval"] = int(len(actuals))
    out["metrics_baseline"] = metrics(preds_baseline, actuals)
    out["metrics_ridge"] = metrics(preds_ridge, actuals)

    # Calibration-ish: bucket predicted returns and compare realized sign frequency
    bins = [-np.inf, -0.10, -0.03, 0.0, 0.03, 0.10, np.inf]
    labels = ["<=-10%", "(-10,-3]%", "(-3,0]%", "(0,3]%", "(3,10]%", ">10%"]
    b = pd.cut(preds_ridge, bins=bins, labels=labels)
    calib = []
    for lab in labels:
        mask = (b == lab)
        if mask.sum() < 10:
            continue
        frac_up = float(np.mean(actuals[mask] > 0))
        calib.append({"bucket": lab, "n": int(mask.sum()), "frac_actual_up": frac_up})
    out["calibration_ridge"] = calib

    # Keep last trained model inputs for forward prediction
    out["_feature_cols"] = feature_cols
    out["_d_model_tail"] = {
        "last_date": d_model["date"].iloc[-1].isoformat(),
        "last_price": float(d_model["adj_close"].iloc[-1]),
    }
    return out

def forecast_next(prices_df: pd.DataFrame, horizon_days: int) -> Dict[str, Any]:
    """
    Fit ridge on all available data and forecast next horizon log-return distribution proxy:
      - Point forecast from ridge
      - Error bands using historical residual std from backtest
    """
    res = {"available": True, "warnings": [], "horizon_days": int(horizon_days)}

    bt = walk_forward_backtest(prices_df, horizon_days=horizon_days)
    if not bt.get("available", False):
        res["available"] = False
        res["warnings"].append("Forecast unavailable because backtest is unavailable.")
        res["backtest"] = bt
        return res

    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except Exception as e:
        res["available"] = False
        res["warnings"].append("scikit-learn required for forecasting.")
        res["warnings"].append(f"Import error: {type(e).__name__}: {str(e)[:200]}")
        return res

    d = build_features(prices_df).dropna().copy()
    d = d.sort_values("date").reset_index(drop=True)

    px = d["adj_close"].astype(float)
    target = np.log(px.shift(-horizon_days) / px)
    d["target_h"] = target
    feature_cols = bt["_feature_cols"]

    d_model = d.dropna(subset=feature_cols + ["target_h"]).copy()
    if d_model.empty:
        res["available"] = False
        res["warnings"].append("Forecast unavailable after alignment (no rows).")
        return res

    X = d_model[feature_cols].values
    y = d_model["target_h"].values

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=5.0, random_state=0)),
    ])
    model.fit(X, y)

    # Predict from the latest feature row
    last_row = d[feature_cols].iloc[[-1]].values
    pred_logret = float(model.predict(last_row)[0])

    # Use ridge RMSE from backtest as an error proxy
    rmse = safe_float(bt["metrics_ridge"].get("rmse"))
    if rmse is None:
        rmse = 0.10

    # Convert to price forecast distribution bands
    last_price = float(d["adj_close"].iloc[-1])
    p50 = last_price * float(np.exp(pred_logret))
    p16 = last_price * float(np.exp(pred_logret - rmse))
    p84 = last_price * float(np.exp(pred_logret + rmse))

    res["last_date"] = d["date"].iloc[-1].isoformat()
    res["last_price"] = last_price
    res["predicted_log_return"] = pred_logret
    res["error_proxy_rmse"] = float(rmse)
    res["forecast_price_bands"] = {"p16": float(p16), "p50": float(p50), "p84": float(p84)}
    res["backtest_summary"] = {
        "n_eval": bt.get("n_eval"),
        "metrics_baseline": bt.get("metrics_baseline"),
        "metrics_ridge": bt.get("metrics_ridge"),
        "calibration_ridge": bt.get("calibration_ridge", []),
    }
    return res


# =========================================================
# 8) DECISION LAYER
# =========================================================
def valuation_label(current_price: float, dcf: Dict[str, Any], multiples: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decision logic is transparent and conservative.
    If DCF missing -> rely more on multiples, but disclose lower confidence.
    """
    out = {"label": "insufficient_data", "confidence": "low", "reasons": []}

    # Multiples sanity flags (no peer comp set here; only own multiples from info)
    pe = safe_float(multiples.get("trailing_pe"))
    pb = safe_float(multiples.get("price_to_book"))
    ev_ebitda = safe_float(multiples.get("ev_to_ebitda"))

    # DCF-based signal
    if dcf.get("available") and "distribution" in dcf:
        p25 = safe_float(dcf["distribution"].get("p25"))
        p75 = safe_float(dcf["distribution"].get("p75"))
        p50 = safe_float(dcf["distribution"].get("p50"))
        if p25 is not None and p75 is not None and p50 is not None:
            if current_price < 0.85 * p25:
                out["label"] = "undervalued"
                out["confidence"] = "medium"
                out["reasons"].append(f"Market price is below 85% of DCF P25 (conservative intrinsic band).")
            elif current_price > 1.15 * p75:
                out["label"] = "overvalued"
                out["confidence"] = "medium"
                out["reasons"].append(f"Market price is above 115% of DCF P75 (conservative intrinsic band).")
            else:
                out["label"] = "fair_value_zone"
                out["confidence"] = "medium"
                out["reasons"].append("Market price lies within the conservative DCF band (P25–P75).")

            out["reasons"].append(f"DCF P50 vs price: P50={p50:.4g}, Price={current_price:.4g}")

    else:
        out["reasons"].append("DCF not available from free fundamentals; label relies on limited multiples/price-only signals.")
        out["confidence"] = "low"

    # Add multiples context (not decisive without peer set)
    if pe is not None:
        out["reasons"].append(f"Trailing P/E (from feed): {pe:.3g}")
    if pb is not None:
        out["reasons"].append(f"Price/Book (from feed): {pb:.3g}")
    if ev_ebitda is not None:
        out["reasons"].append(f"EV/EBITDA (computed from feed): {ev_ebitda:.3g}")

    return out


# =========================================================
# 9) API ENDPOINTS
# =========================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Saudi Valuator Pro</title>
  <style>
    body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial; margin: 24px; color:#111; }
    .card { border:1px solid #e5e5e5; border-radius:14px; padding:16px; margin:12px 0; }
    input { padding:10px 12px; border-radius:10px; border:1px solid #ccc; width: 240px; }
    button { padding:10px 12px; border-radius:10px; border:0; background:#111; color:#fff; cursor:pointer; }
    pre { background:#0b0b0b; color:#eaeaea; padding:12px; border-radius:12px; overflow:auto; }
    .muted { color:#555; font-size: 13px; }
  </style>
</head>
<body>
  <h2>Saudi Valuator Pro</h2>
  <div class="card">
    <div class="muted">Enter a Tadawul ticker like <b>2222.SR</b>. This UI calls <code>/analyze</code>.</div>
    <div style="margin-top:10px;">
      <input id="ticker" value="2222.SR" />
      <input id="horizon" value="63" style="width:100px;margin-left:6px;" />
      <button onclick="run()">Analyze</button>
    </div>
  </div>
  <div class="card">
    <div class="muted">Output (full audit JSON):</div>
    <pre id="out">{}</pre>
  </div>
<script>
async function run(){
  const t = document.getElementById('ticker').value.trim();
  const h = document.getElementById('horizon').value.trim();
  const url = `/analyze?ticker=${encodeURIComponent(t)}&horizon_days=${encodeURIComponent(h)}`;
  document.getElementById('out').textContent = "Loading...";
  const r = await fetch(url);
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
</script>
</body>
</html>
"""
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Saudi Valuator Pro</title>
  <style>
    :root{
      --bg:#f4f6f9; --card:#fff; --text:#0f172a; --muted:#475569;
      --border:#e2e8f0; --shadow:0 6px 18px rgba(15,23,42,.08);
      --good:#16a34a; --mid:#f59e0b; --bad:#dc2626; --accent:#2563eb;
    }
    body{margin:0;padding:24px;background:var(--bg);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;color:var(--text);}
    .wrap{max-width:1200px;margin:0 auto;}
    h1{margin:0 0 16px 0;font-size:28px;}
    .card{background:var(--card);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:16px;margin:12px 0;}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    input,button{font-size:14px;padding:10px 12px;border-radius:10px;border:1px solid var(--border);outline:none}
    input{min-width:220px}
    button{background:var(--accent);color:#fff;border:none;cursor:pointer}
    button:disabled{opacity:.55;cursor:not-allowed}
    .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}
    .span4{grid-column:span 4}
    .span6{grid-column:span 6}
    .span12{grid-column:span 12}
    .k{color:var(--muted);font-size:12px;margin-bottom:6px}
    .v{font-size:18px;font-weight:650}
    .small{font-size:12px;color:var(--muted)}
    .badge{display:inline-flex;align-items:center;gap:8px;padding:7px 10px;border-radius:999px;border:1px solid var(--border);font-weight:650}
    .dot{width:10px;height:10px;border-radius:999px;background:#94a3b8}
    .good .dot{background:var(--good)} .mid .dot{background:var(--mid)} .bad .dot{background:var(--bad)}
    .good{color:var(--good)} .mid{color:var(--mid)} .bad{color:var(--bad)}
    details pre{white-space:pre-wrap;word-break:break-word;background:#0b1220;color:#e5e7eb;padding:14px;border-radius:12px;overflow:auto;max-height:560px}
    .err{color:var(--bad);font-weight:650}
    @media (max-width: 900px){
      .span4,.span6{grid-column:span 12}
    }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Saudi Valuator Pro</h1>

  <div class="card">
    <div class="row">
      <div>
        <div class="k">Ticker (Tadawul like 2222.SR)</div>
        <input id="ticker" value="1120.SR" />
      </div>
      <div>
        <div class="k">Horizon days</div>
        <input id="horizon" value="63" />
      </div>
      <div style="padding-top:18px">
        <button id="btn">Analyze</button>
      </div>
      <div id="status" class="small"></div>
    </div>
  </div>

  <div class="card" id="headline" style="display:none">
    <div class="row" style="justify-content:space-between">
      <div>
        <div class="badge" id="verdictBadge"><span class="dot"></span><span id="verdictText">—</span></div>
        <div class="small" id="metaLine" style="margin-top:8px"></div>
      </div>
      <div class="small" id="asof"></div>
    </div>
  </div>

  <div class="grid" id="summaryGrid" style="display:none">
    <div class="card span4">
      <div class="k">Current price</div>
      <div class="v" id="px">—</div>
      <div class="small" id="pxMeta">—</div>
    </div>

    <div class="card span4">
      <div class="k">DCF intrinsic (P50)</div>
      <div class="v" id="dcfP50">—</div>
      <div class="small" id="dcfBand">—</div>
    </div>

    <div class="card span4">
      <div class="k">Valuation gap vs P50</div>
      <div class="v" id="gap">—</div>
      <div class="small" id="gapPct">—</div>
    </div>

    <div class="card span6">
      <div class="k">Peer multiples z-score (sector)</div>
      <div class="v" id="peerZ">—</div>
      <div class="small" id="peerDetails">—</div>
    </div>

    <div class="card span6">
      <div class="k">Forecast (horizon)</div>
      <div class="v" id="fc">—</div>
      <div class="small" id="fcBand">—</div>
    </div>

    <div class="card span12">
      <div class="k">Notes / warnings</div>
      <div class="small" id="warn">—</div>
    </div>

    <div class="card span12">
      <details>
        <summary style="cursor:pointer;font-weight:650">Full audit JSON</summary>
        <pre id="audit"></pre>
      </details>
    </div>
  </div>

  <div class="card" id="errorCard" style="display:none">
    <div class="err">Error</div>
    <div class="small" id="errText"></div>
  </div>

</div>

<script>
function fmt(x, digits=2){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, {maximumFractionDigits:digits, minimumFractionDigits:digits});
}
function fmtPct(x, digits=2){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return (n*100).toLocaleString(undefined, {maximumFractionDigits:digits, minimumFractionDigits:digits}) + "%";
}
function setBadge(verdict){
  const b = document.getElementById("verdictBadge");
  b.classList.remove("good","mid","bad");
  if (!verdict) { b.classList.add("mid"); return; }
  const v = verdict.toLowerCase();
  if (v.includes("under")) b.classList.add("good");
  else if (v.includes("over")) b.classList.add("bad");
  else b.classList.add("mid");
}

function pick(obj, path){
  try{
    return path.split(".").reduce((a,k)=> (a && a[k] !== undefined) ? a[k] : undefined, obj);
  }catch(e){ return undefined; }
}

async function run(){
  const ticker = document.getElementById("ticker").value.trim();
  const horizon = document.getElementById("horizon").value.trim();
  const btn = document.getElementById("btn");
  const status = document.getElementById("status");
  const errorCard = document.getElementById("errorCard");
  errorCard.style.display = "none";

  btn.disabled = true;
  status.textContent = "Running /analyze …";

  try{
    const url = `/analyze?ticker=${encodeURIComponent(ticker)}&horizon_days=${encodeURIComponent(horizon)}`;
    const r = await fetch(url);
    if(!r.ok){
      const t = await r.text();
      throw new Error(`HTTP ${r.status}: ${t}`);
    }
    const data = await r.json();

    // Headline
    document.getElementById("headline").style.display = "block";
    document.getElementById("summaryGrid").style.display = "grid";

    const name = pick(data,"market_snapshot.name") || "—";
    const sector = pick(data,"market_snapshot.sector") || "—";
    const industry = pick(data,"market_snapshot.industry") || "—";
    document.getElementById("metaLine").textContent = `${name} • ${sector} • ${industry}`;
    document.getElementById("asof").textContent = `asof_utc: ${pick(data,"asof_utc") || "—"}`;

    // Price
    const lastPx = pick(data,"market_snapshot.last_price");
    document.getElementById("px").textContent = fmt(lastPx, 3);
    document.getElementById("pxMeta").textContent = `last_date: ${pick(data,"market_snapshot.last_date") || "—"} • currency: ${pick(data,"market_snapshot.currency") || "—"}`;

    // DCF distribution (expecting keys; if your JSON uses different names, adjust here only)
    const dcfP50 = pick(data,"valuation.dcf.p50") ?? pick(data,"dcf.p50");
    const dcfP25 = pick(data,"valuation.dcf.p25") ?? pick(data,"dcf.p25");
    const dcfP75 = pick(data,"valuation.dcf.p75") ?? pick(data,"dcf.p75");
    document.getElementById("dcfP50").textContent = fmt(dcfP50, 3);
    document.getElementById("dcfBand").textContent = `P25 ${fmt(dcfP25,3)}  •  P75 ${fmt(dcfP75,3)}`;

    // Gap
    if (Number.isFinite(Number(lastPx)) && Number.isFinite(Number(dcfP50))){
      const gap = Number(dcfP50) - Number(lastPx);
      const gp = gap / Number(lastPx);
      document.getElementById("gap").textContent = fmt(gap, 3);
      document.getElementById("gapPct").textContent = fmtPct(gp, 2);
    } else {
      document.getElementById("gap").textContent = "—";
      document.getElementById("gapPct").textContent = "—";
    }

    // Peer z-score
    const z = pick(data,"peer_multiples.zscore_composite");
    const zLabel = (z===undefined) ? "—" : fmt(z, 2);
    document.getElementById("peerZ").textContent = zLabel;
    const peerNote = pick(data,"peer_multiples.note") || "—";
    const used = pick(data,"peer_multiples.used_multiples") || [];
    document.getElementById("peerDetails").textContent =
      `Used: ${Array.isArray(used) ? used.join(", ") : "—"} • ${peerNote}`;

    // Forecast
    const fcP50 = pick(data,"forecast.p50");
    const fcP25 = pick(data,"forecast.p25");
    const fcP75 = pick(data,"forecast.p75");
    document.getElementById("fc").textContent = fmt(fcP50, 3);
    document.getElementById("fcBand").textContent = `P25 ${fmt(fcP25,3)}  •  P75 ${fmt(fcP75,3)}`;

    // Verdict
    const verdict = pick(data,"decision.verdict") || pick(data,"verdict");
    document.getElementById("verdictText").textContent = verdict || "—";
    setBadge(verdict);

    // Warnings
    const w = [];
    const dq = pick(data,"data_quality");
    if (dq && dq.price_warnings) w.push(...dq.price_warnings);
    if (dq && dq.fundamentals_warnings) w.push(...dq.fundamentals_warnings);
    const pmw = pick(data,"peer_multiples.warnings");
    if (Array.isArray(pmw)) w.push(...pmw);
    document.getElementById("warn").textContent = w.length ? w.join(" | ") : "None";

    // Audit JSON
    document.getElementById("audit").textContent = JSON.stringify(data, null, 2);

    status.textContent = "Done.";
  } catch(e){
    document.getElementById("errorCard").style.display = "block";
    document.getElementById("errText").textContent = String(e);
    status.textContent = "";
  } finally{
    btn.disabled = false;
  }
}

document.getElementById("btn").addEventListener("click", run);
</script>
</body>
</html>
    """
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return home()
@app.get("/prices")
def prices(ticker: str = Query(..., description="e.g., 2222.SR")):
    payload = get_prices_with_fallback(ticker)
    return JSONResponse(payload)

@app.get("/fundamentals")
def fundamentals(ticker: str = Query(..., description="e.g., 2222.SR")):
    payload = get_fundamentals(ticker)
    return JSONResponse(payload)

@app.get("/backtest")
def backtest(
    ticker: str = Query(..., description="e.g., 2222.SR"),
    horizon_days: int = Query(PREDICTION_HORIZON_DAYS_DEFAULT, ge=5, le=252),
):
    p = get_prices_with_fallback(ticker)
    df = pd.DataFrame(p.get("prices", []))
    df = df_clean_prices(df)
    bt = walk_forward_backtest(df, horizon_days=horizon_days)
    return JSONResponse({"ticker": ticker, "asof_utc": utc_now_iso(), "backtest": bt, "price_quality": p.get("quality"), "price_warnings": p.get("warnings")})

@app.get("/analyze")
def analyze(
    ticker: str = Query(..., description="e.g., 2222.SR"),
    horizon_days: int = Query(PREDICTION_HORIZON_DAYS_DEFAULT, ge=5, le=252),
    dcf_samples: int = Query(2000, ge=500, le=20000),
):
    # Prices
    price_payload = get_prices_with_fallback(ticker)
    dfp = pd.DataFrame(price_payload.get("prices", []))
    dfp = df_clean_prices(dfp)

    if dfp.empty:
        return JSONResponse(
            {
                "ticker": ticker,
                "asof_utc": utc_now_iso(),
                "errors": ["No price data available from Yahoo/TwelveData/AlphaVantage."],
                "price_meta": price_payload.get("meta"),
                "price_warnings": price_payload.get("warnings"),
                "price_quality": price_payload.get("quality"),
            },
            status_code=400,
        )

    current_price = float(dfp["adj_close"].iloc[-1])
    current_date = dfp["date"].iloc[-1].isoformat()

    # Fundamentals
    fund_payload = get_fundamentals(ticker)
    key_info = extract_key_info(fund_payload)

    # Multiples
multiples = compute_multiples_from_info(current_price, key_info)
company_multiples = {
    "pe": multiples.get("trailing_pe"),
    "pb": multiples.get("price_to_book"),
    "ps": multiples.get("price_to_sales"),
    "ev_ebitda": multiples.get("ev_to_ebitda"),
}

try:
    peers_df = load_peer_multiples_csv()
    peer_block = compute_peer_multiples_zscores(
        ticker=ticker,
        sector=key_info.get("sector"),
        company_multiples=company_multiples,
        peers_df=peers_df,
    )
except Exception as e:
    peer_block = {"available": False, "warnings": [str(e)]}

    # DCF
    dcf_inputs, dcf_warnings = dcf_inputs_from_fundamentals(fund_payload, multiples)
    dcf_res = dcf_monte_carlo(dcf_inputs, n=dcf_samples, seed=7)
    if dcf_warnings:
        dcf_res.setdefault("warnings", [])
        dcf_res["warnings"] = list(dcf_res["warnings"]) + dcf_warnings

    # Forecasting
    forecast_res = forecast_next(dfp, horizon_days=horizon_days)

    # Decision label
    decision = valuation_label(current_price, dcf_res, multiples)

    # Assemble audit pack
    audit = {
        "ticker": ticker,
        "asof_utc": utc_now_iso(),
        "market_snapshot": {
            "last_date": current_date,
            "last_price": current_price,
            "currency": key_info.get("currency"),
            "name": key_info.get("longName") or key_info.get("shortName"),
            "sector": key_info.get("sector"),
            "peer_multiples": peer_block,
            "industry": key_info.get("industry"),
        },
        "data_quality": {
            "prices": price_payload.get("quality"),
            "price_warnings": price_payload.get("warnings", []),
            "price_sources_tried": (price_payload.get("meta", {}) or {}).get("sources_tried", []),
            "fundamentals_warnings": fund_payload.get("warnings", []),
        },
        "fundamentals_raw": fund_payload.get("fundamentals", {}),  # full raw snapshot for audit
        "multiples": multiples,
        "dcf": dcf_res,
        "prediction": forecast_res,
        "decision": decision,
        "disclosures": [
            "This app does not fabricate missing fundamentals. When free feeds do not provide required fields, DCF may be unavailable or have lower confidence.",
            "Prediction is evaluated using walk-forward (time-series) backtesting; performance is reported versus a naive baseline.",
            "DCF uses parameter distributions for reinvestment rate and WACC when not available from free feeds, and these assumptions are shown in the output.",
        ],
    }
    return JSONResponse(audit)


# =========================================================
# 10) LOCAL RUN
# =========================================================
def app_main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

if __name__ == "__main__":
    app_main()
