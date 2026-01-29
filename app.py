# =========================================================
# app.py — PART 1 (CLEANED)
# Foundation, configuration, utilities
# Notes:
# - Removed nothing you need later.
# - Kept ONE copy of each helper (avoid later “last definition wins” bugs).
# - Ensured tz-naive normalization is consistent.
# - Kept / and /health exactly as you had.
# =========================================================

from __future__ import annotations

import os
import math
import random
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================================================
# 0) APP + CORS
# =========================================================
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Saudi Valuator Pro</title>
  <style>
    body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    a { color: #0b63ce; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .card { max-width: 820px; padding: 16px 18px; border: 1px solid #e5e7eb; border-radius: 12px; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Saudi Valuator Pro is running.</h2>
    <p>Use the UI here: <a href="/ui">/ui</a></p>
    <p>Or the API docs here: <a href="/docs">/docs</a></p>
    <p>Or call <code>POST /analyze</code> with JSON like <code>{"ticker":"2222"}</code>.</p>
  </div>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"ok": True}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Saudi Valuator Pro — UI</title>
  <style>
    body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .wrap { max-width: 980px; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    input { padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 10px; min-width: 220px; }
    button { padding: 10px 14px; border: 0; border-radius: 10px; background: #0b63ce; color: white; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .card { margin-top: 16px; padding: 14px 16px; border: 1px solid #e5e7eb; border-radius: 12px; }
    pre { background: #0b1020; color: #d1e7ff; padding: 12px; border-radius: 12px; overflow: auto; }
    .muted { color: #6b7280; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Saudi Valuator Pro — Web UI</h2>
    <div class="row">
      <input id="ticker" placeholder="Ticker (e.g., 2222 or 2222.SR)" />
      <button id="runBtn" onclick="run()">Analyze</button>
      <span class="muted" id="status"></span>
    </div>

    <div class="card">
      <div class="muted">Result (raw JSON):</div>
      <pre id="out">{}</pre>
    </div>

    <div class="card">
      <div class="muted">Quick links:</div>
      <div><a href="/docs">Open API Docs (/docs)</a></div>
      <div><a href="/">Back to Home (/)</a></div>
    </div>
  </div>

<script>
async function run() {
  const btn = document.getElementById("runBtn");
  const status = document.getElementById("status");
  const out = document.getElementById("out");
  const ticker = document.getElementById("ticker").value.trim();

  if (!ticker) {
    status.textContent = "Enter a ticker first.";
    return;
  }

  btn.disabled = true;
  status.textContent = "Running...";
  out.textContent = "{}";

  try {
    const r = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker })
    });

    const data = await r.json();
    out.textContent = JSON.stringify(data, null, 2);

    if (data && data.error) {
      status.textContent = "Done (with error message in JSON).";
    } else {
      status.textContent = "Done.";
    }
  } catch (e) {
    status.textContent = "Failed: " + (e?.message || e);
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

# =========================================================
# 1) GLOBAL CONFIGURATION
# =========================================================
DEFAULT_HISTORY_PERIOD = "5y"
TRADING_DAYS = 252

# Lookback windows
BETA_LOOKBACK_DAYS = TRADING_DAYS * 2
MARKET_RETURN_LOOKBACK_DAYS = TRADING_DAYS * 5
TRAIN_WINDOW_DAYS = TRADING_DAYS * 3
TEST_WINDOW_DAYS = TRADING_DAYS * 1

# Optimization
SOLVER_SAMPLE_STEP = 5
N_WEIGHT_SAMPLES = 6000

# Forecast
FORECAST_YEARS = 5
SPREAD_HORIZON_DAYS = 21  # ~1 month

# Market index
TASI_TICKER = "^TASI.SR"

# API keys (already provided by you)
ALPHA_VANTAGE_KEY = "0LR5JLOBSLOA6Z0A"
TWELVE_DATA_KEY = "ed240f406bab4225ac6e0a98be553aa2"

# Risk-free Excel
RISK_FREE_XLSX_PATH = "saudi_yields.xlsx"
RISK_FREE_COLUMN_NAME = "10-Year government bond yield"

# Hard safety bounds (not assumptions)
GROWTH_MIN = -0.20
GROWTH_MAX = 0.40
WACC_MAX = 0.50

# =========================================================
# 2) JSON-SAFE SERIALIZATION
# =========================================================
def json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return float(obj)

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    return obj


# =========================================================
# 3) BASIC HELPERS (single authoritative versions)
# =========================================================
def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def safe_div(a: float, b: float) -> Optional[float]:
    try:
        if b is None:
            return None
        if not np.isfinite(a) or not np.isfinite(b) or float(b) == 0.0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


def winsorize(arr: np.ndarray, p_low: float = 0.05, p_high: float = 0.95) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return x
    lo = np.quantile(x[mask], p_low)
    hi = np.quantile(x[mask], p_high)
    return np.clip(x, lo, hi)


def last_value_on_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill a low-frequency series (quarterly/annual)
    to a daily index based on last known report date.

    Critical: normalize tz to tz-naive to avoid:
      TypeError: Cannot compare tz-naive and tz-aware timestamps
    """
    if series is None or series.empty:
        return pd.Series(index=dates, dtype=float)

    s = series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)

    d = pd.to_datetime(dates)
    d = d.tz_localize(None) if getattr(d, "tz", None) is not None else d

    tmp = s.reindex(s.index.union(d)).sort_index().ffill()
    out = tmp.reindex(d)
    out.index = dates  # keep original index object
    return out.astype(float)


def ttm_from_quarters(q_series: pd.Series) -> pd.Series:
    """
    Compute trailing-12-month series from quarterly data.
    Index must be tz-naive datetime-like.
    """
    s = q_series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index().rolling(4, min_periods=4).sum()
# =========================================================
# app.py — PART 2 (CLEANED)
# Data fetching (prices + statements + risk-free) + statement extractors
# + market/beta helpers
#
# Changes made:
# - Enforced tz-naive normalization consistently (prices + statements).
# - Added de-duplication + sorting on indices (prevents slice/loc surprises).
# - Made Twelve Data parsing robust to timezone strings and reversed order.
# - Alpha Vantage: defensive parsing + rate-limit message handling preserved via error msg.
# - Statement extractors: normalize BOTH series index and columns safely.
# - StockRequest stays here (fine).
# =========================================================

class DataFetcher:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ]

    def _headers(self):
        return {"User-Agent": random.choice(self.user_agents)}

    @staticmethod
    def clean_saudi_ticker(ticker: str) -> str:
        t = (ticker or "").strip().upper()
        if t.replace(".", "").isdigit() and not t.endswith(".SR"):
            return f"{t}.SR"
        return t

    @staticmethod
    def _tz_naive_index(idx) -> pd.DatetimeIndex:
        di = pd.to_datetime(idx, errors="coerce")
        # di can be DatetimeIndex or Series; normalize to DatetimeIndex
        if not isinstance(di, pd.DatetimeIndex):
            di = pd.DatetimeIndex(di)
        # Strip tz if present
        try:
            di = di.tz_localize(None)
        except Exception:
            # already naive or mixed; best-effort via elementwise conversion
            di = pd.DatetimeIndex(pd.to_datetime(di.astype("datetime64[ns]"), errors="coerce"))
        return di

    @staticmethod
    def _clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        out.index = DataFetcher._tz_naive_index(out.index)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    # ---------- Prices ----------
    def fetch_prices_yahoo(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=False)

        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError(f"No Yahoo price history for {ticker}.")

        hist = self._clean_price_df(hist)

        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
        hist = hist[cols]

        if "Close" not in hist.columns or hist["Close"].dropna().empty:
            raise ValueError(f"Yahoo returned empty Close series for {ticker}.")

        return hist

    def fetch_prices_twelve(self, ticker: str) -> pd.DataFrame:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": 1250,
            "apikey": TWELVE_DATA_KEY,
            "format": "JSON",
        }
        r = requests.get(url, params=params, headers=self._headers(), timeout=12)
        if r.status_code != 200:
            raise ValueError(f"Twelve Data request failed ({r.status_code}).")

        data = r.json()

        # TwelveData sometimes returns {"status":"error","message":...}
        if isinstance(data, dict) and data.get("status") == "error":
            raise ValueError(f"Twelve Data: {data.get('message') or 'Unknown error'}")

        values = data.get("values")
        if not isinstance(values, list) or len(values) == 0:
            msg = data.get("message") or "No Twelve Data values."
            raise ValueError(f"Twelve Data: {msg}")

        rows = []
        for v in values:
            try:
                # tz-naive parse even if datetime includes timezone
                dt = pd.to_datetime(v.get("datetime"), errors="coerce")
                if pd.isna(dt):
                    continue
                dt = dt.tz_localize(None) if getattr(dt, "tzinfo", None) is not None else dt
                close = float(v.get("close"))
                if not np.isfinite(close):
                    continue
                rows.append((dt, close))
            except Exception:
                continue

        if not rows:
            raise ValueError("Twelve Data: could not parse values.")

        df = pd.DataFrame(rows, columns=["Date", "Close"]).set_index("Date")
        df.index = self._tz_naive_index(df.index)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Some TwelveData responses are reverse-chron; sort_index already fixed that.
        if df.empty:
            raise ValueError("Twelve Data: empty parsed dataframe.")

        return df[["Close"]]

    def fetch_prices_alpha_vantage(self, ticker: str) -> pd.DataFrame:
        av_symbol = ticker.replace(".SR", ".SA")
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": av_symbol,
            "outputsize": "full",
            "apikey": ALPHA_VANTAGE_KEY,
        }
        r = requests.get(url, params=params, headers=self._headers(), timeout=12)
        if r.status_code != 200:
            raise ValueError(f"Alpha Vantage request failed ({r.status_code}).")

        data = r.json()
        ts = data.get("Time Series (Daily)")
        if not ts:
            msg = data.get("Note") or data.get("Error Message") or "No daily series."
            raise ValueError(f"Alpha Vantage: {msg}")

        df = pd.DataFrame.from_dict(ts, orient="index")
        if "4. close" not in df.columns:
            raise ValueError("Alpha Vantage: missing close field.")

        df = df.rename(columns={"4. close": "Close"})
        df.index = self._tz_naive_index(df.index)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index().tail(1250)

        if df.empty:
            raise ValueError("Alpha Vantage: empty parsed dataframe.")

        return df[["Close"]]

    def fetch_prices(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> Tuple[pd.DataFrame, str]:
        # Try sources in order; return first that works
        try:
            return self.fetch_prices_yahoo(ticker, period=period), "Yahoo Finance"
        except Exception:
            pass

        try:
            return self.fetch_prices_twelve(ticker), "Twelve Data"
        except Exception:
            pass

        try:
            return self.fetch_prices_alpha_vantage(ticker), "Alpha Vantage"
        except Exception as e:
            raise ValueError(f"All price sources failed: {str(e)}")

    # ---------- Statements (Yahoo via yfinance) ----------
    def fetch_statements_yahoo(self, ticker: str) -> Dict[str, Any]:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        try:
            info = stock.info or {}
        except Exception:
            info = {}

        def safe_attr(name: str):
            try:
                return getattr(stock, name)
            except Exception:
                return None

        fin_a = safe_attr("financials")
        bs_a = safe_attr("balance_sheet")
        cf_a = safe_attr("cashflow")

        fin_q = safe_attr("quarterly_financials")
        bs_q = safe_attr("quarterly_balance_sheet")
        cf_q = safe_attr("quarterly_cashflow")

        def norm_cols(df: Any) -> Any:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return df
            out = df.copy()
            try:
                out.columns = pd.to_datetime(out.columns, errors="coerce").tz_localize(None)
            except Exception:
                pass
            # Ensure stable ordering and no duplicate report dates
            try:
                out = out.loc[:, ~out.columns.duplicated(keep="last")]
                out = out.reindex(sorted(out.columns), axis=1)
            except Exception:
                pass
            return out

        return {
            "info": info,
            "financials_annual": norm_cols(fin_a),
            "balance_sheet_annual": norm_cols(bs_a),
            "cashflow_annual": norm_cols(cf_a),
            "financials_quarterly": norm_cols(fin_q),
            "balance_sheet_quarterly": norm_cols(bs_q),
            "cashflow_quarterly": norm_cols(cf_q),
        }

    # ---------- Risk-free (Excel) ----------
    def fetch_saudi_risk_free_from_excel(self, path: str, column_name: str) -> float:
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            raise ValueError(
                f"Failed to read Excel '{path}'. Install openpyxl and ensure the file exists. Detail: {str(e)}"
            )

        if df is None or df.empty:
            raise ValueError("Excel file is empty.")

        col = None
        for c in df.columns:
            if str(c).strip().lower() == str(column_name).strip().lower():
                col = c
                break
        if col is None:
            raise ValueError(f"Column '{column_name}' not found in Excel. Available columns: {list(df.columns)}")

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            raise ValueError(f"No numeric values found in column '{column_name}'.")

        last_val = float(s.iloc[-1])
        rf = last_val / 100.0 if last_val > 1.0 else last_val

        if not np.isfinite(rf) or rf <= 0 or rf > 0.50:
            raise ValueError(f"Risk-free out of bounds after parsing: {rf}")

        return rf


# =========================================================
# Statement row extractors (robust to naming differences)
# =========================================================
def _row_lookup(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    idx_lower = {str(i).strip().lower(): i for i in df.index}
    for n in names:
        key = str(n).strip().lower()
        if key in idx_lower:
            return df.loc[idx_lower[key]]
    return None


def _row_contains(df: pd.DataFrame, must_contain: List[str]) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    must = [m.lower() for m in must_contain]
    for idx in df.index:
        s = str(idx).lower()
        if all(m in s for m in must):
            return df.loc[idx]
    return None


def _series_from_row(
    df: pd.DataFrame,
    row_names: List[str],
    contains: Optional[List[str]] = None
) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    r = _row_lookup(df, row_names)
    if r is None and contains is not None:
        r = _row_contains(df, contains)
    if r is None:
        return None

    s = pd.to_numeric(r, errors="coerce")

    # Statement row "s" is typically indexed by report dates (columns of the dataframe)
    try:
        s.index = pd.to_datetime(s.index, errors="coerce").tz_localize(None)
    except Exception:
        return None

    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# =========================================================
# Market / beta helpers
# =========================================================
def annualized_geo_mean_return(prices: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) < periods_per_year + 1:
        raise ValueError("Not enough history to estimate market return.")

    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    n = len(prices) - 1
    years = n / periods_per_year

    if start <= 0 or end <= 0 or years <= 0:
        raise ValueError("Invalid series for return estimation.")

    return (end / start) ** (1.0 / years) - 1.0


def beta_regression(stock_prices: pd.Series, market_prices: pd.Series) -> float:
    s = pd.to_numeric(stock_prices, errors="coerce")
    m = pd.to_numeric(market_prices, errors="coerce")

    df = pd.DataFrame({"s": s, "m": m}).dropna()
    if len(df) < 120:
        raise ValueError("Not enough overlapping history for beta.")

    rs = np.log(df["s"].replace(0.0, np.nan)).diff().dropna()
    rm = np.log(df["m"].replace(0.0, np.nan)).diff().dropna()

    aligned = pd.DataFrame({"rs": rs, "rm": rm}).dropna()
    if len(aligned) < 120:
        raise ValueError("Not enough return observations for beta.")

    cov = np.cov(aligned["rs"], aligned["rm"], ddof=1)[0, 1]
    var = np.var(aligned["rm"], ddof=1)

    if var <= 0 or not np.isfinite(var):
        raise ValueError("Market variance invalid; cannot compute beta.")

    b = float(cov / var)
    if not np.isfinite(b):
        raise ValueError("Beta is not finite.")

    return b


# =========================================================
# Request model
# =========================================================
class StockRequest(BaseModel):
    ticker: str

# =========================================================
# app.py — PART 3 (CLEANED)
# Helpers (time-alignment + robustness) + Valuation models
# + Valuation-anchor weight tuning + Spread engine (features + walk-forward)
#
# Key fixes vs your PART 3:
# 1) REMOVED duplicates: safe_div, _to_float, ttm_from_quarters, last_value_on_or_before, winsorize
#    (they already exist in PART 1). Duplicates can silently override and break logic.
# 2) Enforced tz-naive index normalization inside spread engine too (prevents tz-aware vs tz-naive).
# 3) Fixed fillna(method=...) compatibility (pandas 3 removed it): always use .ffill()/.bfill().
# 4) Made ridge robust: handles NaNs via row filtering + standardization; adds intercept.
# 5) Walk-forward: prevents lookahead explicitly and requires minimum training rows.
#
# IMPORTANT:
# - Because PART 1 already defines: _to_float, safe_div, winsorize, last_value_on_or_before, ttm_from_quarters,
#   this PART 3 assumes those exist and DOES NOT redefine them.
# =========================================================

# =========================================================
# TZ utilities (needed here because spread engine uses daily indices heavily)
# =========================================================
def _to_naive_dt_index(idx) -> pd.DatetimeIndex:
    di = pd.to_datetime(idx, errors="coerce")
    if not isinstance(di, pd.DatetimeIndex):
        di = pd.DatetimeIndex(di)
    try:
        di = di.tz_localize(None)
    except Exception:
        # if already tz-naive or mixed, this will fail; keep best-effort
        pass
    return di


# =========================================================
# Valuation models
# =========================================================
def dcf_per_share_from_fcff(
    fcff0: float,
    wacc: float,
    g: float,
    shares: float,
    net_debt: float,
    market_long_run_g: float,
    years: int = FORECAST_YEARS,
) -> float:
    """
    FCFF-based DCF -> equity per-share.
    Terminal growth capped by market long-run CAGR (data-driven).
    """
    if shares <= 0:
        raise ValueError("shares <= 0")
    if not np.isfinite(fcff0) or fcff0 <= 0:
        raise ValueError("fcff0 must be positive and finite")
    if not np.isfinite(wacc) or wacc <= 0 or wacc > WACC_MAX:
        raise ValueError("wacc must be positive and finite")
    if not np.isfinite(g):
        raise ValueError("g not finite")

    g_term = min(float(g), float(market_long_run_g))
    if wacc <= g_term:
        raise ValueError("WACC <= terminal growth")

    pv_sum = 0.0
    last = None
    for i in range(1, years + 1):
        fcff_i = fcff0 * ((1.0 + g) ** i)
        pv_sum += fcff_i / ((1.0 + wacc) ** i)
        last = fcff_i

    tv = (last * (1.0 + g_term)) / (wacc - g_term)
    pv_tv = tv / ((1.0 + wacc) ** years)

    ev = pv_sum + pv_tv
    equity_value = ev - net_debt
    return equity_value / shares


# =========================================================
# Robust weight tuning for valuation anchor
# =========================================================
def robust_loss_mape(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0)
    if mask.sum() == 0:
        return float("inf")
    yy = y[mask]
    yh = yhat[mask]
    ape = np.abs((yh - yy) / yy)
    ape = winsorize(ape, 0.02, 0.98)
    return float(np.mean(ape) * 100.0)


def optimize_weights_dirichlet(
    y: np.ndarray,
    X: np.ndarray,
    avail: np.ndarray,
    n_samples: int = N_WEIGHT_SAMPLES,
) -> np.ndarray:
    """
    Sample weights from Dirichlet; keep best by robust MAPE.
    X shape: (n_models, n_points)
    """
    n_models = int(X.shape[0])
    if not np.any(avail):
        raise ValueError("No models available")

    idx = np.where(avail)[0]
    k = int(len(idx))

    best_w_full = np.zeros(n_models, dtype=float)
    best_loss = float("inf")

    rnd = np.random.default_rng(42)

    candidates: List[np.ndarray] = []
    # pure single-model baselines
    for j in idx:
        w = np.zeros(n_models, dtype=float)
        w[int(j)] = 1.0
        candidates.append(w)

    # Dirichlet mixtures over available models only
    draws = rnd.dirichlet(np.ones(k, dtype=float), size=int(n_samples))
    for d in draws:
        w = np.zeros(n_models, dtype=float)
        w[idx] = d
        candidates.append(w)

    for w in candidates:
        yhat = np.nansum(X.T * w, axis=1)
        loss = robust_loss_mape(y, yhat)
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_w_full = w.copy()

    if best_w_full.sum() <= 0:
        raise ValueError("Weight search failed")

    return best_w_full / best_w_full.sum()


# =========================================================
# Spread engine: predict short-horizon valuation spread dynamics (not pure price)
# Pure numpy ridge (no sklearn) + walk-forward OOS
# =========================================================
SPREAD_HORIZON_DAYS = 21

RIDGE_L2 = 10.0
MIN_TRAIN_ROWS = 220  # require enough rows after shifting


def _zscore(s: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (s - mu) / sd


def _logret(px: pd.Series) -> pd.Series:
    p = pd.to_numeric(px, errors="coerce").astype(float).replace(0.0, np.nan)
    return np.log(p).diff()


def _rolling_vol(px: pd.Series, window: int) -> pd.Series:
    r = _logret(px)
    return r.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def build_spread_features(df_core: pd.DataFrame, shares_daily: pd.Series) -> pd.DataFrame:
    """
    Required columns:
      - Close, MktClose, V_anchor
    Optional:
      - Volume
    """
    df = df_core.copy()

    df.index = _to_naive_dt_index(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for c in ["Close", "MktClose", "V_anchor"]:
        if c not in df.columns:
            raise ValueError(f"build_spread_features missing required column: {c}")

    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    mkt = pd.to_numeric(df["MktClose"], errors="coerce").astype(float)
    V = pd.to_numeric(df["V_anchor"], errors="coerce").astype(float)

    spread = (close - V) / V.replace(0.0, np.nan)
    abs_spread = spread.abs()

    # returns/momentum
    r1 = close.pct_change(1)
    r5 = close.pct_change(5)
    r21 = close.pct_change(21)

    mkt_r5 = mkt.pct_change(5)
    mkt_r21 = mkt.pct_change(21)

    # vol regime
    vol21 = _rolling_vol(close, 21)
    vol63 = _rolling_vol(close, 63)
    mkt_vol63 = _rolling_vol(mkt, 63)

    # liquidity
    if "Volume" in df.columns and df["Volume"].notna().any():
        volu = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
        dollar_vol = (volu * close).replace([np.inf, -np.inf], np.nan)
        liq_z = _zscore(np.log1p(dollar_vol), 63)
    else:
        liq_z = pd.Series(index=df.index, dtype=float)

    # mcap drift proxy
    sh = pd.to_numeric(shares_daily.reindex(df.index), errors="coerce").astype(float)
    mcap = (sh * close).replace([np.inf, -np.inf], np.nan)
    mcap_z = _zscore(np.log1p(mcap), 252)

    # mean reversion
    spread_chg_5 = spread.diff(5)
    spread_chg_21 = spread.diff(21)

    feat = pd.DataFrame(index=df.index)
    feat["spread"] = spread
    feat["abs_spread"] = abs_spread
    feat["r1"] = r1
    feat["r5"] = r5
    feat["r21"] = r21
    feat["mkt_r5"] = mkt_r5
    feat["mkt_r21"] = mkt_r21
    feat["vol21"] = vol21
    feat["vol63"] = vol63
    feat["mkt_vol63"] = mkt_vol63
    feat["liq_z"] = liq_z
    feat["mcap_z"] = mcap_z
    feat["spread_chg_5"] = spread_chg_5
    feat["spread_chg_21"] = spread_chg_21

    # clean + robustify
    for c in feat.columns:
        x = feat[c].replace([np.inf, -np.inf], np.nan).values.astype(float)
        feat[c] = pd.Series(winsorize(x, 0.01, 0.99), index=feat.index)

    return feat


def _ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    lam: float = RIDGE_L2,
) -> float:
    """
    Ridge with intercept and standardization.
    """
    if X_train.ndim != 2:
        return float("nan")

    # rows where y and all X are finite
    ok = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
    Xt = X_train[ok]
    yt = y_train[ok]

    if yt.size < 80:
        return float("nan")

    mu = Xt.mean(axis=0)
    sd = Xt.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)

    Xt_s = (Xt - mu) / sd
    xp_s = (X_pred - mu) / sd

    # add intercept
    Xt_b = np.hstack([np.ones((Xt_s.shape[0], 1)), Xt_s])
    xp_b = np.hstack([np.ones((1, 1)), xp_s.reshape(1, -1)])

    p = Xt_b.shape[1]
    I = np.eye(p)
    I[0, 0] = 0.0  # don't penalize intercept

    A = Xt_b.T @ Xt_b + lam * I
    b = Xt_b.T @ yt

    try:
        w = np.linalg.solve(A, b)
    except Exception:
        return float("nan")

    return float((xp_b @ w).ravel()[0])


def walk_forward_spread_forecast(
    df_feat: pd.DataFrame,
    y_target: pd.Series,
    horizon: int = SPREAD_HORIZON_DAYS,
    train_days: int = TRAIN_WINDOW_DAYS,
    test_days: int = TEST_WINDOW_DAYS,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Predict y(t) using features(t-horizon).
    Walk-forward OOS only (last `test_days` points).
    """
    df = df_feat.copy()
    df.index = _to_naive_dt_index(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    y = y_target.copy()
    y.index = _to_naive_dt_index(y.index)
    y = y[~y.index.duplicated(keep="last")].sort_index()

    idx = df.index.intersection(y.index)
    df = df.reindex(idx)
    y = pd.to_numeric(y.reindex(idx), errors="coerce").astype(float)

    # shift features so row at t contains features from t-horizon
    X_all = df.shift(horizon).apply(pd.to_numeric, errors="coerce").astype(float)

    preds = pd.Series(index=idx, dtype=float)

    n = len(idx)
    if n < (train_days + 50):
        return preds, []

    test_start = max(0, n - test_days)

    for t in range(test_start, n):
        train_end = t
        train_start = max(0, train_end - train_days)

        X_train = X_all.iloc[train_start:train_end].values.astype(float)
        y_train = y.iloc[train_start:train_end].values.astype(float)
        X_pred = X_all.iloc[t].values.astype(float)

        # minimum training size check AFTER shift (critical)
        if np.isfinite(y_train).sum() < MIN_TRAIN_ROWS:
            preds.iloc[t] = np.nan
            continue

        # If prediction row is too empty, skip
        if np.isfinite(X_pred).sum() < 3:
            preds.iloc[t] = np.nan
            continue

        preds.iloc[t] = _ridge_fit_predict(X_train, y_train, X_pred, lam=RIDGE_L2)

    # meta checkpoints
    def checkpoint(label: str, days_back: int) -> Optional[Dict[str, Any]]:
        i = n - 1 - days_back
        if i < test_start or i < 0:
            return None
        d = idx[i]
        if not np.isfinite(preds.loc[d]):
            return None
        return {"period": label, "date": d}

    meta: List[Dict[str, Any]] = []
    ck = checkpoint("3 Months Ago (OOS)", 63)
    if ck: meta.append(ck)
    ck = checkpoint("6 Months Ago (OOS)", 126)
    if ck: meta.append(ck)
    ck = checkpoint("1 Year Ago (OOS)", 252)
    if ck: meta.append(ck)

    # first non-NaN inside OOS window
    ts = test_start
    while ts < n and not np.isfinite(preds.iloc[ts]):
        ts += 1
    if ts < n:
        meta.append({"period": "Test Start (OOS)", "date": idx[ts]})

    return preds, meta

# =========================================================
# app.py — PART 4 (CLEANED + FIXED)
# Continue inside analyze_stock() AFTER eps_ttm_daily has been built.
#
# FIXES applied (real issues in your snippet):
# 1) close_series was used BEFORE it was defined (in WACC block). Moved definition earlier.
# 2) safe_div in your app returns Optional[float] in PART 1, but PART 3 uses float NaN.
#    Here we treat safe_div as returning float (NaN allowed) and guard accordingly.
# 3) mkt_close.reindex(dates).values can break if mkt_close is not aligned/tz normalized.
#    Ensure tz-naive and reindex safely.
# 4) P_hat_realized = V_lag*(1+pred) produces NaNs at start; chart series now uses V_anchor*(1+pred_now)
#    only for “current fair value”, while history uses P_hat_realized but ffill/bfill.
# 5) rolling_target_multiple: quantiles with all-NaN window can throw warnings; we keep as-is but robustify.
# 6) DCF loop over every date can be slow; kept logic but reduced overhead and guarded.
# 7) method_flags keys always set (fcff/growth/wacc) for transparency.
# =========================================================

    # ---- Price series (DEFINE EARLY; needed by WACC and multiples) ----
    close_series = pd.Series(index=dates, data=stock_close.reindex(dates).values.astype(float), dtype=float)

    # ---- BVPS daily (equity / shares) ----
    if eq_q is not None and eq_q.dropna().size >= 1:
        eq_daily = last_value_on_or_before(eq_q, dates)
        bvps_daily = eq_daily / shares_daily.replace(0.0, np.nan)
        method_flags["bvps"] = "equity_over_shares_ttm_aligned"
    else:
        bookv = _to_float(info.get("bookValue"))  # usually per-share
        bvps_daily = pd.Series(index=dates, data=(bookv if bookv is not None else np.nan), dtype=float)
        method_flags["bvps"] = "fallback_info_bookValue"

    # ---- EBITDA TTM daily (best-effort) ----
    # Approx EBITDA = EBIT + D&A (TTM), else fall back to info["ebitda"]
    ebitda_ttm_daily = pd.Series(index=dates, dtype=float)
    if ebit_q is not None and ebit_q.dropna().size >= 4:
        ebit_ttm = ttm_from_quarters(ebit_q)
        ebit_ttm_daily = last_value_on_or_before(ebit_ttm, dates)

        if da_q is not None and da_q.dropna().size >= 4:
            da_ttm = ttm_from_quarters(da_q)
            da_ttm_daily = last_value_on_or_before(da_ttm, dates)
            ebitda_ttm_daily = ebit_ttm_daily + da_ttm_daily
            method_flags["ebitda"] = "ttm_ebit_plus_da"
        else:
            ebitda_ttm_daily = ebit_ttm_daily
            method_flags["ebitda"] = "ttm_ebit_proxy_no_da"
    else:
        ebitda_info = _to_float(info.get("ebitda"))  # trailing 12m
        ebitda_ttm_daily = pd.Series(index=dates, data=(ebitda_info if ebitda_info is not None else np.nan), dtype=float)
        method_flags["ebitda"] = "fallback_info_ebitda"

    # ---- FCFF TTM daily (CFO - CapEx), best-effort ----
    fcff_ttm_daily = pd.Series(index=dates, dtype=float)
    if cfo_q is not None and cfo_q.dropna().size >= 4 and capex_q is not None and capex_q.dropna().size >= 4:
        cfo_ttm = ttm_from_quarters(cfo_q)
        capex_ttm = ttm_from_quarters(capex_q)

        cfo_ttm_daily = last_value_on_or_before(cfo_ttm, dates)
        capex_ttm_daily = last_value_on_or_before(capex_ttm, dates)

        # normalize capex sign: cashflow often reports capex as negative outflow
        cap = capex_ttm_daily.values.astype(float)
        cap = np.where(np.isfinite(cap), cap, np.nan)
        cap_out = np.where(cap < 0, -cap, cap)

        fcff_ttm_daily = pd.Series(index=dates, data=(cfo_ttm_daily.values.astype(float) - cap_out), dtype=float)
        method_flags["fcff"] = "ttm_cfo_minus_capex"
    else:
        method_flags["fcff"] = "unavailable"

    # ---- Growth estimate (data-driven from FCFF YoY else EPS YoY else 0) ----
    growth_daily = pd.Series(index=dates, dtype=float)
    if fcff_ttm_daily.dropna().size > 400:
        g = (fcff_ttm_daily / fcff_ttm_daily.shift(TRADING_DAYS)) - 1.0
        growth_daily = g.clip(GROWTH_MIN, GROWTH_MAX)
        method_flags["growth"] = "fcff_yoy"
    elif eps_ttm_daily.dropna().size > 400:
        g = (eps_ttm_daily / eps_ttm_daily.shift(TRADING_DAYS)) - 1.0
        growth_daily = g.clip(GROWTH_MIN, GROWTH_MAX)
        method_flags["growth"] = "eps_yoy"
    else:
        growth_daily = pd.Series(index=dates, data=0.0, dtype=float)
        method_flags["growth"] = "fallback_zero_due_to_insufficient_history"

    # ---- WACC (data-minimal): cost of debt proxy = rf, weights from E vs D ----
    # NOTE: close_series MUST already exist (fixed above).
    D = net_debt_daily.clip(lower=0.0).astype(float)
    E = (shares_daily.replace(0.0, np.nan) * close_series).astype(float)

    Vcap = (D + E).replace(0.0, np.nan)
    wd = (D / Vcap).clip(0.0, 0.95)
    we = (E / Vcap).clip(0.05, 1.0)

    Rd = float(rf)  # proxy
    wacc_daily = (we * Re + wd * Rd * (1.0 - T)).clip(0.0, WACC_MAX)
    method_flags["wacc"] = "wacc_equity_capm_debt_rf_proxy"

    # ---- Build observed multiples series ----
    pe_obs = close_series / eps_ttm_daily.replace(0.0, np.nan)
    pb_obs = close_series / bvps_daily.replace(0.0, np.nan)

    ev_daily = (close_series * shares_daily.replace(0.0, np.nan)) + net_debt_daily
    ev_ebitda_obs = ev_daily / ebitda_ttm_daily.replace(0.0, np.nan)

    # ---- Self-anchored target multiples (rolling median of own history) ----
    def rolling_target_multiple(obs: pd.Series, window: int = TRADING_DAYS * 2) -> pd.Series:
        m = obs.replace([np.inf, -np.inf], np.nan).copy()
        m = m.where(m > 0)

        # rolling quantiles can be NaN-heavy early; that is OK.
        ql = m.rolling(window, min_periods=max(30, window // 6)).quantile(0.10)
        qh = m.rolling(window, min_periods=max(30, window // 6)).quantile(0.90)

        m_clip = m.clip(lower=ql, upper=qh)
        return m_clip.rolling(window, min_periods=max(30, window // 6)).median()

    pe_target = rolling_target_multiple(pe_obs)
    pb_target = rolling_target_multiple(pb_obs)
    ev_ebitda_target = rolling_target_multiple(ev_ebitda_obs)

    pe_model = pe_target * eps_ttm_daily
    pb_model = pb_target * bvps_daily
    ev_ebitda_model = (ev_ebitda_target * ebitda_ttm_daily - net_debt_daily) / shares_daily.replace(0.0, np.nan)

    # ---- DCF model daily (only if FCFF available) ----
    dcf_model = pd.Series(index=dates, dtype=float)
    market_long_run_g = float(rm_exp)  # derived earlier from market series

    if fcff_ttm_daily.dropna().size > 200:
        dcf_vals = np.full(shape=len(dates), fill_value=np.nan, dtype=float)

        # iterate by position (faster than .loc in a loop)
        fcff_arr = fcff_ttm_daily.values.astype(float)
        wacc_arr = wacc_daily.values.astype(float)
        g_arr = growth_daily.values.astype(float)
        sh_arr = shares_daily.values.astype(float)
        nd_arr = net_debt_daily.values.astype(float)

        for i, dt in enumerate(dates):
            fcff0 = fcff_arr[i]
            w = wacc_arr[i]
            g = g_arr[i]
            sh = sh_arr[i]
            nd = nd_arr[i] if np.isfinite(nd_arr[i]) else 0.0

            if not (np.isfinite(fcff0) and fcff0 > 0 and np.isfinite(w) and w > 0 and np.isfinite(g) and np.isfinite(sh) and sh > 0):
                continue

            try:
                dcf_ps = dcf_per_share_from_fcff(
                    fcff0=float(fcff0),
                    wacc=float(w),
                    g=float(np.clip(g, GROWTH_MIN, GROWTH_MAX)),
                    shares=float(sh),
                    net_debt=float(nd),
                    market_long_run_g=market_long_run_g,
                    years=FORECAST_YEARS,
                )
                dcf_vals[i] = float(dcf_ps) if np.isfinite(dcf_ps) else np.nan
            except Exception:
                continue

        dcf_model = pd.Series(index=dates, data=dcf_vals, dtype=float)

    # ---- Build valuation matrix X (models) ----
    models = {
        "dcf": dcf_model,
        "pe": pe_model,
        "pb": pb_model,
        "ev_ebitda": ev_ebitda_model,
    }

    X = np.vstack([
        models["dcf"].values.astype(float),
        models["pe"].values.astype(float),
        models["pb"].values.astype(float),
        models["ev_ebitda"].values.astype(float),
    ])

    avail = np.array([
        np.isfinite(models["dcf"]).sum() > 150,
        np.isfinite(models["pe"]).sum() > 150,
        np.isfinite(models["pb"]).sum() > 150,
        np.isfinite(models["ev_ebitda"]).sum() > 150,
    ], dtype=bool)

    if not np.any(avail):
        return JSONResponse({"error": "No valuation models available."}, status_code=200)

    # ---- Train valuation weights on past ----
    y_all = close_series.values.astype(float)
    n = len(dates)
    train_start_idx = max(0, n - TRAIN_WINDOW_DAYS)

    y_train = y_all[train_start_idx:]
    X_train = X[:, train_start_idx:]

    try:
        w_val = optimize_weights_dirichlet(y_train, X_train, avail, n_samples=N_WEIGHT_SAMPLES)
    except Exception:
        w_val = np.zeros(4, dtype=float)
        idxs = np.where(avail)[0]
        w_val[idxs] = 1.0 / len(idxs)

    V_anchor_arr = np.nansum((X.T * w_val), axis=1)
    V_anchor = pd.Series(index=dates, data=V_anchor_arr.astype(float), dtype=float)

    # ---- Spread engine ----
    # Build df_core with safe reindexing
    mkt_series = mkt_close.copy()
    try:
        mkt_series.index = pd.to_datetime(mkt_series.index).tz_localize(None)
    except Exception:
        pass
    mkt_series = mkt_series.sort_index()

    df_core = pd.DataFrame(index=dates)
    df_core["Close"] = close_series.astype(float)
    df_core["MktClose"] = pd.Series(index=dates, data=mkt_series.reindex(dates).values.astype(float), dtype=float)
    df_core["V_anchor"] = V_anchor.astype(float)

    # Optional volume
    try:
        if isinstance(hist, pd.DataFrame) and "Volume" in hist.columns:
            v = pd.to_numeric(hist["Volume"], errors="coerce")
            v.index = pd.to_datetime(v.index).tz_localize(None)
            v = v.reindex(dates).astype(float)
            if v.dropna().size > 50:
                df_core["Volume"] = v
    except Exception:
        pass

    feat = build_spread_features(df_core, shares_daily)

    delta_pct = (df_core["Close"] - df_core["V_anchor"]) / df_core["V_anchor"].replace(0.0, np.nan)
    y_target = delta_pct.astype(float)

    preds_delta, backtest_meta = walk_forward_spread_forecast(
        df_feat=feat,
        y_target=y_target,
        horizon=SPREAD_HORIZON_DAYS,
        train_days=TRAIN_WINDOW_DAYS,
        test_days=TEST_WINDOW_DAYS,
    )

    V_lag = V_anchor.shift(SPREAD_HORIZON_DAYS)
    P_hat_realized = V_lag * (1.0 + preds_delta)

    # Current predicted delta
    if preds_delta.dropna().size:
        delta_hat_now = float(preds_delta.dropna().iloc[-1])
    else:
        delta_hat_now = float(delta_pct.dropna().iloc[-1]) if delta_pct.dropna().size else 0.0

    V_now = float(V_anchor.iloc[-1]) if np.isfinite(V_anchor.iloc[-1]) else float(current_price)
    fair_value_1m = V_now * (1.0 + delta_hat_now)

    # ---- Backtest rows ----
    backtest = []
    for row in backtest_meta:
        d = row["date"]
        if d not in df_core.index:
            continue
        actual = float(df_core.loc[d, "Close"]) if np.isfinite(df_core.loc[d, "Close"]) else np.nan
        modelv = float(P_hat_realized.loc[d]) if d in P_hat_realized.index and np.isfinite(P_hat_realized.loc[d]) else np.nan
        if np.isfinite(actual) and np.isfinite(modelv) and actual > 0:
            backtest.append({"period": row["period"], "actual": actual, "model": modelv})

    # Chart fair values: best-effort fill
    fair_series_for_chart = P_hat_realized.reindex(dates).astype(float)
    fair_series_for_chart = fair_series_for_chart.ffill().bfill()
    fair_values_list = fair_series_for_chart.tolist()

    # ---- Returns ----
    def pct_return(series: pd.Series, days: int) -> Optional[float]:
        s = series.dropna()
        if s.size < days + 1:
            return None
        a = float(s.iloc[-1])
        b = float(s.iloc[-(days + 1)])
        if b <= 0:
            return None
        return (a / b - 1.0) * 100.0

    returns = {
        "1m": pct_return(close_series, 21),
        "3m": pct_return(close_series, 63),
        "6m": pct_return(close_series, 126),
        "1y": pct_return(close_series, 252),
        "2y": pct_return(close_series, 504),
    }

    # ---- Point-in-time metrics ----
    eps_now = float(eps_ttm_daily.iloc[-1]) if np.isfinite(eps_ttm_daily.iloc[-1]) else np.nan
    bvps_now = float(bvps_daily.iloc[-1]) if np.isfinite(bvps_daily.iloc[-1]) else np.nan

    pe_now = safe_div(float(current_price), float(eps_now)) if np.isfinite(eps_now) and eps_now != 0 else np.nan
    book_value_now = (bvps_now * float(shares_daily.iloc[-1])) if np.isfinite(bvps_now) else np.nan

    model_breakdown = {
        "dcf": float(dcf_model.iloc[-1]) if np.isfinite(dcf_model.iloc[-1]) else None,
        "pe_model": float(pe_model.iloc[-1]) if np.isfinite(pe_model.iloc[-1]) else None,
        "pb_model": float(pb_model.iloc[-1]) if np.isfinite(pb_model.iloc[-1]) else None,
        "ev_ebitda_model": float(ev_ebitda_model.iloc[-1]) if np.isfinite(ev_ebitda_model.iloc[-1]) else None,
    }

    # DCF projections (for UI)
    dcf_proj = []
    try:
        fcff0_now = float(fcff_ttm_daily.iloc[-1]) if np.isfinite(fcff_ttm_daily.iloc[-1]) else np.nan
        g0 = float(growth_daily.iloc[-1]) if np.isfinite(growth_daily.iloc[-1]) else 0.0
        if np.isfinite(fcff0_now) and fcff0_now > 0:
            for i in range(1, FORECAST_YEARS + 1):
                dcf_proj.append(float(fcff0_now * ((1.0 + g0) ** i)))
    except Exception:
        dcf_proj = []

    upside = safe_div((float(fair_value_1m) - float(current_price)), float(current_price)) * 100.0 if current_price > 0 else 0.0
    if np.isfinite(upside):
        if upside > 8:
            verdict = "Undervalued"
        elif upside < -8:
            verdict = "Overvalued"
        else:
            verdict = "Fairly Valued"
    else:
        verdict, upside = "Fairly Valued", 0.0

    response = {
        "valuation_summary": {
            "company_name": company_name,
            "sector": sector,
            "current_price": float(current_price),
            "fair_value": float(fair_value_1m) if np.isfinite(fair_value_1m) else None,
            "upside_percent": float(upside) if np.isfinite(upside) else 0.0,
            "verdict": verdict,
            "model_breakdown": model_breakdown,
            "dcf_projections": dcf_proj,
            "method_flags": method_flags,
        },
        "metrics": {
            "market_cap": float(mcap_now) if np.isfinite(mcap_now) else None,
            "pe_ratio": float(pe_now) if np.isfinite(pe_now) else None,
            "eps": float(eps_now) if np.isfinite(eps_now) else None,
            "beta": float(beta) if np.isfinite(beta) else None,
            "growth_rate": float(growth_daily.iloc[-1]) if np.isfinite(growth_daily.iloc[-1]) else None,
            "book_value": float(book_value_now) if np.isfinite(book_value_now) else None,
            "wacc": float(wacc_daily.iloc[-1]) if np.isfinite(wacc_daily.iloc[-1]) else None,
        },
        "returns": returns,
        "optimized_weights": {
            "dcf": float(w_val[0]),
            "pe": float(w_val[1]),
            "pb": float(w_val[2]),
            "ev_ebitda": float(w_val[3]),
        },
        "backtest": backtest,
        "historical_data": {
            "dates": dates_ms,
            "prices": prices_list,
            "fair_values": fair_values_list,
        },
    }

    return JSONResponse(json_safe(response), status_code=200)
# =========================================================
# app.py — PART 5 (CLEANED + FIXED)
# Missing pieces referenced by PART 4:
#   - tz normalization (fix tz-aware vs tz-naive)
#   - safe_div
#   - spread-feature builder
#   - walk-forward spread forecaster (pure numpy ridge)
#   - __main__ runner
#
# FIXES applied (real issues):
# 1) _to_naive_datetime_index: tz_convert(None) can fail on non-UTC; use tz_localize(None) safely.
# 2) _standardize_train_apply: returned (mu, sd) but actually returned mu only; fixed to return (mu, sd).
# 3) _logret: log(0) -> -inf; now replaces nonpositive with NaN before log.
# 4) build_spread_features: log(close) and log(mkt) must guard <=0; now safe.
# 5) _ridge_fit_predict: previously filtered only by finite y; X could be all-NaN -> becomes 0 after std,
#    which is OK, but we ensure shapes and handle empty train.
# 6) walk_forward_spread_forecast: MIN_TRAIN_ROWS should be checked AFTER shifting (many early NaNs);
#    also require enough finite rows in X_train after shift. Added checks.
# 7) Constant definitions kept here; ensure they do not conflict with earlier parts (use one source of truth).
# =========================================================

SPREAD_HORIZON_DAYS = 21  # ~1M
RIDGE_L2 = 2.0            # small shrinkage for stability
MIN_TRAIN_ROWS = 220      # require ~1y-ish realized rows for y_train
MIN_FINITE_X_ROWS = 160   # require enough usable feature rows after shift

def safe_div(a: float, b: float) -> float:
    try:
        if b is None or a is None:
            return float("nan")
        a = float(a)
        b = float(b)
        if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
            return float("nan")
        return a / b
    except Exception:
        return float("nan")

def _to_naive_datetime_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Fix: TypeError: Cannot compare tz-naive and tz-aware timestamps
    Standardize everything to tz-naive timestamps (timezone removed).
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)

    # If tz-aware -> strip tz safely
    try:
        if idx.tz is not None:
            return idx.tz_localize(None)
    except Exception:
        pass

    # Robust fallback: try localize(None) if possible
    try:
        return idx.tz_localize(None)
    except Exception:
        return idx

def _normalize_prices_df(hist: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns tz-aware indices. Force tz-naive and sorted unique index.
    """
    if hist is None or hist.empty:
        return hist
    out = hist.copy()
    out.index = _to_naive_datetime_index(pd.to_datetime(out.index))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out

def _normalize_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    out = s.copy()
    out.index = _to_naive_datetime_index(pd.to_datetime(out.index))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out

def _logret(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x.where(x > 0.0)  # avoid log(0) and log(negative)
    return np.log(x).diff()

def _zscore(s: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (s - mu) / sd

def build_spread_features(df_core: pd.DataFrame, shares_daily: pd.Series) -> pd.DataFrame:
    """
    Features capture:
      - valuation anchor vs price (spread + mean reversion)
      - liquidity proxies (volume, dollar volume, turnover)
      - market regime (market momentum/vol) + corr to market
      - momentum / reversals
      - volatility clustering
    """
    df = df_core.copy()

    # Ensure tz-naive index
    df.index = _to_naive_datetime_index(pd.to_datetime(df.index))
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for c in ["Close", "MktClose", "V_anchor"]:
        if c not in df.columns:
            raise ValueError(f"build_spread_features missing required column: {c}")

    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    mkt = pd.to_numeric(df["MktClose"], errors="coerce").astype(float)
    V = pd.to_numeric(df["V_anchor"], errors="coerce").astype(float)

    close_pos = close.where(close > 0.0)
    mkt_pos = mkt.where(mkt > 0.0)

    # Spread level and change
    spread = (close - V) / V.replace(0.0, np.nan)
    spread_chg_5 = spread.diff(5)
    spread_chg_21 = spread.diff(21)

    # Returns (log)
    r1 = _logret(close_pos)
    rm1 = _logret(mkt_pos)

    # Momentum (log price change windows)
    mom_5 = np.log(close_pos).diff(5)
    mom_21 = np.log(close_pos).diff(21)
    mom_63 = np.log(close_pos).diff(63)

    # Realized volatility (not annualized; regression does not need absolute scaling)
    vol_21 = r1.rolling(21).std(ddof=0)
    vol_63 = r1.rolling(63).std(ddof=0)

    # Market regime proxies
    mkt_mom_21 = np.log(mkt_pos).diff(21)
    mkt_vol_63 = rm1.rolling(63).std(ddof=0)

    # Correlation proxy
    corr_63 = r1.rolling(63).corr(rm1)

    # Liquidity proxies
    vol_feat = pd.Series(index=df.index, dtype=float)
    dollar_vol = pd.Series(index=df.index, dtype=float)
    turnover = pd.Series(index=df.index, dtype=float)

    if "Volume" in df.columns and pd.to_numeric(df["Volume"], errors="coerce").notna().any():
        vol_raw = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
        vol_raw = vol_raw.where(vol_raw >= 0.0)

        vol_feat = np.log1p(vol_raw)
        dollar_vol = np.log1p(vol_raw * close_pos)

        sh = pd.to_numeric(shares_daily.reindex(df.index), errors="coerce").astype(float)
        turnover = vol_raw / sh.replace(0.0, np.nan)

    # Z-scored features (stability)
    spread_z_252 = _zscore(spread, 252)
    mom_z_63 = _zscore(mom_21, 63)
    vol_z_252 = _zscore(vol_21, 252)
    mkt_mom_z_252 = _zscore(mkt_mom_21, 252)

    out = pd.DataFrame(index=df.index)
    out["spread"] = spread
    out["spread_chg_5"] = spread_chg_5
    out["spread_chg_21"] = spread_chg_21

    out["r1"] = r1
    out["mom_5"] = mom_5
    out["mom_21"] = mom_21
    out["mom_63"] = mom_63

    out["vol_21"] = vol_21
    out["vol_63"] = vol_63

    out["rm1"] = rm1
    out["mkt_mom_21"] = mkt_mom_21
    out["mkt_vol_63"] = mkt_vol_63
    out["corr_63"] = corr_63

    out["log_vol"] = vol_feat
    out["log_dollar_vol"] = dollar_vol
    out["turnover"] = turnover

    out["spread_z_252"] = spread_z_252
    out["mom_z_63"] = mom_z_63
    out["vol_z_252"] = vol_z_252
    out["mkt_mom_z_252"] = mkt_mom_z_252

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def _standardize_train_apply(X_train: np.ndarray, X_apply: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Standardize with train mean/std (ddof=0).
    Returns: X_train_std, X_apply_std, (mu, sd)
    """
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)

    Xtr = (X_train - mu) / sd
    Xap = (X_apply - mu) / sd

    # zero-impute remaining NaNs after standardization (ridge is fine with this)
    Xtr = np.where(np.isfinite(Xtr), Xtr, 0.0)
    Xap = np.where(np.isfinite(Xap), Xap, 0.0)

    return Xtr, Xap, (mu, sd)

def _ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, l2: float = RIDGE_L2) -> float:
    """
    Ridge regression with intercept (added column of ones).
    """
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D")
    if X_pred.ndim != 1:
        X_pred = X_pred.reshape(-1)

    y_train = y_train.astype(float)

    # Finite y rows
    m = np.isfinite(y_train)
    if m.sum() < 30:
        return float("nan")

    Xt = X_train[m]
    yt = y_train[m]

    # Add intercept
    ones = np.ones((Xt.shape[0], 1), dtype=float)
    Xb = np.hstack([ones, Xt])

    # Regularize all except intercept
    p = Xb.shape[1]
    I = np.eye(p, dtype=float)
    I[0, 0] = 0.0

    A = Xb.T @ Xb + l2 * I
    b = Xb.T @ yt

    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]

    # Predict
    Xp = X_pred.reshape(1, -1)
    Xp = np.hstack([np.ones((1, 1), dtype=float), Xp])
    return float(Xp @ w)

def walk_forward_spread_forecast(
    df_feat: pd.DataFrame,
    y_target: pd.Series,
    horizon: int,
    train_days: int,
    test_days: int,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Predict y(t) using features(t-horizon). We do that by shifting features by +horizon.
    Walk-forward OOS: last `test_days` are OOS; each OOS point fits on a rolling train window.
    """
    df = df_feat.copy()
    df.index = _to_naive_datetime_index(pd.to_datetime(df.index))
    df = df[~df.index.duplicated(keep="last")].sort_index()

    y = y_target.copy()
    y.index = _to_naive_datetime_index(pd.to_datetime(y.index))
    y = y[~y.index.duplicated(keep="last")].sort_index()

    # Align to common index
    idx = df.index.intersection(y.index)
    df = df.reindex(idx)
    y = y.reindex(idx).astype(float)

    # Shift features so row at t uses info from t-horizon
    X_df = df.shift(horizon)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").astype(float)

    preds = pd.Series(index=idx, dtype=float)

    n = len(idx)
    if n < (train_days + 50):
        return preds, []

    oos_start = max(0, n - test_days)
    meta: List[Dict[str, Any]] = []

    for i in range(oos_start, n):
        d = idx[i]

        train_end = i
        train_start = max(0, train_end - train_days)

        X_train_df = X_df.iloc[train_start:train_end]
        y_train_s = y.iloc[train_start:train_end]
        x_row = X_df.iloc[i]

        # Require enough y rows
        if y_train_s.dropna().size < MIN_TRAIN_ROWS:
            preds.loc[d] = np.nan
            continue

        # Require enough usable X rows after shift
        finite_rows = np.isfinite(X_train_df.values.astype(float)).all(axis=1)
        if finite_rows.sum() < MIN_FINITE_X_ROWS:
            preds.loc[d] = np.nan
            continue

        # Require some finite features in current row
        if np.isfinite(x_row.values.astype(float)).sum() < 3:
            preds.loc[d] = np.nan
            continue

        Xtr = X_train_df.values.astype(float)
        ytr = y_train_s.values.astype(float)
        xap = x_row.values.astype(float).reshape(1, -1)

        Xtr_std, xap_std, _ = _standardize_train_apply(Xtr, xap)

        yhat = _ridge_fit_predict(Xtr_std, ytr, xap_std.flatten(), l2=float(RIDGE_L2))
        preds.loc[d] = yhat

    # UI meta labels
    def _pick(days_back: int, label: str):
        j = n - 1 - days_back
        if j < oos_start or j < 0 or j >= n:
            return
        meta.append({"date": idx[j], "period": f"{label} (OOS)"})

    _pick(63, "3 Months Ago")
    _pick(126, "6 Months Ago")
    _pick(252, "1 Year Ago")
    meta.append({"date": idx[oos_start], "period": "Test Start (OOS)"})

    return preds, meta

# =========================================================
# OPTIONAL: local runner
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

