# =========================================================
# app.py — PART 1
# Foundation, configuration, utilities
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# 3) BASIC HELPERS
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
        if b == 0 or not np.isfinite(a) or not np.isfinite(b):
            return None
        return a / b
    except Exception:
        return None


def winsorize(arr: np.ndarray, p_low=0.05, p_high=0.95) -> np.ndarray:
    arr = arr.astype(float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return arr

    lo = np.quantile(arr[mask], p_low)
    hi = np.quantile(arr[mask], p_high)
    return np.clip(arr, lo, hi)


def last_value_on_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill a low-frequency series (quarterly/annual)
    to a daily index based on last known report date.
    """
    if series is None or series.empty:
        return pd.Series(index=dates, dtype=float)

    s = series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    dates = pd.to_datetime(dates).tz_localize(None)

    tmp = s.reindex(s.index.union(dates)).sort_index().ffill()
    return tmp.reindex(dates)


def ttm_from_quarters(q_series: pd.Series) -> pd.Series:
    """
    Compute trailing-12-month series from quarterly data.
    """
    s = q_series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index().rolling(4, min_periods=4).sum()

# =========================================================
# app.py — PART 2
# Data fetching (prices + statements + risk-free) + market/beta helpers
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

    # ---------- Prices ----------
    def fetch_prices_yahoo(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=False)

        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError(f"No Yahoo price history for {ticker}.")

        # Normalize index tz (fix tz-naive vs tz-aware comparisons downstream)
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist.sort_index()

        # Keep only what we use
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
        hist = hist[cols]

        if hist["Close"].dropna().empty:
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
        if "values" not in data or not isinstance(data["values"], list) or len(data["values"]) == 0:
            msg = data.get("message") or "No Twelve Data values."
            raise ValueError(f"Twelve Data: {msg}")

        rows = []
        for v in data["values"]:
            try:
                dt = pd.to_datetime(v["datetime"]).tz_localize(None)
                close = float(v["close"])
                rows.append((dt, close))
            except Exception:
                continue

        if not rows:
            raise ValueError("Twelve Data: could not parse values.")

        df = pd.DataFrame(rows, columns=["Date", "Close"]).set_index("Date").sort_index()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            raise ValueError("Twelve Data: empty parsed dataframe.")
        return df[["Close"]]

    def fetch_prices_alpha_vantage(self, ticker: str) -> pd.DataFrame:
        # Alpha Vantage uses ".SA" convention for Saudi listings in some mappings
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
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_index().tail(1250)

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

        # Normalize statement column indices into tz-naive datetimes when possible
        def norm_cols(df: Any) -> Any:
            if not isinstance(df, pd.DataFrame) or df is None or df.empty:
                return df
            out = df.copy()
            try:
                out.columns = pd.to_datetime(out.columns).tz_localize(None)
            except Exception:
                # leave as-is if cannot parse
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
    r = _row_lookup(df, row_names)
    if r is None and contains is not None:
        r = _row_contains(df, contains)
    if r is None:
        return None

    s = pd.to_numeric(r, errors="coerce")
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception:
        # If statement columns aren't parseable datetimes, bail
        return None

    return s.sort_index()


# =========================================================
# Market / beta helpers
# =========================================================
def annualized_geo_mean_return(prices: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    prices = prices.dropna()
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
    df = pd.DataFrame({"s": stock_prices, "m": market_prices}).dropna()
    if len(df) < 120:
        raise ValueError("Not enough overlapping history for beta.")

    rs = np.log(df["s"]).diff().dropna()
    rm = np.log(df["m"]).diff().dropna()

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
# Request model (kept here so endpoint can be added later)
# =========================================================
class StockRequest(BaseModel):
    ticker: str
# =========================================================
# app.py — PART 3
# Helpers (time-alignment + robustness) + Valuation models + Spread engine (features + walk-forward)
# =========================================================

# ---------- Small numeric helpers ----------
def safe_div(a: float, b: float) -> float:
    try:
        if b is None or b == 0 or not np.isfinite(b):
            return np.nan
        if a is None or not np.isfinite(a):
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


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


# ---------- Time-aligned fundamentals helpers ----------
def ttm_from_quarters(q_series: pd.Series) -> pd.Series:
    # q_series indexed by report date; compute rolling sum of last 4 quarters
    s = q_series.sort_index()
    return s.rolling(4, min_periods=4).sum()


def last_value_on_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    # Forward-fill to daily dates based on last known report date.
    if series is None or series.empty:
        return pd.Series(index=dates, dtype=float)

    s = series.sort_index().copy()
    # Normalize timezone to avoid tz-aware vs tz-naive issues
    try:
        s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception:
        pass

    d = pd.to_datetime(dates)
    try:
        d = d.tz_localize(None)
    except Exception:
        pass

    tmp = s.reindex(s.index.union(d)).sort_index().ffill()
    out = tmp.reindex(d)
    out.index = dates  # preserve original index object
    return out.astype(float)


def winsorize(arr: np.ndarray, p_low: float = 0.05, p_high: float = 0.95) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return x
    lo = np.quantile(x[ok], p_low)
    hi = np.quantile(x[ok], p_high)
    return np.clip(x, lo, hi)


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
    FCFF-based DCF -> Equity value -> per share.
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

    g_term = min(g, market_long_run_g)
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
    n_models = X.shape[0]
    if not np.any(avail):
        raise ValueError("No models available")

    idx = np.where(avail)[0]
    k = len(idx)

    best_w_full = np.zeros(n_models, dtype=float)
    best_loss = float("inf")

    rnd = np.random.default_rng(42)

    candidates: List[np.ndarray] = []
    for j in idx:
        w = np.zeros(n_models, dtype=float)
        w[j] = 1.0
        candidates.append(w)

    draws = rnd.dirichlet(np.ones(k, dtype=float), size=n_samples)
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

    best_w_full = best_w_full / best_w_full.sum()
    return best_w_full


# =========================================================
# Spread engine: predict short-horizon valuation spread dynamics (not pure price)
# Professional-grade: stable features + ridge regression + walk-forward evaluation
# =========================================================
SPREAD_HORIZON_DAYS = 21  # ~1 month trading days


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (s - mu) / sd


def _logret(px: pd.Series) -> pd.Series:
    return np.log(px.replace(0.0, np.nan)).diff()


def _rolling_vol(px: pd.Series, window: int) -> pd.Series:
    r = _logret(px)
    return r.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def build_spread_features(df_core: pd.DataFrame, shares_daily: pd.Series) -> pd.DataFrame:
    """
    Inputs expected columns:
      - Close (stock)
      - MktClose (market)
      - V_anchor (valuation anchor)
      - Volume (optional)
    Returns feature dataframe aligned to df_core.index.
    """
    df = df_core.copy()

    # Basic sanity
    for c in ["Close", "MktClose", "V_anchor"]:
        if c not in df.columns:
            raise ValueError(f"build_spread_features missing required column: {c}")

    close = df["Close"].astype(float)
    mkt = df["MktClose"].astype(float)
    V = df["V_anchor"].astype(float)

    # Core spread measures
    spread = (close - V) / V.replace(0.0, np.nan)  # signed % deviation
    abs_spread = spread.abs()

    # Returns / momentum
    r1 = close.pct_change(1)
    r5 = close.pct_change(5)
    r21 = close.pct_change(21)
    mkt_r5 = mkt.pct_change(5)
    mkt_r21 = mkt.pct_change(21)

    # Volatility regime
    vol21 = _rolling_vol(close, 21)
    vol63 = _rolling_vol(close, 63)
    mkt_vol63 = _rolling_vol(mkt, 63)

    # Liquidity proxy
    if "Volume" in df.columns and df["Volume"].notna().any():
        volu = pd.to_numeric(df["Volume"], errors="coerce")
        dollar_vol = (volu * close).replace([np.inf, -np.inf], np.nan)
        liq_z = _zscore(np.log1p(dollar_vol), 63)
    else:
        liq_z = pd.Series(index=df.index, dtype=float)

    # Market cap drift proxy (if shares move; else mostly constant)
    sh = shares_daily.reindex(df.index).astype(float)
    mcap = (sh * close).replace([np.inf, -np.inf], np.nan)
    mcap_z = _zscore(np.log1p(mcap), 252)

    # Mean-reversion structure (spread tends to revert; include lagged spread + change)
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

    # Clean extreme outliers (robustify)
    for c in feat.columns:
        feat[c] = feat[c].replace([np.inf, -np.inf], np.nan)
        # winsorize by global quantiles to avoid absurd spikes from data glitches
        x = feat[c].values.astype(float)
        feat[c] = pd.Series(winsorize(x, 0.01, 0.99), index=feat.index)

    return feat


def _ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    lam: float = 10.0,
) -> float:
    """
    Closed-form ridge with standardization.
    Predict scalar for one X_pred row.
    """
    # Drop rows with NaNs
    ok = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
    Xt = X_train[ok]
    yt = y_train[ok]
    if yt.size < 80:
        return np.nan

    # Standardize
    mu = Xt.mean(axis=0)
    sd = Xt.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (Xt - mu) / sd

    # Ridge closed form: (X'X + lam I)^{-1} X'y
    p = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(p)
    b = Xs.T @ yt
    try:
        w = np.linalg.solve(A, b)
    except Exception:
        return np.nan

    xp = (X_pred - mu) / sd
    return float(xp @ w)


def walk_forward_spread_forecast(
    df_feat: pd.DataFrame,
    y_target: pd.Series,
    horizon: int = SPREAD_HORIZON_DAYS,
    train_days: int = TRAIN_WINDOW_DAYS,
    test_days: int = TEST_WINDOW_DAYS,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Predict delta-spread at realized date t using features at (t-horizon).
    Returns:
      - preds: Series indexed by realized dates (same index as df_feat)
      - meta: list of backtest checkpoints (3M/6M/1Y/TestStart style)
    """
    idx = df_feat.index

    # Align: features used from t-h; target realized at t
    X_all = df_feat.shift(horizon)
    y_all = y_target.astype(float)

    preds = pd.Series(index=idx, dtype=float)

    n = len(idx)
    test_start = max(0, n - test_days)

    # Walk forward over the test window only (professional-grade OOS)
    for t in range(test_start, n):
        train_end = t  # up to t-1 (since X is lagged, using info strictly from <= t-h)
        train_start = max(0, train_end - train_days)

        X_train = X_all.iloc[train_start:train_end].values.astype(float)
        y_train = y_all.iloc[train_start:train_end].values.astype(float)
        X_pred = X_all.iloc[t].values.astype(float)

        pred = _ridge_fit_predict(X_train, y_train, X_pred, lam=10.0)
        preds.iloc[t] = pred

    # Backtest checkpoints (based on realized-date index)
    def checkpoint(label: str, days_back: int) -> Optional[Dict[str, Any]]:
        i = n - 1 - days_back
        if i < 0:
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

    # Test start (first OOS point with a prediction)
    # Find first non-NaN in preds within test region
    ts_idx = test_start
    while ts_idx < n and not np.isfinite(preds.iloc[ts_idx]):
        ts_idx += 1
    if ts_idx < n:
        meta.append({"period": "Test Start (OOS)", "date": idx[ts_idx]})

    return preds, meta
# =========================================================
# app.py — PART 4
# Continue inside analyze_stock() AFTER eps_ttm_daily has been built.
# This part completes: BVPS, EBITDA, FCFF, growth, WACC, multiples models,
# valuation anchor, spread forecast, chart series, backtest rows, and response.
# =========================================================

    # ---- BVPS daily (equity / shares) ----
    if eq_q is not None and eq_q.dropna().size >= 1:
        eq_daily = last_value_on_or_before(eq_q, dates)
        bvps_daily = eq_daily / shares_daily.replace(0.0, np.nan)
    else:
        # fallback: info bookValue is usually per-share
        bookv = _to_float(info.get("bookValue"))
        bvps_daily = pd.Series(index=dates, data=(bookv if bookv is not None else np.nan), dtype=float)

    # ---- EBITDA TTM daily (best-effort) ----
    # If not present directly, approximate EBITDA = EBIT + D&A (TTM)
    ebitda_ttm_daily = pd.Series(index=dates, dtype=float)
    if ebit_q is not None and ebit_q.dropna().size >= 4:
        ebit_ttm = ttm_from_quarters(ebit_q)
        ebit_ttm_daily = last_value_on_or_before(ebit_ttm, dates)
        if da_q is not None and da_q.dropna().size >= 4:
            da_ttm = ttm_from_quarters(da_q)
            da_ttm_daily = last_value_on_or_before(da_ttm, dates)
            ebitda_ttm_daily = ebit_ttm_daily + da_ttm_daily
        else:
            ebitda_ttm_daily = ebit_ttm_daily
    else:
        # fallback: info ebitda (usually trailing 12m) constant
        ebitda_info = _to_float(info.get("ebitda"))
        ebitda_ttm_daily = pd.Series(index=dates, data=(ebitda_info if ebitda_info is not None else np.nan), dtype=float)

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
    D = net_debt_daily.clip(lower=0.0)
    E = (shares_daily.replace(0.0, np.nan) * close_series).astype(float)
    Vcap = (D + E).replace(0.0, np.nan)
    wd = (D / Vcap).clip(0.0, 0.95)
    we = (E / Vcap).clip(0.05, 1.0)
    Rd = rf  
    wacc_daily = (we * Re + wd * Rd * (1.0 - T)).clip(0.0, WACC_MAX)
    method_flags["wacc"] = "wacc_equity_capm_debt_rf_proxy"

    # ---- Build observed multiples series ----
    close_series = pd.Series(index=dates, data=stock_close.values.astype(float), dtype=float)
    pe_obs = close_series / eps_ttm_daily.replace(0.0, np.nan)
    pb_obs = close_series / bvps_daily.replace(0.0, np.nan)

    ev_daily = (close_series * shares_daily.replace(0.0, np.nan)) + net_debt_daily
    ev_ebitda_obs = ev_daily / ebitda_ttm_daily.replace(0.0, np.nan)

    # ---- Self-anchored target multiples (rolling median of own history) ----
    def rolling_target_multiple(obs: pd.Series, window: int = TRADING_DAYS * 2) -> pd.Series:
        m = obs.replace([np.inf, -np.inf], np.nan).copy()
        m = m.where(m > 0)
        ql = m.rolling(window).quantile(0.10)
        qh = m.rolling(window).quantile(0.90)
        m_clip = m.clip(lower=ql, upper=qh)
        return m_clip.rolling(window).median()

    pe_target = rolling_target_multiple(pe_obs)
    pb_target = rolling_target_multiple(pb_obs)
    ev_ebitda_target = rolling_target_multiple(ev_ebitda_obs)

    pe_model = pe_target * eps_ttm_daily
    pb_model = pb_target * bvps_daily
    ev_ebitda_model = (ev_ebitda_target * ebitda_ttm_daily - net_debt_daily) / shares_daily.replace(0.0, np.nan)

    # ---- DCF model daily (only if FCFF available) ----
    dcf_model = pd.Series(index=dates, dtype=float)
    market_long_run_g = rm_exp

    if fcff_ttm_daily.dropna().size > 200:
        dcf_vals = []
        for dt in dates:
            fcff0 = float(fcff_ttm_daily.loc[dt]) if np.isfinite(fcff_ttm_daily.loc[dt]) else np.nan
            w = float(wacc_daily.loc[dt]) if np.isfinite(wacc_daily.loc[dt]) else np.nan
            g = float(growth_daily.loc[dt]) if np.isfinite(growth_daily.loc[dt]) else np.nan
            sh = float(shares_daily.loc[dt]) if np.isfinite(shares_daily.loc[dt]) else np.nan
            nd = float(net_debt_daily.loc[dt]) if np.isfinite(net_debt_daily.loc[dt]) else 0.0
            try:
                if np.isfinite(fcff0) and fcff0 > 0 and np.isfinite(w) and w > 0 and np.isfinite(g) and np.isfinite(sh) and sh > 0:
                    dcf_ps = dcf_per_share_from_fcff(
                        fcff0=fcff0,
                        wacc=w,
                        g=float(np.clip(g, GROWTH_MIN, GROWTH_MAX)),
                        shares=sh,
                        net_debt=nd,
                        market_long_run_g=market_long_run_g,
                        years=FORECAST_YEARS,
                    )
                    dcf_vals.append(float(dcf_ps) if np.isfinite(dcf_ps) else np.nan)
                else:
                    dcf_vals.append(np.nan)
            except Exception:
                dcf_vals.append(np.nan)
        dcf_model = pd.Series(index=dates, data=np.array(dcf_vals, dtype=float), dtype=float)

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
        return JSONResponse(
            {"error": "No valuation models available."},
            status_code=200
        )

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

    V_anchor = np.nansum((X.T * w_val), axis=1)
    V_anchor = pd.Series(index=dates, data=V_anchor.astype(float), dtype=float)

    # ---- Spread engine ----
    df_core = pd.DataFrame(index=dates)
    df_core["Close"] = close_series.astype(float)
    df_core["MktClose"] = pd.Series(index=dates, data=mkt_close.reindex(dates).values.astype(float), dtype=float)
    df_core["V_anchor"] = V_anchor.astype(float)

    vol_series = None
    try:
        if isinstance(hist, pd.DataFrame) and "Volume" in hist.columns:
            v = pd.to_numeric(hist["Volume"], errors="coerce").reindex(dates).astype(float)
            if v.dropna().size > 50:
                vol_series = v
    except Exception:
        vol_series = None

    if vol_series is not None:
        df_core["Volume"] = vol_series

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
        actual = float(df_core.loc[d, "Close"]) if d in df_core.index else np.nan
        modelv = float(P_hat_realized.loc[d]) if d in P_hat_realized.index else np.nan
        if np.isfinite(actual) and np.isfinite(modelv) and actual > 0:
            backtest.append({"period": row["period"], "actual": actual, "model": modelv})

    fair_series_for_chart = P_hat_realized.reindex(dates).astype(float).ffill().bfill()
    fair_values_list = fair_series_for_chart.tolist()

    # ---- Returns ----
    def pct_return(series: pd.Series, days: int) -> Optional[float]:
        s = series.dropna()
        if s.size < days + 1:
            return None
        a, b = float(s.iloc[-1]), float(s.iloc[-(days + 1)])
        return (a / b - 1.0) * 100.0 if b > 0 else None

    returns = {
        "1m": pct_return(close_series, 21),
        "3m": pct_return(close_series, 63),
        "6m": pct_return(close_series, 126),
        "1y": pct_return(close_series, 252),
        "2y": pct_return(close_series, 504),
    }

    eps_now = float(eps_ttm_daily.iloc[-1]) if np.isfinite(eps_ttm_daily.iloc[-1]) else np.nan
    bvps_now = float(bvps_daily.iloc[-1]) if np.isfinite(bvps_daily.iloc[-1]) else np.nan
    pe_now = safe_div(current_price, eps_now) if np.isfinite(eps_now) and eps_now != 0 else np.nan
    book_value_now = (bvps_now * float(shares_daily.iloc[-1])) if np.isfinite(bvps_now) else np.nan

    model_breakdown = {
        "dcf": float(dcf_model.iloc[-1]) if np.isfinite(dcf_model.iloc[-1]) else None,
        "pe_model": float(pe_model.iloc[-1]) if np.isfinite(pe_model.iloc[-1]) else None,
        "pb_model": float(pb_model.iloc[-1]) if np.isfinite(pb_model.iloc[-1]) else None,
        "ev_ebitda_model": float(ev_ebitda_model.iloc[-1]) if np.isfinite(ev_ebitda_model.iloc[-1]) else None,
    }

    dcf_proj = []
    try:
        fcff0 = float(fcff_ttm_daily.iloc[-1]) if np.isfinite(fcff_ttm_daily.iloc[-1]) else np.nan
        g0 = float(growth_daily.iloc[-1]) if np.isfinite(growth_daily.iloc[-1]) else 0.0
        if np.isfinite(fcff0) and fcff0 > 0:
            for i in range(1, FORECAST_YEARS + 1):
                dcf_proj.append(float(fcff0 * ((1.0 + g0) ** i)))
    except Exception:
        dcf_proj = []

    upside = safe_div((fair_value_1m - current_price), current_price) * 100.0 if current_price > 0 else 0.0
    if np.isfinite(upside):
        if upside > 8: verdict = "Undervalued"
        elif upside < -8: verdict = "Overvalued"
        else: verdict = "Fairly Valued"
    else:
        verdict, upside = "Fairly Valued", 0.0

    response = {
        "valuation_summary": {
            "company_name": company_name,
            "sector": sector,
            "current_price": current_price,
            "fair_value": float(fair_value_1m) if np.isfinite(fair_value_1m) else None,
            "upside_percent": float(upside),
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
            "dcf": float(w_val[0]), "pe": float(w_val[1]),
            "pb": float(w_val[2]), "ev_ebitda": float(w_val[3]),
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
# app.py — PART 5
# Missing pieces referenced by PART 4:
#   - tz normalization (fix tz-aware vs tz-naive)
#   - safe_div
#   - spread-feature builder
#   - walk-forward spread forecaster (no sklearn; pure numpy ridge)
#   - (optional) __main__ runner for local dev
# =========================================================

SPREAD_HORIZON_DAYS = 21  # ~1M
RIDGE_L2 = 2.0            # small shrinkage for stability
MIN_TRAIN_ROWS = 220      # require ~1y-ish rows after shifting

def safe_div(a: float, b: float) -> float:
    try:
        if b is None:
            return float("nan")
        b = float(b)
        a = float(a)
        if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
            return float("nan")
        return a / b
    except Exception:
        return float("nan")

def _to_naive_datetime_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Fix: TypeError: Cannot compare tz-naive and tz-aware timestamps
    We standardize everything to tz-naive UTC-like timestamps (timezone removed).
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if idx.tz is not None:
        return idx.tz_convert(None)
    # Some inputs have mixed tz-awareness; robust fallback:
    try:
        # if any element is tz-aware, normalize via to_datetime then tz_localize(None)
        if hasattr(idx, "tz_localize"):
            return idx.tz_localize(None)
    except Exception:
        pass
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
    return np.log(x).diff()

def _zscore(s: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)

def build_spread_features(df_core: pd.DataFrame, shares_daily: pd.Series) -> pd.DataFrame:
    """
    Features to capture:
      - valuation anchor vs price (spread + mean reversion)
      - liquidity/flow proxies (volume, dollar volume, turnover)
      - market regime + beta-ish exposure (market returns, vol)
      - momentum / reversals
      - volatility clustering
    Everything stays “data-derived” from available series.
    """
    df = df_core.copy()

    # Ensure tz-naive index
    df.index = _to_naive_datetime_index(pd.to_datetime(df.index))
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Base series
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    mkt = pd.to_numeric(df["MktClose"], errors="coerce").astype(float)
    V = pd.to_numeric(df["V_anchor"], errors="coerce").astype(float)

    # Spread level and change
    spread = (close - V) / V.replace(0.0, np.nan)
    spread_chg_5 = spread.diff(5)
    spread_chg_21 = spread.diff(21)

    # Returns
    r1 = _logret(close)
    rm1 = _logret(mkt)

    # Momentum (log price change windows)
    mom_5 = np.log(close).diff(5)
    mom_21 = np.log(close).diff(21)
    mom_63 = np.log(close).diff(63)

    # Realized volatility (annualized-ish, but scale not critical for regression)
    vol_21 = r1.rolling(21).std(ddof=0)
    vol_63 = r1.rolling(63).std(ddof=0)

    # Market regime / risk-on proxy
    mkt_mom_21 = np.log(mkt).diff(21)
    mkt_vol_63 = rm1.rolling(63).std(ddof=0)

    # Correlation proxy (rolling corr to market)
    corr_63 = r1.rolling(63).corr(rm1)

    # Liquidity proxies (if volume present)
    vol_feat = pd.Series(index=df.index, dtype=float)
    dollar_vol = pd.Series(index=df.index, dtype=float)
    turnover = pd.Series(index=df.index, dtype=float)

    if "Volume" in df.columns:
        vol_raw = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
        vol_feat = np.log1p(vol_raw)
        dollar_vol = np.log1p(vol_raw * close)
        sh = pd.to_numeric(shares_daily.reindex(df.index), errors="coerce").astype(float)
        turnover = (vol_raw / sh.replace(0.0, np.nan))

    # Z-scored features (helps regression stability)
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

    # Clean
    out = out.replace([np.inf, -np.inf], np.nan)

    return out

def _standardize_train_apply(X_train: np.ndarray, X_apply: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize with train mean/std (ddof=0). Returns X_train_std, X_apply_std, (mu, sd).
    """
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    Xtr = (X_train - mu) / sd
    Xap = (X_apply - mu) / sd
    Xtr = np.where(np.isfinite(Xtr), Xtr, 0.0)
    Xap = np.where(np.isfinite(Xap), Xap, 0.0)
    return Xtr, Xap, mu

def _ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, l2: float = RIDGE_L2) -> float:
    """
    Ridge regression with intercept (added column of ones).
    """
    # Filter finite rows
    m = np.isfinite(y_train)
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D")
    if m.sum() < 30:
        return float("nan")

    Xt = X_train[m]
    yt = y_train[m].astype(float)

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
    yhat = float(Xp @ w)
    return yhat

def walk_forward_spread_forecast(
    df_feat: pd.DataFrame,
    y_target: pd.Series,
    horizon: int,
    train_days: int,
    test_days: int,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Predict spread(t) using features(t-horizon). We align by shifting features forward.
    - Features used for predicting y at date t are from date t-horizon.
    - Walk-forward OOS: last `test_days` are OOS; each OOS point fits on a rolling train window.
    Returns:
      preds (Series aligned to realized dates), backtest_meta rows used for UI.
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

    # Shift features so that row at t contains features from t-horizon
    X_df = df.shift(horizon)

    # Choose feature columns (all numeric)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").astype(float)

    # Build prediction series
    preds = pd.Series(index=idx, dtype=float)

    n = len(idx)
    if n < (train_days + 50):
        return preds, []

    # Define OOS window: last test_days
    oos_start = max(0, n - test_days)
    oos_idx = list(range(oos_start, n))

    # For UI meta: pick 4 anchor rows if available
    meta = []

    for i in oos_idx:
        d = idx[i]

        # train window end must be BEFORE d (no look-ahead)
        train_end = i
        train_start = max(0, train_end - train_days)

        # Extract train set: [train_start, train_end)
        X_train = X_df.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]

        # Need current X row
        x_row = X_df.iloc[i]

        # Drop rows with too many NaNs (but keep sparse; we zero-impute after standardization)
        if y_train.dropna().size < MIN_TRAIN_ROWS:
            preds.loc[d] = np.nan
            continue

        # If x_row is all NaN, cannot predict
        if np.isfinite(x_row.values.astype(float)).sum() < 3:
            preds.loc[d] = np.nan
            continue

        # Convert to arrays
        Xtr = X_train.values.astype(float)
        ytr = y_train.values.astype(float)
        xap = x_row.values.astype(float)

        # Standardize using train stats
        Xtr_std, xap_std, _ = _standardize_train_apply(Xtr, xap.reshape(1, -1))

        # Predict
        yhat = _ridge_fit_predict(Xtr_std, ytr, xap_std.flatten(), l2=RIDGE_L2)
        preds.loc[d] = yhat

    # Build simple meta rows for UI (most recent OOS points)
    # We try to label: 3M ago, 6M ago, 1Y ago, Test Start
    def _pick_date_label(days_back: int, label: str):
        j = n - 1 - days_back
        if j < oos_start or j < 0 or j >= n:
            return
        meta.append({"date": idx[j], "period": f"{label} (OOS)"})

    _pick_date_label(63, "3 Months Ago")
    _pick_date_label(126, "6 Months Ago")
    _pick_date_label(252, "1 Year Ago")
    if oos_start < n:
        meta.append({"date": idx[oos_start], "period": "Test Start (OOS)"})

    return preds, meta

# =========================================================
# OPTIONAL: local runner
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
