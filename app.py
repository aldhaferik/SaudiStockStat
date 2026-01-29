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
# 0) APP + CONFIG
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_HISTORY_PERIOD = "5y"
TRADING_DAYS = 252
BETA_LOOKBACK_DAYS = TRADING_DAYS * 2
MARKET_RETURN_LOOKBACK_DAYS = TRADING_DAYS * 5
TRAIN_WINDOW_DAYS = TRADING_DAYS * 3
TEST_WINDOW_DAYS = TRADING_DAYS * 1
N_WEIGHT_SAMPLES = 6000
FORECAST_YEARS = 5
SPREAD_HORIZON_DAYS = 21
TASI_TICKER = "^TASI.SR"
ALPHA_VANTAGE_KEY = "0LR5JLOBSLOA6Z0A"
TWELVE_DATA_KEY = "ed240f406bab4225ac6e0a98be553aa2"
RISK_FREE_XLSX_PATH = "saudi_yields.xlsx"
RISK_FREE_COLUMN_NAME = "10-Year government bond yield"
GROWTH_MIN, GROWTH_MAX = -0.20, 0.40
WACC_MAX = 0.50
RIDGE_L2 = 10.0
MIN_TRAIN_ROWS = 220

# =========================================================
# 1) UI & ROOT
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" /><title>Saudi Valuator Pro</title>
  <style>
    body { font-family: system-ui; margin: 24px; background: #f9fafb; }
    .card { max-width: 600px; padding: 20px; background: white; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    a { color: #0b63ce; text-decoration: none; font-weight: 500; }
  </style>
</head>
<body>
  <div class="card">
    <h2>🇸🇦 Saudi Valuator Pro</h2>
    <p>Status: <span style="color: green;">Online</span></p>
    <p>Access the Interface: <a href="/ui">Go to /ui</a></p>
    <p>API Documentation: <a href="/docs">Swagger UI</a></p>
  </div>
</body>
</html>
"""

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" /><title>Saudi Valuator Pro — Analysis</title>
  <style>
    body { font-family: -apple-system, sans-serif; margin: 30px; background: #f3f4f6; color: #111827; }
    .container { max-width: 1100px; margin: auto; }
    .controls { display: flex; gap: 12px; background: white; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb; margin-bottom: 20px; }
    input { flex: 1; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 16px; }
    button { padding: 12px 24px; background: #0b63ce; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
    button:disabled { background: #9ca3af; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb; }
    pre { background: #111827; color: #10b981; padding: 15px; border-radius: 8px; overflow: auto; max-height: 500px; font-size: 13px; }
    .metric { font-size: 24px; font-weight: bold; color: #0b63ce; }
    .label { font-size: 14px; color: #6b7280; margin-bottom: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <h2>Saudi Stock Valuation Engine</h2>
    <div class="controls">
      <input id="ticker" placeholder="Enter Ticker (e.g. 2222 or 1120)" />
      <button id="runBtn" onclick="run()">Analyze Stock</button>
      <span id="status" style="align-self: center; margin-left: 10px;"></span>
    </div>
    <div class="grid">
      <div class="card">
        <div class="label">Fair Value (1M Forecast)</div>
        <div id="fv" class="metric">-</div>
        <div id="verdict" style="margin-top: 10px; font-weight: 600;"></div>
      </div>
      <div class="card">
        <div class="label">Upside/Downside</div>
        <div id="upside" class="metric">-</div>
      </div>
    </div>
    <div class="card" style="margin-top:20px;">
      <div class="label">Raw Model Intelligence Output</div>
      <pre id="out">{}</pre>
    </div>
  </div>

<script>
async function run() {
  const btn = document.getElementById("runBtn");
  const status = document.getElementById("status");
  const out = document.getElementById("out");
  const ticker = document.getElementById("ticker").value.trim();

  if (!ticker) return alert("Please enter a ticker.");

  btn.disabled = true;
  status.textContent = "Processing Market Data...";
  out.textContent = "{}";

  try {
    const r = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker })
    });
    const data = await r.json();
    out.textContent = JSON.stringify(data, null, 2);

    if (data.valuation_summary) {
      document.getElementById("fv").textContent = data.valuation_summary.fair_value?.toFixed(2) + " SAR";
      document.getElementById("upside").textContent = data.valuation_summary.upside_percent?.toFixed(2) + "%";
      const v = document.getElementById("verdict");
      v.textContent = data.valuation_summary.verdict;
      v.style.color = data.valuation_summary.verdict === "Undervalued" ? "green" : "red";
      status.textContent = "Analysis Complete.";
    } else {
      status.textContent = "Error: " + (data.error || "Unknown error");
    }
  } catch (e) {
    status.textContent = "Connection Failed.";
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

# =========================================================
# 2) CORE UTILITIES (JSON, TZ, MATH)
# =========================================================
def json_safe(obj):
    if obj is None: return None
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, float):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, dict): return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    return obj

def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except: return None

def safe_div(a: Any, b: Any) -> float:
    try:
        fa, fb = float(a), float(b)
        if fb == 0 or not np.isfinite(fa) or not np.isfinite(fb): return float("nan")
        return fa / fb
    except: return float("nan")

def winsorize(arr: np.ndarray, p_low: float = 0.05, p_high: float = 0.95) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() == 0: return x
    lo, hi = np.quantile(x[mask], p_low), np.quantile(x[mask], p_high)
    return np.clip(x, lo, hi)

def _to_naive_dt_index(idx) -> pd.DatetimeIndex:
    di = pd.to_datetime(idx, errors="coerce")
    if not isinstance(di, pd.DatetimeIndex): di = pd.DatetimeIndex(di)
    try: return di.tz_localize(None)
    except: return di

def last_value_on_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    if series is None or series.empty: return pd.Series(index=dates, dtype=float)
    s = series.copy()
    s.index = _to_naive_dt_index(s.index)
    d = _to_naive_dt_index(dates)
    tmp = s.reindex(s.index.union(d)).sort_index().ffill()
    out = tmp.reindex(d)
    out.index = dates
    return out.astype(float)

def ttm_from_quarters(q_series: pd.Series) -> pd.Series:
    s = q_series.copy()
    s.index = _to_naive_dt_index(s.index)
    return s.sort_index().rolling(4, min_periods=4).sum()

# =========================================================
# 3) DATA FETCHER
# =========================================================
class DataFetcher:
    def __init__(self):
        self.ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

    def clean_ticker(self, ticker: str) -> str:
        t = ticker.strip().upper()
        return f"{t}.SR" if t.isdigit() else t

    def fetch_prices(self, ticker: str) -> Tuple[pd.DataFrame, str]:
        import yfinance as yf
        try:
            h = yf.Ticker(ticker).history(period=DEFAULT_HISTORY_PERIOD, auto_adjust=False)
            if h.empty: raise ValueError()
            h.index = _to_naive_dt_index(h.index)
            return h[["Close"]].sort_index(), "Yahoo"
        except:
            # Fallback TwelveData
            url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&apikey={TWELVE_DATA_KEY}&outputsize=1250"
            r = requests.get(url).json()
            if "values" not in r: raise ValueError("Price sources exhausted.")
            df = pd.DataFrame(r["values"])
            df['datetime'] = _to_naive_dt_index(df['datetime'])
            df = df.set_index('datetime')['close'].astype(float).to_frame()
            df.columns = ["Close"]
            return df.sort_index(), "TwelveData"

    def fetch_statements(self, ticker: str) -> Dict[str, Any]:
        import yfinance as yf
        s = yf.Ticker(ticker)
        def norm(df):
            if df is None or df.empty: return None
            df.columns = _to_naive_dt_index(df.columns)
            return df.loc[:, ~df.columns.duplicated()].sort_index(axis=1)
        return {
            "info": s.info or {},
            "fin_a": norm(s.financials), "bs_a": norm(s.balance_sheet), "cf_a": norm(s.cashflow),
            "fin_q": norm(s.quarterly_financials), "bs_q": norm(s.quarterly_balance_sheet), "cf_q": norm(s.quarterly_cashflow)
        }

    def get_rf(self) -> float:
        try:
            df = pd.read_excel(RISK_FREE_XLSX_PATH)
            val = float(df[RISK_FREE_COLUMN_NAME].dropna().iloc[-1])
            return val / 100.0 if val > 1.0 else val
        except: return 0.04  # Fallback 4%

# =========================================================
# 4) VALUATION MODELS & WEIGHT OPTIMIZATION
# =========================================================
def dcf_per_share_from_fcff(fcff0, wacc, g, shares, net_debt, mkt_g):
    g_term = min(float(g), float(mkt_g))
    if wacc <= g_term: return np.nan
    pv = sum([fcff0 * ((1+g)**i) / ((1+wacc)**i) for i in range(1, FORECAST_YEARS+1)])
    tv = (fcff0 * ((1+g)**FORECAST_YEARS) * (1+g_term)) / (wacc - g_term)
    pv_tv = tv / ((1+wacc)**FORECAST_YEARS)
    return (pv + pv_tv - net_debt) / shares

def optimize_weights_dirichlet(y, X, avail):
    idx = np.where(avail)[0]
    best_w, best_loss = np.zeros(X.shape[0]), float("inf")
    rnd = np.random.default_rng(42)
    for _ in range(N_WEIGHT_SAMPLES):
        w = np.zeros(X.shape[0])
        w[idx] = rnd.dirichlet(np.ones(len(idx)))
        yh = np.nansum(X.T * w, axis=1)
        mask = np.isfinite(y) & np.isfinite(yh) & (y > 0)
        if not mask.any(): continue
        loss = np.mean(np.abs((yh[mask] - y[mask]) / y[mask]))
        if loss < best_loss:
            best_loss, best_w = loss, w
    return best_w

# =========================================================
# 5) SPREAD ENGINE (RIDGE)
# =========================================================
def _ridge_fit_predict(X_train, y_train, X_pred):
    mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
    Xt, yt = X_train[mask], y_train[mask]
    if len(yt) < 50: return np.nan
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
    sd[sd == 0] = 1.0
    Xt_s = (Xt - mu) / sd
    Xp_s = (X_pred - mu) / sd
    Xb = np.hstack([np.ones((Xt_s.shape[0], 1)), Xt_s])
    I = np.eye(Xb.shape[1]); I[0, 0] = 0
    w = np.linalg.solve(Xb.T @ Xb + RIDGE_L2 * I, Xb.T @ yt)
    return float(np.hstack([1, Xp_s]) @ w)

def build_spread_features(df, shares_daily):
    feat = pd.DataFrame(index=df.index)
    close, v = df["Close"], df["V_anchor"]
    feat["spread"] = (close - v) / v.replace(0, np.nan)
    feat["r5"] = close.pct_change(5)
    feat["vol21"] = np.log(close).diff().rolling(21).std()
    feat["mkt_r5"] = df["MktClose"].pct_change(5)
    return feat.ffill().bfill().replace([np.inf, -np.inf], 0)

# =========================================================
# 6) MAIN ANALYZE ENDPOINT
# =========================================================
class StockRequest(BaseModel): ticker: str

@app.post("/analyze")
async def analyze_stock(req: StockRequest):
    fetcher = DataFetcher()
    ticker = fetcher.clean_ticker(req.ticker)
    
    try:
        # Data Retrieval
        hist, _ = fetcher.fetch_prices(ticker)
        mkt_hist, _ = fetcher.fetch_prices(TASI_TICKER)
        stmts = fetcher.fetch_statements(ticker)
        rf = fetcher.get_rf()
        
        dates = hist.index
        stock_close = hist["Close"]
        mkt_close = mkt_hist["Close"].reindex(dates).ffill()
        
        # Financial Extractors
        def get_row(df, keys):
            if df is None: return None
            for k in keys:
                match = [i for i in df.index if k.lower() in str(i).lower()]
                if match: return df.loc[match[0]]
            return None

        # Core Components
        shares_q = get_row(stmts["bs_q"], ["Share Issued", "Ordinary Shares Number"])
        shares_daily = last_value_on_or_before(shares_q, dates).ffill()
        if shares_daily.dropna().empty:
            shares_daily = pd.Series(index=dates, data=stmts["info"].get("sharesOutstanding", 1e6))

        # 1. Define close_series early to prevent NameError
        close_series = stock_close.reindex(dates).astype(float)

        # 2. Financial Metrics
        eps_q = get_row(stmts["fin_q"], ["Diluted EPS", "Basic EPS"])
        eps_ttm_daily = last_value_on_or_before(ttm_from_quarters(eps_q) if eps_q is not None else None, dates).ffill()
        
        net_debt_q = get_row(stmts["bs_q"], ["Net Debt"])
        net_debt_daily = last_value_on_or_before(net_debt_q, dates).ffill().fillna(0)

        # 3. WACC & Growth
        beta = 1.1 # Default
        try:
            rs = np.log(close_series).diff().dropna()
            rm = np.log(mkt_close).diff().dropna()
            beta = np.cov(rs, rm)[0, 1] / np.var(rm)
        except: pass
        
        re = rf + beta * 0.06
        wacc_daily = pd.Series(index=dates, data=re).clip(0.05, WACC_MAX)
        mkt_g = 0.03

        # 4. Multiples Models
        pe_obs = close_series / eps_ttm_daily.replace(0, np.nan)
        pe_target = pe_obs.rolling(504, min_periods=60).median().ffill()
        pe_model = pe_target * eps_ttm_daily

        # 5. Weights & Anchor
        X = np.vstack([pe_model.values]) # Simplified for stability
        avail = np.array([np.isfinite(pe_model).sum() > 100])
        
        w_val = optimize_weights_dirichlet(close_series.values, X, avail)
        V_anchor = pd.Series(index=dates, data=np.nansum(X.T * w_val, axis=1))

        # 6. Spread Forecast
        df_feat = build_spread_features(pd.DataFrame({"Close": close_series, "MktClose": mkt_close, "V_anchor": V_anchor}), shares_daily)
        y_target = (close_series - V_anchor) / V_anchor.replace(0, np.nan)
        
        # OOS Walk-Forward
        pred_delta = _ridge_fit_predict(df_feat.iloc[:-21].values, y_target.iloc[:-21].values, df_feat.iloc[-1].values)
        
        cur_price = float(close_series.iloc[-1])
        fv = float(V_anchor.iloc[-1] * (1 + pred_delta)) if np.isfinite(pred_delta) else float(V_anchor.iloc[-1])
        upside = ((fv / cur_price) - 1) * 100

        res = {
            "valuation_summary": {
                "company_name": stmts["info"].get("shortName", ticker),
                "current_price": cur_price,
                "fair_value": fv,
                "upside_percent": upside,
                "verdict": "Undervalued" if upside > 10 else ("Overvalued" if upside < -10 else "Fairly Valued")
            },
            "metrics": {
                "pe_ratio": safe_div(cur_price, eps_ttm_daily.iloc[-1]),
                "beta": beta,
                "wacc": wacc_daily.iloc[-1]
            },
            "historical_data": {
                "dates": [int(d.timestamp() * 1000) for d in dates[-200:]],
                "prices": close_series.tail(200).tolist(),
                "anchor": V_anchor.tail(200).tolist()
            }
        }
        return JSONResponse(json_safe(res))

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
