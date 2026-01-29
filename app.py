# app.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

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
# 1) CONFIG (price prediction, not valuation)
# =========================================================
DEFAULT_HISTORY_PERIOD = "5y"
TRADING_DAYS = 252

TASI_TICKER = "^TASI.SR"

# Backup price sources (as you requested)
ALPHA_VANTAGE_KEY = "0LR5JLOBSLOA6Z0A"
TWELVE_DATA_KEY = "ed240f406bab4225ac6e0a98be553aa2"

# Model config
LOOKBACK_LAGS = [1, 2, 3, 5, 10, 21, 63]          # returns lags (1d..3m)
VOL_WINDOWS = [10, 21, 63]                        # rolling vol windows
MA_WINDOWS = [10, 21, 63, 126]                    # moving averages
RSI_WINDOW = 14

# Training / evaluation
TRAIN_WINDOW_DAYS = TRADING_DAYS * 3              # last 3y for training at each step
TEST_STEP_DAYS = 21                               # step forward monthly
MIN_ROWS_TO_TRAIN = 400                           # require enough rows after feature dropna

# Forecast horizons (trading days)
HORIZONS = [1, 5]

# =========================================================
# 2) JSON-SAFE SERIALIZATION
# =========================================================
def json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj

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

# =========================================================
# 3) HTML UI (kept simple; shows forecasts + backtest)
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Saudi Price Predictor</title>
  <script src="https://code.highcharts.com/highcharts.js"></script>
  <style>
    :root { --bg:#f0f2f5; --card:#fff; --primary:#0a192f; --accent:#007aff; --text:#333; }
    body { font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background:var(--bg); margin:0; padding:20px; color:var(--text); }
    .container { max-width: 1200px; margin: 0 auto; }
    .search-bar { background: var(--card); padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); display: flex; gap: 10px; margin-bottom: 25px; }
    input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; outline: none; }
    button { padding: 12px 25px; background: var(--primary); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .loading { text-align:center; padding:40px; display:none; }
    .spinner { border:4px solid #f3f3f3; border-top:4px solid var(--accent); border-radius:50%; width:40px; height:40px; animation:spin 1s linear infinite; margin:0 auto 15px; }
    @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
    .card { background: var(--card); border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 20px; }
    .row { display:grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .big { font-size: 34px; font-weight: 800; color: var(--primary); }
    .muted { color:#777; font-size:13px; margin-top:6px; }
    table { width:100%; border-collapse: collapse; margin-top: 10px; }
    th { text-align:left; font-size:11px; color:#888; padding-bottom:8px; border-bottom:1px solid #eee; }
    td { padding:10px 0; font-size:13px; border-bottom:1px solid #f6f6f6; }
    #error { display:none; padding: 15px; background:#ffebee; color:#c62828; border-radius:8px; margin-bottom:20px; }
    @media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="container">
  <div class="search-bar">
    <input type="text" id="ticker" placeholder="Enter Ticker (e.g. 1120)" />
    <button onclick="analyze()" id="btn">PREDICT</button>
  </div>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <h3>Training + walk-forward backtest…</h3>
    <p class="muted">This is a supervised price-forecast model (not valuation)</p>
  </div>

  <div id="error"></div>

  <div id="dashboard" style="display:none;">
    <div class="row">
      <div class="card">
        <div class="big" id="name">--</div>
        <div class="muted" id="subtitle">--</div>
        <div style="margin-top:14px;">
          <div style="font-size:14px; color:#555;">Current Price</div>
          <div style="font-size:42px; font-weight:900;" id="price">--</div>
        </div>
      </div>

      <div class="card">
        <div style="font-size:13px; font-weight:800; color:#888; text-transform:uppercase; letter-spacing:.5px;">Forecasts</div>
        <div style="margin-top:12px;">
          <div style="font-size:14px; color:#555;">Next day (T+1)</div>
          <div style="font-size:32px; font-weight:900;" id="p1">--</div>
          <div class="muted" id="p1r">--</div>
        </div>
        <div style="margin-top:14px;">
          <div style="font-size:14px; color:#555;">5 trading days (T+5)</div>
          <div style="font-size:32px; font-weight:900;" id="p5">--</div>
          <div class="muted" id="p5r">--</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div id="chartContainer" style="height: 360px;"></div>
    </div>

    <div class="row">
      <div class="card">
        <div style="font-size:13px; font-weight:800; color:#888; text-transform:uppercase; letter-spacing:.5px;">Walk-forward Backtest</div>
        <table>
          <thead>
            <tr>
              <th>Period</th>
              <th>Actual</th>
              <th>Pred</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody id="btBody"></tbody>
        </table>
      </div>

      <div class="card">
        <div style="font-size:13px; font-weight:800; color:#888; text-transform:uppercase; letter-spacing:.5px;">Metrics (OOS)</div>
        <table>
          <tbody>
            <tr><th>MAPE</th><td id="mape">--</td></tr>
            <tr><th>MAE</th><td id="mae">--</td></tr>
            <tr><th>Directional Accuracy</th><td id="diracc">--</td></tr>
            <tr><th>Data Sources</th><td id="sources">--</td></tr>
          </tbody>
        </table>
        <div class="muted" style="margin-top:10px;">Directional accuracy = % times sign(pred return) == sign(actual return)</div>
      </div>
    </div>
  </div>
</div>

<script>
async function analyze() {
  const ticker = document.getElementById('ticker').value;
  const btn = document.getElementById('btn');
  const loading = document.getElementById('loading');
  const dashboard = document.getElementById('dashboard');
  const err = document.getElementById('error');

  if(!ticker) return;

  dashboard.style.display = 'none';
  err.style.display = 'none';
  loading.style.display = 'block';
  btn.disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ticker: ticker})
    });

    const text = await res.text();
    let data;
    try { data = JSON.parse(text); }
    catch (e) { throw new Error("Non-JSON response: " + text.slice(0,200)); }

    loading.style.display = 'none';
    btn.disabled = false;

    if (data.error) {
      err.innerText = data.error;
      err.style.display = 'block';
      return;
    }

    document.getElementById('name').innerText = data.company_name ?? "--";
    document.getElementById('subtitle').innerText = (ticker.toUpperCase() + ".SR") + " • " + (data.sector ?? "Unknown");
    document.getElementById('price').innerText = Number(data.current_price ?? 0).toFixed(2);

    document.getElementById('p1').innerText = Number(data.forecasts?.p1 ?? 0).toFixed(2);
    document.getElementById('p1r').innerText = "Pred return: " + Number(data.forecasts?.r1 ?? 0).toFixed(2) + "%";

    document.getElementById('p5').innerText = Number(data.forecasts?.p5 ?? 0).toFixed(2);
    document.getElementById('p5r').innerText = "Pred return: " + Number(data.forecasts?.r5 ?? 0).toFixed(2) + "%";

    document.getElementById('mape').innerText = (data.metrics?.mape ?? "--") + "%";
    document.getElementById('mae').innerText = (data.metrics?.mae ?? "--");
    document.getElementById('diracc').innerText = (data.metrics?.dir_acc ?? "--") + "%";
    document.getElementById('sources').innerText = data.source_used ?? "--";

    // Backtest table
    const btBody = document.getElementById('btBody');
    btBody.innerHTML = "";
    (data.backtest_points || []).forEach(b => {
      const a = Number(b.actual);
      const p = Number(b.pred);
      const diff = (a && isFinite(a)) ? Math.abs((p - a) / a) * 100 : 0;
      const row = `<tr>
        <td>${b.period ?? ""}</td>
        <td>${isFinite(a) ? a.toFixed(2) : "N/A"}</td>
        <td>${isFinite(p) ? p.toFixed(2) : "N/A"}</td>
        <td style="font-weight:800;">${diff.toFixed(1)}%</td>
      </tr>`;
      btBody.innerHTML += row;
    });

    // Chart
    const dates = data.series?.dates || [];
    const prices = data.series?.actual || [];
    const pred1 = data.series?.pred1 || [];
    if (dates.length && prices.length && pred1.length) {
      Highcharts.chart('chartContainer', {
        chart: { backgroundColor: 'transparent' },
        title: { text: 'Actual vs 1-day Ahead Walk-forward Predictions' },
        xAxis: { type: 'datetime' },
        yAxis: { title: { text: null }, gridLineColor: '#eee' },
        series: [{
          name: 'Actual Price',
          data: dates.map((d, i) => [d, prices[i]]),
          type: 'area'
        },{
          name: 'Predicted (T+1)',
          data: dates.map((d, i) => [d, pred1[i]]),
          type: 'line',
          lineWidth: 2
        }],
        credits: { enabled: false }
      });
    }

    dashboard.style.display = 'block';

  } catch (e) {
    loading.style.display = 'none';
    btn.disabled = false;
    err.innerText = "Error: " + (e?.message || e);
    err.style.display = 'block';
  }
}
</script>

</body>
</html>
"""

# =========================================================
# 4) DATA FETCHER (prices + basic info)
# =========================================================
class DataFetcher:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ]

    def _headers(self):
        return {"User-Agent": np.random.choice(self.user_agents)}

    @staticmethod
    def clean_saudi_ticker(ticker: str) -> str:
        t = (ticker or "").strip().upper()
        if t.replace(".", "").isdigit() and not t.endswith(".SR"):
            return f"{t}.SR"
        return t

    def fetch_prices_yahoo(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError(f"No Yahoo price history for {ticker}.")
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
                rows.append((pd.to_datetime(v["datetime"]), float(v["close"])))
            except Exception:
                continue
        if not rows:
            raise ValueError("Twelve Data: could not parse values.")
        df = pd.DataFrame(rows, columns=["Date", "Close"]).set_index("Date").sort_index()
        return df

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
        df.index = pd.to_datetime(df.index)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_index().tail(1250)
        if df.empty:
            raise ValueError("Alpha Vantage: empty parsed dataframe.")
        return df[["Close"]]

    def fetch_prices(self, ticker: str, period: str = DEFAULT_HISTORY_PERIOD) -> Tuple[pd.DataFrame, str]:
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

    def fetch_info_yahoo(self, ticker: str) -> Dict[str, Any]:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        try:
            info = stock.info or {}
        except Exception:
            info = {}
        return info

# =========================================================
# 5) FEATURE ENGINEERING (daily features)
# =========================================================
def rsi(series: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def make_features(px: pd.Series, mkt: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": px, "mkt": mkt}).dropna()
    df["ret"] = np.log(df["close"]).diff()
    df["mkt_ret"] = np.log(df["mkt"]).diff()

    # Lags
    for lag in LOOKBACK_LAGS:
        df[f"ret_l{lag}"] = df["ret"].shift(lag)
        df[f"mkt_ret_l{lag}"] = df["mkt_ret"].shift(lag)

    # Volatility
    for w in VOL_WINDOWS:
        df[f"vol_{w}"] = df["ret"].rolling(w, min_periods=w).std()
        df[f"mkt_vol_{w}"] = df["mkt_ret"].rolling(w, min_periods=w).std()

    # Moving average ratios (trend)
    for w in MA_WINDOWS:
        ma = df["close"].rolling(w, min_periods=w).mean()
        df[f"ma_ratio_{w}"] = df["close"] / ma - 1.0

    # RSI
    df["rsi"] = rsi(df["close"], RSI_WINDOW)

    # Market-relative momentum proxy
    df["rel_ret_21"] = df["ret"].rolling(21, min_periods=21).sum() - df["mkt_ret"].rolling(21, min_periods=21).sum()

    # Day-of-week effect (0..4)
    df["dow"] = df.index.dayofweek.astype(float)

    return df

def add_targets(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        # predict future log return over h days
        out[f"y_ret_{h}"] = out["ret"].shift(-h).rolling(h).sum()
    return out

# =========================================================
# 6) MODEL (sklearn, no exotic dependencies)
# =========================================================
def train_model_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()

    X_test = test_df[feature_cols].to_numpy()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        l2_regularization=0.5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)

def walk_forward_backtest(
    df: pd.DataFrame,
    horizons: List[int],
) -> Dict[str, Any]:
    # Build features/targets set
    feature_cols = [c for c in df.columns if c not in ["close", "mkt", "ret", "mkt_ret"] and not c.startswith("y_ret_")]
    results = {h: {"pred_ret": pd.Series(index=df.index, dtype=float)} for h in horizons}

    n = len(df)
    if n < (TRAIN_WINDOW_DAYS + 100):
        raise ValueError("Not enough data for walk-forward training window.")

    # Walk forward by monthly steps; at each step, train on trailing TRAIN_WINDOW_DAYS
    idxs = np.arange(0, n, TEST_STEP_DAYS, dtype=int)

    for end_i in idxs:
        if end_i < TRAIN_WINDOW_DAYS:
            continue
        train_start = max(0, end_i - TRAIN_WINDOW_DAYS)
        train = df.iloc[train_start:end_i].dropna()
        test = df.iloc[end_i:end_i + TEST_STEP_DAYS].dropna()

        if len(train) < MIN_ROWS_TO_TRAIN or len(test) < 5:
            continue

        for h in horizons:
            target_col = f"y_ret_{h}"
            tr = train.dropna(subset=[target_col])
            te = test.copy()

            if len(tr) < MIN_ROWS_TO_TRAIN:
                continue

            # Predict returns for the test slice
            preds = train_model_and_predict(tr, te, target_col, feature_cols)
            results[h]["pred_ret"].loc[te.index] = preds

    # Convert predicted returns to predicted prices (using last known close at each date)
    out = {}
    for h in horizons:
        pred_ret = results[h]["pred_ret"]
        # Predicted price at date t = close[t] * exp(pred_ret[t])
        pred_price = df["close"] * np.exp(pred_ret)
        out[h] = {
            "pred_ret": pred_ret,
            "pred_price": pred_price,
        }
    return out

def compute_oos_metrics(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    mask = actual.notna() & pred.notna() & np.isfinite(actual) & np.isfinite(pred) & (actual > 0)
    if mask.sum() == 0:
        return {"mape": np.nan, "mae": np.nan}

    a = actual[mask].to_numpy()
    p = pred[mask].to_numpy()

    mape = float(np.mean(np.abs((p - a) / a)) * 100.0)
    mae = float(np.mean(np.abs(p - a)))
    return {"mape": mape, "mae": mae}

def directional_accuracy(actual_close: pd.Series, pred_close: pd.Series) -> float:
    # compares sign of 1-day return implied by pred vs actual
    a_ret = actual_close.pct_change()
    p_ret = pred_close.pct_change()
    mask = a_ret.notna() & p_ret.notna()
    if mask.sum() == 0:
        return np.nan
    hit = np.mean(np.sign(a_ret[mask]) == np.sign(p_ret[mask]))
    return float(hit * 100.0)

# =========================================================
# 7) REQUEST MODEL
# =========================================================
class StockRequest(BaseModel):
    ticker: str

# =========================================================
# 8) MAIN PREDICTION ENDPOINT
# =========================================================
@app.post("/predict")
def predict_price(request: StockRequest):
    try:
        fetcher = DataFetcher()
        ticker = fetcher.clean_saudi_ticker(request.ticker)

        # Prices
        hist, source_stock = fetcher.fetch_prices(ticker, period=DEFAULT_HISTORY_PERIOD)
        mkt_hist, source_mkt = fetcher.fetch_prices(TASI_TICKER, period=DEFAULT_HISTORY_PERIOD)

        if hist is None or hist.empty or "Close" not in hist.columns:
            return JSONResponse({"error": "No valid stock prices."}, status_code=200)
        if mkt_hist is None or mkt_hist.empty or "Close" not in mkt_hist.columns:
            return JSONResponse({"error": "No valid market index prices (TASI)."}, status_code=200)

        stock_close = hist["Close"].astype(float).dropna()
        mkt_close = mkt_hist["Close"].astype(float).dropna()

        # Align by date intersection
        aligned = pd.DataFrame({"s": stock_close, "m": mkt_close}).dropna()
        if len(aligned) < 600:
            return JSONResponse({"error": "Not enough overlapping history between stock and TASI for supervised prediction."}, status_code=200)

        px = aligned["s"]
        mk = aligned["m"]

        # Build features + targets
        feat = make_features(px, mk)
        feat = add_targets(feat, HORIZONS)
        feat = feat.dropna()

        if len(feat) < MIN_ROWS_TO_TRAIN:
            return JSONResponse({"error": "Insufficient data after feature engineering."}, status_code=200)

        # Walk-forward backtest predictions
        wf = walk_forward_backtest(feat, HORIZONS)

        # Use 1-day ahead series for chart/metrics
        pred1 = wf[1]["pred_price"]
        actual = feat["close"]

        met = compute_oos_metrics(actual, pred1)
        diracc = directional_accuracy(actual, pred1)

        # Current forecasts: train on last TRAIN_WINDOW_DAYS and predict next horizon using last row features
        # We reuse the same modeling function but fit once per horizon on the latest window.
        feature_cols = [c for c in feat.columns if c not in ["close", "mkt", "ret", "mkt_ret"] and not c.startswith("y_ret_")]

        current_price = float(actual.iloc[-1])
        last_row = feat.iloc[[-1]].copy()

        forecasts = {}
        for h in HORIZONS:
            target_col = f"y_ret_{h}"
            train = feat.iloc[-TRAIN_WINDOW_DAYS:].dropna(subset=[target_col])
            if len(train) < MIN_ROWS_TO_TRAIN:
                forecasts[f"p{h}"] = None
                forecasts[f"r{h}"] = None
                continue

            from sklearn.ensemble import HistGradientBoostingRegressor

            X_train = train[feature_cols].to_numpy()
            y_train = train[target_col].to_numpy()
            X_last = last_row[feature_cols].to_numpy()

            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=6,
                max_iter=400,
                l2_regularization=0.5,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred_ret = float(model.predict(X_last)[0])
            pred_price = current_price * float(np.exp(pred_ret))

            forecasts[f"p{h}"] = pred_price
            forecasts[f"r{h}"] = pred_ret * 100.0

        # Backtest points (OOS-style snapshots)
        bt_points = []
        for label, days in [("3 Months Ago (OOS)", 63), ("6 Months Ago (OOS)", 126), ("1 Year Ago (OOS)", 252)]:
            if len(actual) > days:
                dt = actual.index[-days]
                a = float(actual.loc[dt])
                p = _to_float(pred1.loc[dt])
                if p is not None:
                    bt_points.append({"period": label, "actual": a, "pred": float(p)})

        # Info
        info = fetcher.fetch_info_yahoo(ticker)
        company_name = info.get("longName") or f"Saudi Stock {request.ticker}"
        sector = (info.get("sector") or "Unknown").title()

        dates_ms = (actual.index.astype(np.int64) // 10**6).tolist()

        result = {
            "company_name": company_name,
            "sector": sector,
            "current_price": current_price,
            "forecasts": {
                "p1": forecasts.get("p1"),
                "r1": forecasts.get("r1"),
                "p5": forecasts.get("p5"),
                "r5": forecasts.get("r5"),
            },
            "metrics": {
                "mape": None if not np.isfinite(met["mape"]) else round(float(met["mape"]), 2),
                "mae": None if not np.isfinite(met["mae"]) else round(float(met["mae"]), 4),
                "dir_acc": None if not np.isfinite(diracc) else round(float(diracc), 2),
            },
            "backtest_points": bt_points,
            "series": {
                "dates": dates_ms,
                "actual": [float(x) for x in actual.to_list()],
                "pred1": [None if (not np.isfinite(x)) else float(x) for x in pred1.reindex(actual.index).to_numpy()],
            },
            "source_used": f"{source_stock} (stock), {source_mkt} (TASI), sklearn HGBR",
            "note": "This is a supervised short-horizon predictor trained on daily features; not a valuation model.",
        }

        return JSONResponse(content=json_safe(result), status_code=200)

    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {str(e)}"}, status_code=200)

# =========================================================
# 9) RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
