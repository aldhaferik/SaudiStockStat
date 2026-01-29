# app.py
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
# 1) CONFIG
#    - Goal: accuracy via time-aligned fundamentals + spread engine
# =========================================================
DEFAULT_HISTORY_PERIOD = "5y"
TRADING_DAYS = 252

# windows
BETA_LOOKBACK_DAYS = TRADING_DAYS * 2              # ~2y
MARKET_RETURN_LOOKBACK_DAYS = TRADING_DAYS * 5     # ~5y
TRAIN_WINDOW_DAYS = TRADING_DAYS * 3               # train weights on last 3y
TEST_WINDOW_DAYS = TRADING_DAYS * 1                # test on last 1y (walk-forward)
SOLVER_SAMPLE_STEP = 5                             # sample every ~week for optimization
N_WEIGHT_SAMPLES = 4000                            # Dirichlet search for valuation weights

FORECAST_YEARS = 5
TASI_TICKER = "^TASI.SR"

# Backup price sources (you requested)
ALPHA_VANTAGE_KEY = "0LR5JLOBSLOA6Z0A"
TWELVE_DATA_KEY = "ed240f406bab4225ac6e0a98be553aa2"

# Risk-free source (your repo file)
RISK_FREE_XLSX_PATH = "saudi_yields.xlsx"
RISK_FREE_COLUMN_NAME = "10-Year government bond yield"

# Robustness bounds (sanity only)
GROWTH_MIN = -0.20
GROWTH_MAX = 0.40
WACC_MAX = 0.50

# Spread model settings
SPREAD_HORIZON_DAYS = 21  # ~1 month
RIDGE_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
FEATURE_MIN_ROWS = 200

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

# =========================================================
# 3) SMALL HTML UI (unchanged)
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Saudi Valuator Pro</title>
    <script src="https://code.highcharts.com/highcharts.js"></script>
    <style>
        :root { --bg: #f0f2f5; --card: #ffffff; --primary: #0a192f; --accent: #007aff; --text: #333; }
        body { font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background-color: var(--bg); margin: 0; padding: 20px; color: var(--text); }
        .container { max-width: 1200px; margin: 0 auto; }
        .search-bar { background: var(--card); padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); display: flex; gap: 10px; margin-bottom: 25px; }
        input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; outline: none; }
        button { padding: 12px 25px; background: var(--primary); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }

        .top-section { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
        .bottom-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .full-width { grid-column: span 2; }

        .card { background: var(--card); border-radius: 12px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); position: relative; }
        .card-title { font-size: 13px; font-weight: 700; color: #888; text-transform: uppercase; margin-bottom: 20px; letter-spacing: 0.5px; border-bottom: 1px solid #eee; padding-bottom: 10px; }

        .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; }
        .company-name { font-size: 28px; font-weight: 800; color: var(--primary); margin: 0; line-height: 1.2; }
        .ticker-tag { background: #eee; padding: 4px 8px; border-radius: 4px; font-family: monospace; color: #555; font-size: 14px; }
        .big-price { font-size: 42px; font-weight: 800; color: #333; text-align: right; }
        .price-sub { font-size: 13px; color: #888; text-align: right; margin-top: -5px; }

        .verdict-bar { padding: 15px; border-radius: 8px; text-align: center; font-weight: 800; text-transform: uppercase; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .v-red { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .v-green { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
        .v-gray { background: #f5f5f5; color: #616161; border: 1px solid #e0e0e0; }

        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .stat-box { background: #f8f9fa; padding: 12px; border-radius: 8px; }
        .stat-label { font-size: 11px; font-weight: 700; color: #888; text-transform: uppercase; margin-bottom: 5px; }
        .stat-val { font-size: 16px; font-weight: 600; color: #333; }

        .fv-header { text-align: center; margin-bottom: 20px; }
        .fv-big { font-size: 48px; font-weight: 800; color: var(--accent); }
        .fv-sub { font-size: 13px; color: #888; }
        .sector-tag { font-size: 11px; background: #e0f2f1; color: #00695c; padding: 4px 8px; border-radius: 4px; display:inline-block; margin-top:5px; }
        .dyn-badge { font-size: 9px; background: #333; color: #fff; padding: 2px 5px; border-radius: 3px; margin-left: 5px; vertical-align: middle; }

        .fv-row { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f0f0f0; }
        .fv-row:last-child { border-bottom: none; }
        .fv-label { font-size: 14px; color: #555; }
        .fv-num { font-weight: 700; color: #333; }

        .weight-container { margin-top: 5px; }
        .weight-bar { height: 4px; background: #eee; border-radius: 2px; width: 100%; overflow: hidden; }
        .weight-fill { height: 100%; background: #007aff; width: 0%; }

        .data-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .data-table th { text-align: left; font-size: 11px; color: #888; padding-bottom: 8px; border-bottom: 1px solid #eee; }
        .data-table td { padding: 10px 0; font-size: 13px; font-weight: 500; border-bottom: 1px solid #f9f9f9; }

        .returns-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; text-align: center; }
        .ret-box { background: #f8f9fa; padding: 8px; border-radius: 6px; }
        .ret-label { font-size: 11px; color: #666; margin-bottom: 4px; font-weight: bold; }
        .ret-val { font-size: 14px; font-weight: 600; }
        .pos { color: #28cd41; } .neg { color: #ff3b30; }

        .loading { text-align: center; padding: 40px; display: none; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid var(--accent); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        @media (max-width: 900px) {
            .top-section, .bottom-section { grid-template-columns: 1fr; }
            .full-width { grid-column: span 1; }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="search-bar">
        <input type="text" id="ticker" placeholder="Enter Ticker (e.g. 1120)" />
        <button onclick="analyze()" id="btn">ANALYZE</button>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <h3>Calculating Intrinsic Value...</h3>
        <p style="color:#666; font-size:14px;">Time-aligning fundamentals (TTM) + spread engine (1M forecast)</p>
    </div>

    <div id="error" style="display:none; padding: 15px; background: #ffebee; color: #c62828; border-radius: 8px; margin-bottom: 20px;"></div>

    <div id="dashboard" style="display:none;">

        <div class="top-section">
            <div class="card">
                <div class="header-row">
                    <div>
                        <h1 class="company-name" id="name">--</h1>
                        <span class="ticker-tag" id="tickerDisplay">--</span>
                    </div>
                    <div>
                        <div class="big-price" id="price">--</div>
                        <div class="price-sub">Current Market Price</div>
                    </div>
                </div>

                <div id="verdictBar" class="verdict-bar">--</div>

                <div class="stats-grid">
                    <div class="stat-box"><div class="stat-label">Market Cap</div><div class="stat-val" id="mcap">--</div></div>
                    <div class="stat-box"><div class="stat-label">P/E Ratio</div><div class="stat-val" id="pe">--</div></div>
                    <div class="stat-box"><div class="stat-label">EPS (TTM)</div><div class="stat-val" id="eps">--</div></div>
                    <div class="stat-box"><div class="stat-label">Beta <span id="beta_tag"></span></div><div class="stat-val" id="beta">--</div></div>
                    <div class="stat-box"><div class="stat-label">Growth <span id="growth_tag"></span></div><div class="stat-val" id="growth">--</div></div>
                    <div class="stat-box"><div class="stat-label">Book Value</div><div class="stat-val" id="book">--</div></div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">FAIR VALUE</div>
                <div class="fv-header">
                    <div class="fv-big" id="fair">--</div>
                    <div class="fv-sub">Model target (1M forecast)</div>
                    <div id="sectorMsg" class="sector-tag">--</div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">DCF (WACC: <span id="wacc_display">--</span>)</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_dcf_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="dcf_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_dcf_txt">--</div>
                    </div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">P/E Model (TTM)</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_pe_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="pe_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_pe_txt">--</div>
                    </div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">P/B Model</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_pb_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="pb_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_pb_txt">--</div>
                    </div>
                </div>

                <div class="fv-row">
                    <div style="width:70%">
                        <span class="fv-label">EV/EBITDA Model (TTM)</span>
                        <div class="weight-container">
                            <div class="weight-bar"><div class="weight-fill" id="w_ev_ebitda_fill"></div></div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span class="fv-num" id="ev_ebitda_val">--</span>
                        <div style="font-size:10px; color:#aaa;" id="w_ev_ebitda_txt">--</div>
                    </div>
                </div>

            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <div id="chartContainer" style="height: 350px;"></div>
        </div>

        <div class="bottom-section">
            <div class="card">
                <div class="card-title">Walk-forward Backtest (1M ahead)</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Actual Price</th>
                            <th>Model</th>
                            <th>Error</th>
                        </tr>
                    </thead>
                    <tbody id="backtestBody"></tbody>
                </table>
            </div>

            <div class="card">
                <div class="card-title">Historical Returns</div>
                <div class="returns-grid" style="margin-bottom: 25px;">
                    <div class="ret-box"><div class="ret-label">1M</div><div class="ret-val" id="r1m">--</div></div>
                    <div class="ret-box"><div class="ret-label">3M</div><div class="ret-val" id="r3m">--</div></div>
                    <div class="ret-box"><div class="ret-label">6M</div><div class="ret-val" id="r6m">--</div></div>
                    <div class="ret-box"><div class="ret-label">1Y</div><div class="ret-val" id="r1y">--</div></div>
                    <div class="ret-box"><div class="ret-label">2Y</div><div class="ret-val" id="r2y">--</div></div>
                </div>

                <div class="card-title">Future Projections (DCF)</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Year</th>
                            <th>Projected FCFF</th>
                            <th>Growth</th>
                        </tr>
                    </thead>
                    <tbody id="forecastBody"></tbody>
                </table>
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
        const res = await fetch('/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker: ticker})
        });

        const text = await res.text();
        let data;
        try { data = JSON.parse(text); }
        catch (e) { throw new Error("Non-JSON response: " + text.slice(0, 200)); }

        loading.style.display = 'none';
        btn.disabled = false;

        if (data.error) {
            err.innerText = data.error;
            err.style.display = 'block';
            return;
        }

        const s = data.valuation_summary;
        const m = data.metrics;
        const r = data.returns;
        const backtest = data.backtest;
        const weights = data.optimized_weights;

        document.getElementById('name').innerText = s.company_name ?? "--";
        document.getElementById('tickerDisplay').innerText = ticker.toUpperCase() + ".SR";
        document.getElementById('price').innerText = (s.current_price ?? 0).toFixed(2);
        document.getElementById('fair').innerText = (s.fair_value ?? 0).toFixed(2);
        document.getElementById('sectorMsg').innerText = s.sector ?? "--";

        const vb = document.getElementById('verdictBar');
        const upside = s.upside_percent ?? 0;
        const label = (s.verdict ?? "Fairly Valued").toUpperCase();
        const sign = upside > 0 ? "+" : "";
        vb.innerText = `${label} (${sign}${Number(upside).toFixed(1)}% Upside)`;
        vb.className = "verdict-bar " + (label === "UNDERVALUED" ? "v-green" : (label === "OVERVALUED" ? "v-red" : "v-gray"));

        const fmt = (num) => (num === null || num === undefined) ? "N/A" : Number(num).toFixed(2);
        const fmtBig = (num) => (num === null || num === undefined) ? "N/A" : (Number(num) / 1000000000).toFixed(2) + "B";

        document.getElementById('mcap').innerText = fmtBig(m.market_cap);
        document.getElementById('pe').innerText = fmt(m.pe_ratio);
        document.getElementById('eps').innerText = fmt(m.eps);
        document.getElementById('beta').innerText = (m.beta === null || m.beta === undefined) ? "N/A" : Number(m.beta).toFixed(2);
        document.getElementById('growth').innerText = (m.growth_rate === null || m.growth_rate === undefined) ? "N/A" : (Number(m.growth_rate) * 100).toFixed(1) + "%";
        document.getElementById('book').innerText = fmt(m.book_value);

        document.getElementById('wacc_display').innerText =
            (m.wacc === null || m.wacc === undefined) ? "N/A" : (Number(m.wacc) * 100).toFixed(1) + "%";

        document.getElementById('beta_tag').innerHTML = '<span class="dyn-badge">LIVE</span>';
        document.getElementById('growth_tag').innerHTML = '<span class="dyn-badge">TTM</span>';

        document.getElementById('dcf_val').innerText = fmt(s.model_breakdown?.dcf);
        document.getElementById('pe_val').innerText = fmt(s.model_breakdown?.pe_model);
        document.getElementById('pb_val').innerText = fmt(s.model_breakdown?.pb_model);
        document.getElementById('ev_ebitda_val').innerText = fmt(s.model_breakdown?.ev_ebitda_model);

        const setW = (key, val) => {
            const pct = ((val ?? 0) * 100).toFixed(0) + "%";
            const fill = document.getElementById(`w_${key}_fill`);
            const txt = document.getElementById(`w_${key}_txt`);
            if (fill) fill.style.width = pct;
            if (txt) txt.innerText = "Weight: " + pct;
        };
        setW('dcf', weights?.dcf);
        setW('pe', weights?.pe);
        setW('pb', weights?.pb);
        setW('ev_ebitda', weights?.ev_ebitda);

        const setRet = (id, val) => {
            const el = document.getElementById(id);
            if (!el) return;
            if (val === null || val === undefined) { el.innerText = "--"; el.className = "ret-val"; return; }
            const n = Number(val);
            el.innerText = (n > 0 ? "+" : "") + n.toFixed(1) + "%";
            el.className = "ret-val " + (n > 0 ? "pos" : "neg");
        };
        setRet('r1m', r?.["1m"]);
        setRet('r3m', r?.["3m"]);
        setRet('r6m', r?.["6m"]);
        setRet('r1y', r?.["1y"]);
        setRet('r2y', r?.["2y"]);

        const fcBody = document.getElementById('forecastBody');
        if (fcBody) {
            fcBody.innerHTML = "";
            const currentYear = new Date().getFullYear();
            const projections = Array.isArray(s.dcf_projections) ? s.dcf_projections : [];
            projections.forEach((val, i) => {
                const row = `<tr>
                    <td>${currentYear + i + 1}</td>
                    <td>${Number(val).toFixed(0)} SAR</td>
                    <td style="color:#28cd41;">+${(Number(m.growth_rate || 0) * 100).toFixed(1)}%</td>
                </tr>`;
                fcBody.innerHTML += row;
            });
        }

        const btBody = document.getElementById('backtestBody');
        if (btBody) {
            btBody.innerHTML = "";
            (Array.isArray(backtest) ? backtest : []).forEach(b => {
                const actual = Number(b.actual);
                const model = Number(b.model);
                const diff = (actual && isFinite(actual)) ? Math.abs((model - actual) / actual) * 100 : 0;
                const color = diff < 15 ? "#28cd41" : "#f0ad4e";
                const row = `<tr>
                    <td>${b.period ?? ""}</td>
                    <td>${isFinite(actual) ? actual.toFixed(2) : "N/A"}</td>
                    <td>${isFinite(model) ? model.toFixed(2) : "N/A"}</td>
                    <td style="color:${color}; font-weight:bold;">${diff.toFixed(1)}%</td>
                </tr>`;
                btBody.innerHTML += row;
            });
        }

        const dates = data.historical_data?.dates || [];
        const prices = data.historical_data?.prices || [];
        const fairVals = data.historical_data?.fair_values || [];

        if (dates.length && prices.length && fairVals.length) {
            Highcharts.chart('chartContainer', {
                chart: { backgroundColor: 'transparent' },
                title: { text: 'Actual vs Model (1M-ahead forecast aligned to realized dates)' },
                xAxis: { type: 'datetime' },
                yAxis: { title: { text: null }, gridLineColor: '#eee' },
                series: [{
                    name: 'Actual Price',
                    data: dates.map((d, i) => [d, prices[i]]),
                    type: 'area'
                }, {
                    name: 'Model Forecast (1M-ahead)',
                    data: dates.map((d, i) => [d, fairVals[i]]),
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
# 4) DATA FETCHER (prices + statements)
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

        # Annual
        fin_a = safe_attr("financials")
        bs_a = safe_attr("balance_sheet")
        cf_a = safe_attr("cashflow")

        # Quarterly
        fin_q = safe_attr("quarterly_financials")
        bs_q = safe_attr("quarterly_balance_sheet")
        cf_q = safe_attr("quarterly_cashflow")

        return {
            "info": info,
            "financials_annual": fin_a,
            "balance_sheet_annual": bs_a,
            "cashflow_annual": cf_a,
            "financials_quarterly": fin_q,
            "balance_sheet_quarterly": bs_q,
            "cashflow_quarterly": cf_q,
        }

    # ---------- Risk-free (Excel) ----------
    def fetch_saudi_risk_free_from_excel(self, path: str, column_name: str) -> float:
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Failed to read Excel '{path}'. Install openpyxl and ensure the file exists. Detail: {str(e)}")

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
# 5) HELPERS (time alignment + robustness)
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

def _to_tz_naive_index(idx: pd.Index) -> pd.Index:
    # Fix "Cannot compare tz-naive and tz-aware timestamps"
    try:
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            return idx.tz_convert(None)
    except Exception:
        try:
            if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
                return idx.tz_localize(None)
        except Exception:
            pass
    return idx

def _ensure_tz_naive_series(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None or s.empty:
        return s
    try:
        si = pd.to_datetime(s.index, errors="coerce")
        si = _to_tz_naive_index(pd.DatetimeIndex(si))
        out = s.copy()
        out.index = si
        out = out[~out.index.isna()]
        return out.sort_index()
    except Exception:
        return s

def _row_lookup(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx_lower = {str(i).strip().lower(): i for i in df.index}
    for n in names:
        key = str(n).strip().lower()
        if key in idx_lower:
            return df.loc[idx_lower[key]]
    return None

def _row_contains(df: pd.DataFrame, must_contain: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    must = [m.lower() for m in must_contain]
    for idx in df.index:
        s = str(idx).lower()
        if all(m in s for m in must):
            return df.loc[idx]
    return None

def _series_from_row(df: pd.DataFrame, row_names: List[str], contains: Optional[List[str]] = None) -> Optional[pd.Series]:
    r = _row_lookup(df, row_names)
    if r is None and contains is not None:
        r = _row_contains(df, contains)
    if r is None:
        return None
    s = pd.to_numeric(r, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()
    s = _ensure_tz_naive_series(s)
    if s is None or s.empty:
        return None
    return s.sort_index()

def ttm_from_quarters(q_series: pd.Series) -> pd.Series:
    s = q_series.sort_index()
    return s.rolling(4, min_periods=4).sum()

def last_value_on_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    # forward-fill to dates based on last known report date (all tz-naive)
    if series is None or series.empty:
        return pd.Series(index=dates, dtype=float)
    s = _ensure_tz_naive_series(series)
    if s is None or s.empty:
        return pd.Series(index=dates, dtype=float)

    dates = pd.DatetimeIndex(_to_tz_naive_index(dates))
    s = s.sort_index()
    tmp_idx = s.index.union(dates)
    tmp = s.reindex(tmp_idx).sort_index().ffill()
    return pd.Series(index=dates, data=tmp.reindex(dates).values, dtype=float)

def winsorize(arr: np.ndarray, p_low=0.05, p_high=0.95) -> np.ndarray:
    x = arr.copy()
    m = np.isfinite(x)
    if m.sum() == 0:
        return arr
    lo = np.quantile(x[m], p_low)
    hi = np.quantile(x[m], p_high)
    out = x.copy()
    out[m] = np.clip(out[m], lo, hi)
    return out

def safe_div(a, b) -> float:
    if a is None or b is None:
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a / b)

# =========================================================
# 6) MARKET/BETA HELPERS
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
# 7) VALUATION MODELS (DCF + self-anchored multiples)
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
    n_models = X.shape[0]
    if not np.any(avail):
        raise ValueError("No models available")

    idx = np.where(avail)[0]
    k = len(idx)

    best_w_full = np.zeros(n_models, dtype=float)
    best_loss = float("inf")

    rnd = np.random.default_rng(42)

    candidates = []
    for j in idx:
        w = np.zeros(n_models, dtype=float)
        w[j] = 1.0
        candidates.append(w)

    alpha = np.ones(k, dtype=float)
    draws = rnd.dirichlet(alpha, size=n_samples)
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
# 8) SPREAD ENGINE (forecast delta% 1M ahead via ridge)
# =========================================================
def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, lam: float) -> float:
    # closed-form ridge with intercept
    X = X_train
    y = y_train.reshape(-1, 1)
    n, p = X.shape

    X1 = np.hstack([np.ones((n, 1)), X])
    p1 = p + 1

    I = np.eye(p1)
    I[0, 0] = 0.0  # don't penalize intercept

    A = X1.T @ X1 + lam * I
    b = X1.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b

    Xp = np.hstack([np.ones((X_pred.shape[0], 1)), X_pred])
    yp = Xp @ w
    return float(yp.ravel()[0])

def build_spread_features(
    df: pd.DataFrame,
    shares: pd.Series,
) -> pd.DataFrame:
    # df columns: Close, Volume, MktClose, V_anchor
    out = pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)
    mkt = df["MktClose"].astype(float)

    ret1 = np.log(close).diff()
    mret1 = np.log(mkt).diff()

    # momentum (log returns sums)
    out["mom_5"] = ret1.rolling(5).sum()
    out["mom_21"] = ret1.rolling(21).sum()
    out["mom_63"] = ret1.rolling(63).sum()
    out["mom_126"] = ret1.rolling(126).sum()

    out["m_mom_21"] = mret1.rolling(21).sum()
    out["m_mom_63"] = mret1.rolling(63).sum()

    # vol
    out["vol_21"] = ret1.rolling(21).std()
    out["vol_63"] = ret1.rolling(63).std()
    out["m_vol_21"] = mret1.rolling(21).std()
    out["m_vol_63"] = mret1.rolling(63).std()

    # drawdown
    roll_max = close.rolling(252).max()
    out["dd_252"] = (close / roll_max) - 1.0
    m_roll_max = mkt.rolling(252).max()
    out["m_dd_252"] = (mkt / m_roll_max) - 1.0

    # liquidity proxies
    vol = df.get("Volume")
    if vol is not None:
        vol = vol.astype(float).replace(0.0, np.nan)
        # Amihud illiquidity ~ |ret| / (price * volume)
        out["amihud_21"] = (ret1.abs() / (close * vol)).rolling(21).mean()
        # turnover ~ volume / shares
        sh = shares.astype(float).replace(0.0, np.nan)
        out["turnover_21"] = (vol / sh).rolling(21).mean()
    else:
        out["amihud_21"] = np.nan
        out["turnover_21"] = np.nan

    # valuation gap signals (do not include future info)
    V = df["V_anchor"].astype(float)
    out["gap_pct"] = (close - V) / V.replace(0.0, np.nan)
    out["gap_z_63"] = (out["gap_pct"] - out["gap_pct"].rolling(63).mean()) / out["gap_pct"].rolling(63).std()

    # beta instability proxy (rolling corr * vol ratio)
    corr_126 = ret1.rolling(126).corr(mret1)
    beta_126 = corr_126 * (out["vol_63"] / out["m_vol_63"].replace(0.0, np.nan))
    out["beta_proxy_126"] = beta_126

    # clean
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def walk_forward_spread_forecast(
    df_feat: pd.DataFrame,
    y_target: pd.Series,
    horizon: int,
    train_days: int,
    test_days: int,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    y_target indexed by date, represents delta% at date (t+h) (i.e. shifted)
    We predict y_target using features at date t, and align predictions to the realized date (t+h).
    """
    idx = df_feat.index
    preds = pd.Series(index=idx, dtype=float)

    # choose a stable subset of features
    Xall = df_feat.copy()
    # winsorize numeric columns to reduce regime spikes dominating
    for c in Xall.columns:
        x = Xall[c].to_numpy(dtype=float)
        Xall[c] = winsorize(x, 0.02, 0.98)

    # rolling walk-forward: for each prediction date t, train on [t-train_days, t)
    # and predict y at date t (which actually corresponds to realized date t in y_target index)
    # BUT: y_target is already shifted to realized dates; we must use features from (date - horizon)
    # We'll construct aligned matrices: use X at (t-horizon) to predict y at t.
    # So prediction is for realized date t, using available features at t-horizon.

    # Build aligned frame
    X_lag = Xall.shift(horizon)
    aligned = pd.concat([X_lag, y_target.rename("y")], axis=1).dropna()
    if len(aligned) < FEATURE_MIN_ROWS:
        return preds, []

    dates = aligned.index

    # define test region: last test_days of aligned samples (approx)
    test_start = dates[-min(test_days, len(dates))]

    backtest_rows: List[Dict[str, Any]] = []

    # iterate through test dates at step SOLVER_SAMPLE_STEP for table + chart
    for t in dates:
        if t < test_start:
            continue

        # training window ends at t (exclusive)
        train_start = t - pd.Timedelta(days=int(train_days * 1.6))  # trading->calendar cushion
        train_slice = aligned.loc[train_start:t].iloc[:-1]  # exclude current t for strict OOS
        if len(train_slice) < 150:
            continue

        X_train = train_slice[Xall.columns].to_numpy(dtype=float)
        y_train = train_slice["y"].to_numpy(dtype=float)

        # prediction row at date t uses lagged features already aligned
        row = aligned.loc[[t]]
        X_pred = row[Xall.columns].to_numpy(dtype=float)

        # standardize using train stats (avoid leakage)
        mu = np.nanmean(X_train, axis=0)
        sd = np.nanstd(X_train, axis=0)
        sd = np.where(sd == 0, 1.0, sd)

        X_train_z = (X_train - mu) / sd
        X_pred_z = (X_pred - mu) / sd

        # choose lambda by simple internal holdout (last 20% of training)
        n = len(y_train)
        split = int(n * 0.8)
        X_tr, y_tr = X_train_z[:split], y_train[:split]
        X_va, y_va = X_train_z[split:], y_train[split:]

        best_lam = RIDGE_LAMBDAS[0]
        best_loss = float("inf")
        for lam in RIDGE_LAMBDAS:
            # predict validation in batch by fitting once and applying
            # We'll reuse ridge_fit_predict with vector loop (small validation sizes)
            # More stable: fit once, compute weights, then eval
            X1 = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
            y1 = y_tr.reshape(-1, 1)
            I = np.eye(X1.shape[1])
            I[0, 0] = 0.0
            A = X1.T @ X1 + lam * I
            b = X1.T @ y1
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(A) @ b
            Xv1 = np.hstack([np.ones((X_va.shape[0], 1)), X_va])
            yhat = (Xv1 @ w).ravel()
            loss = float(np.mean(np.abs(yhat - y_va)))
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_lam = lam

        yhat_t = ridge_fit_predict(X_train_z, y_train, X_pred_z, best_lam)
        preds.loc[t] = yhat_t

    # Build backtest table on a few anchor periods (nearest trading day)
    # Here "Period" references the realized date (t), the prediction was made about horizon days earlier.
    # We'll report 3M/6M/1Y/TestStart by selecting realized dates.
    realized = aligned.index
    if len(realized) >= 50:
        def nearest_date(target: pd.Timestamp) -> Optional[pd.Timestamp]:
            r = realized[realized <= target]
            return r[-1] if len(r) else None

        today = realized[-1]
        picks = [
            ("3 Months Ago (OOS)", today - pd.Timedelta(days=92)),
            ("6 Months Ago (OOS)", today - pd.Timedelta(days=183)),
            ("1 Year Ago (OOS)", today - pd.Timedelta(days=365)),
            ("Test Start (OOS)", test_start),
        ]
        for label, dt in picks:
            d = nearest_date(dt) if label != "Test Start (OOS)" else test_start
            if d is None:
                continue
            if not np.isfinite(preds.loc[d]):
                continue
            # actual price on realized date
            # user-facing backtest: compare price at realized date vs model forecast price
            backtest_rows.append({"period": label, "date": d})

    return preds, backtest_rows

# =========================================================
# 9) REQUEST MODEL
# =========================================================
class StockRequest(BaseModel):
    ticker: str

# =========================================================
# 10) MAIN ANALYSIS ENDPOINT
# =========================================================
@app.post("/analyze")
def analyze_stock(request: StockRequest):
    try:
        fetcher = DataFetcher()
        ticker = fetcher.clean_saudi_ticker(request.ticker)

        # ---------- Prices (stock + market index) ----------
        hist, source_stock = fetcher.fetch_prices(ticker, period=DEFAULT_HISTORY_PERIOD)
        mkt_hist, source_mkt = fetcher.fetch_prices(TASI_TICKER, period=DEFAULT_HISTORY_PERIOD)

        if hist is None or hist.empty or "Close" not in hist.columns or hist["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for stock."}, status_code=200)
        if mkt_hist is None or mkt_hist.empty or "Close" not in mkt_hist.columns or mkt_hist["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for market index (^TASI.SR)."}, status_code=200)

        # normalize tz
        hist = hist.copy()
        mkt_hist = mkt_hist.copy()
        hist.index = pd.DatetimeIndex(pd.to_datetime(hist.index, errors="coerce"))
        mkt_hist.index = pd.DatetimeIndex(pd.to_datetime(mkt_hist.index, errors="coerce"))
        hist.index = _to_tz_naive_index(pd.DatetimeIndex(hist.index))
        mkt_hist.index = _to_tz_naive_index(pd.DatetimeIndex(mkt_hist.index))
        hist = hist[~hist.index.isna()].sort_index()
        mkt_hist = mkt_hist[~mkt_hist.index.isna()].sort_index()

        stock_close_raw = hist["Close"].astype(float).dropna()
        mkt_close_raw = mkt_hist["Close"].astype(float).dropna()

        aligned_px = pd.DataFrame({"stock": stock_close_raw, "mkt": mkt_close_raw}).dropna()
        if len(aligned_px) < 300:
            return JSONResponse({"error": "Not enough overlapping price history between stock and TASI."}, status_code=200)

        stock_close = aligned_px["stock"]
        mkt_close = aligned_px["mkt"]
        dates = pd.DatetimeIndex(stock_close.index)
        dates = pd.DatetimeIndex(_to_tz_naive_index(dates))

        current_price = float(stock_close.iloc[-1])

        # keep volume if available
        vol_series = None
        if "Volume" in hist.columns:
            vol_series = hist["Volume"].reindex(dates).astype(float)

        prices_list = stock_close.tolist()
        dates_ms = (dates.view("int64") // 10**6).astype(int).tolist()

        # ---------- Statements ----------
        pack = fetcher.fetch_statements_yahoo(ticker)
        info = pack.get("info") or {}

        fin_q = pack.get("financials_quarterly")
        bs_q = pack.get("balance_sheet_quarterly")
        cf_q = pack.get("cashflow_quarterly")

        fin_a = pack.get("financials_annual")
        bs_a = pack.get("balance_sheet_annual")
        cf_a = pack.get("cashflow_annual")

        company_name = info.get("longName") or f"Saudi Stock {request.ticker}"
        sector = (info.get("sector") or "Unknown").title()

        # Shares & market cap
        shares_now = _to_float(info.get("sharesOutstanding"))
        mcap_now = _to_float(info.get("marketCap"))
        if mcap_now is None and shares_now is not None:
            mcap_now = shares_now * current_price
        if mcap_now is None or shares_now is None or shares_now <= 0:
            return JSONResponse({"error": "Missing/invalid sharesOutstanding or marketCap from statements source."}, status_code=200)

        # ---------- Risk-free (Excel) ----------
        try:
            rf = fetcher.fetch_saudi_risk_free_from_excel(RISK_FREE_XLSX_PATH, RISK_FREE_COLUMN_NAME)
            rf_method = "excel_10y"
        except Exception as e:
            return JSONResponse({"error": f"Could not fetch Saudi risk-free rate from Excel: {str(e)}"}, status_code=200)

        # ---------- Market return + ERP (data-driven from TASI) ----------
        try:
            mkt_tail = mkt_close.tail(min(MARKET_RETURN_LOOKBACK_DAYS, len(mkt_close)))
            rm_exp = annualized_geo_mean_return(mkt_tail)
            erp = rm_exp - rf
            if not np.isfinite(erp):
                raise ValueError("ERP not finite")
        except Exception as e:
            return JSONResponse({"error": f"Could not compute market return/ERP from TASI: {str(e)}"}, status_code=200)

        # ---------- Beta (regression vs TASI) ----------
        try:
            s_beta = stock_close.tail(min(BETA_LOOKBACK_DAYS, len(stock_close)))
            m_beta = mkt_close.tail(min(BETA_LOOKBACK_DAYS, len(mkt_close)))
            beta = beta_regression(s_beta, m_beta)
            beta_method = "regression_log_returns"
        except Exception as e:
            return JSONResponse({"error": f"Could not compute regression beta vs TASI: {str(e)}"}, status_code=200)

        # ---------- Cost of equity ----------
        Re = rf + beta * erp

        method_flags = {
            "rf": rf_method,
            "beta": beta_method,
            "fundamentals": "quarterly_ttm" if (isinstance(fin_q, pd.DataFrame) and not fin_q.empty) else "annual_fallback",
            "prices_source_stock": source_stock,
            "prices_source_market": source_mkt,
            "walk_forward": f"train={TRAIN_WINDOW_DAYS}d,test={TEST_WINDOW_DAYS}d,h={SPREAD_HORIZON_DAYS}d",
        }

        # ---- Fundamental series (prefer quarterly -> TTM) ----
        ni_q = _series_from_row(fin_q, ["Net Income", "NetIncome"], contains=["net", "income"]) if isinstance(fin_q, pd.DataFrame) else None
        eq_q = _series_from_row(bs_q, ["Total Stockholder Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"], contains=["total", "equity"]) if isinstance(bs_q, pd.DataFrame) else None
        cfo_q = _series_from_row(cf_q, ["Total Cash From Operating Activities", "Operating Cash Flow"], contains=["operating", "cash"]) if isinstance(cf_q, pd.DataFrame) else None
        capex_q = _series_from_row(cf_q, ["Capital Expenditures", "Capital Expenditure"], contains=["capital", "expend"]) if isinstance(cf_q, pd.DataFrame) else None
        da_q = _series_from_row(cf_q, ["Depreciation", "Depreciation And Amortization"], contains=["depreciation"]) if isinstance(cf_q, pd.DataFrame) else None
        ebit_q = _series_from_row(fin_q, ["Operating Income", "OperatingIncome", "EBIT", "Ebit"], contains=["operating", "income"]) if isinstance(fin_q, pd.DataFrame) else None

        st_debt_q = _series_from_row(bs_q, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], contains=["short", "debt"]) if isinstance(bs_q, pd.DataFrame) else None
        lt_debt_q = _series_from_row(bs_q, ["Long Term Debt", "LongTermDebt"], contains=["long", "debt"]) if isinstance(bs_q, pd.DataFrame) else None
        cash_q = _series_from_row(bs_q, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents"], contains=["cash"]) if isinstance(bs_q, pd.DataFrame) else None

        shares_q = _series_from_row(
            bs_q,
            ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
            contains=["shares"],
        ) if isinstance(bs_q, pd.DataFrame) else None

        # fallback to annual if needed
        if ni_q is None and isinstance(fin_a, pd.DataFrame) and not fin_a.empty:
            ni_q = _series_from_row(fin_a, ["Net Income", "NetIncome"], contains=["net", "income"])
        if eq_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            eq_q = _series_from_row(bs_a, ["Total Stockholder Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"], contains=["total", "equity"])
        if cfo_q is None and isinstance(cf_a, pd.DataFrame) and not cf_a.empty:
            cfo_q = _series_from_row(cf_a, ["Total Cash From Operating Activities", "Operating Cash Flow"], contains=["operating", "cash"])
        if capex_q is None and isinstance(cf_a, pd.DataFrame) and not cf_a.empty:
            capex_q = _series_from_row(cf_a, ["Capital Expenditures", "Capital Expenditure"], contains=["capital", "expend"])
        if da_q is None and isinstance(cf_a, pd.DataFrame) and not cf_a.empty:
            da_q = _series_from_row(cf_a, ["Depreciation", "Depreciation And Amortization"], contains=["depreciation"])
        if ebit_q is None and isinstance(fin_a, pd.DataFrame) and not fin_a.empty:
            ebit_q = _series_from_row(fin_a, ["Operating Income", "OperatingIncome", "EBIT", "Ebit"], contains=["operating", "income"])
        if st_debt_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            st_debt_q = _series_from_row(bs_a, ["Short Long Term Debt", "Short Term Debt", "Current Debt"], contains=["short", "debt"])
        if lt_debt_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            lt_debt_q = _series_from_row(bs_a, ["Long Term Debt", "LongTermDebt"], contains=["long", "debt"])
        if cash_q is None and isinstance(bs_a, pd.DataFrame) and not bs_a.empty:
            cash_q = _series_from_row(bs_a, ["Cash", "Cash And Cash Equivalents", "CashAndCashEquivalents"], contains=["cash"])

        # ---- Effective tax rate (best-effort from annual; if missing -> 0) ----
        T = 0.0
        try:
            pretax_a = _series_from_row(fin_a, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax"], contains=["before", "tax"]) if isinstance(fin_a, pd.DataFrame) else None
            tax_a = _series_from_row(fin_a, ["Tax Provision", "Income Tax Expense", "IncomeTaxExpense"], contains=["tax"]) if isinstance(fin_a, pd.DataFrame) else None
            if pretax_a is not None and tax_a is not None and pretax_a.dropna().size > 0 and tax_a.dropna().size > 0:
                px = float(pretax_a.dropna().iloc[-1])
                tx = float(tax_a.dropna().iloc[-1])
                if np.isfinite(px) and px > 0 and np.isfinite(tx):
                    T = max(0.0, min(tx / px, 0.35))
        except Exception:
            pass

        # ---- Time-varying net debt -> daily ----
        st_debt_q = st_debt_q if st_debt_q is not None else pd.Series(dtype=float)
        lt_debt_q = lt_debt_q if lt_debt_q is not None else pd.Series(dtype=float)
        cash_q = cash_q if cash_q is not None else pd.Series(dtype=float)

        debt_q = (st_debt_q.fillna(0.0) + lt_debt_q.fillna(0.0)).sort_index()
        cash_q = cash_q.fillna(0.0).sort_index()
        net_debt_q = (debt_q - cash_q).sort_index()
        net_debt_daily = last_value_on_or_before(net_debt_q, dates)

        # ---- Shares daily (best-effort) ----
        if shares_q is not None and shares_q.dropna().size >= 2:
            shares_daily = last_value_on_or_before(shares_q, dates)
            if not np.isfinite(shares_daily.dropna().median()) or shares_daily.dropna().median() <= 0:
                shares_daily = pd.Series(index=dates, data=float(shares_now))
                method_flags["shares"] = "constant_info"
            else:
                method_flags["shares"] = "report_aligned_best_effort"
        else:
            shares_daily = pd.Series(index=dates, data=float(shares_now))
            method_flags["shares"] = "constant_info"

        # ---- EPS TTM daily ----
        if ni_q is not None and ni_q.dropna().size >= 4:
            ni_ttm = ttm_from_quarters(ni_q)
            ni_ttm_daily = last_value_on_or_before(ni_ttm, dates)
            eps_ttm_daily = ni_ttm_daily / shares_daily.replace(0.0, np.nan)
        else:
            # fallback: info trailingEps if present (constant)
            trailing_eps = _to_float(info.get("trailingEps"))
            eps_ttm_daily = pd.Series(index=dates, data=(trailing_eps if trailing_eps is not None else np.nan), dtype=float)

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
            capex_out = capex_ttm_daily.copy()
            capex_out = np.where(np.isfinite(capex_out.values), capex_out.values, np.nan)
            capex_out = np.where(capex_out < 0, -capex_out, capex_out)

            fcff_ttm_daily = pd.Series(index=dates, data=(cfo_ttm_daily.values - capex_out), dtype=float)
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

        # ---- WACC (data-minimal): cost of debt proxy = rf, weights from E vs D (time-varying net debt) ----
        # This is deliberately conservative: if we cannot infer interest rate reliably, we don't invent it.
        D = net_debt_daily.clip(lower=0.0)
        E = pd.Series(index=dates, data=float(mcap_now), dtype=float)  # best-effort constant market cap
        Vcap = (D + E).replace(0.0, np.nan)
        wd = (D / Vcap).clip(0.0, 0.95)
        we = (E / Vcap).clip(0.05, 1.0)
        Rd = rf  # proxy when interest expense isn't reliably extractable
        wacc_daily = (we * Re + wd * Rd * (1.0 - T)).clip(0.0, WACC_MAX)
        method_flags["wacc"] = "wacc_equity_capm_debt_rf_proxy"

        # ---- Build observed multiples series ----
        close_series = pd.Series(index=dates, data=stock_close.values, dtype=float)

        pe_obs = close_series / eps_ttm_daily.replace(0.0, np.nan)
        pb_obs = close_series / bvps_daily.replace(0.0, np.nan)

        ev_daily = (close_series * shares_daily.replace(0.0, np.nan)) + net_debt_daily
        ev_ebitda_obs = ev_daily / ebitda_ttm_daily.replace(0.0, np.nan)

        # ---- Self-anchored target multiples (rolling median of own history) ----
        def rolling_target_multiple(obs: pd.Series, window: int = TRADING_DAYS * 2) -> pd.Series:
            m = obs.copy()
            m = m.replace([np.inf, -np.inf], np.nan)
            m = m.where(m > 0)  # multiples must be positive to be meaningful
            # winsorize within rolling window by clipping to rolling quantiles
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
        market_long_run_g = rm_exp  # data-driven cap from TASI CAGR
        if fcff_ttm_daily.dropna().size > 200:
            # per-day DCF uses current TTM FCFF as fcff0; per-share via shares, net debt, wacc, growth
            # NOTE: if fcff0 <=0 for some dates, those dates remain NaN.
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
                        dcf_vals.append(dcf_ps)
                    else:
                        dcf_vals.append(np.nan)
                except Exception:
                    dcf_vals.append(np.nan)
            dcf_model = pd.Series(index=dates, data=dcf_vals, dtype=float)

        # ---- Build valuation matrix X (models) ----
        models = {
            "dcf": dcf_model,
            "pe": pe_model,
            "pb": pb_model,
            "ev_ebitda": ev_ebitda_model,
        }

        X = np.vstack([models["dcf"].values, models["pe"].values, models["pb"].values, models["ev_ebitda"].values])
        avail = np.array([np.isfinite(models["dcf"]).sum() > 150,
                          np.isfinite(models["pe"]).sum() > 150,
                          np.isfinite(models["pb"]).sum() > 150,
                          np.isfinite(models["ev_ebitda"]).sum() > 150], dtype=bool)

        # If literally none: fail clearly (should be rare now because P/B or P/E usually works)
        if not np.any(avail):
            return JSONResponse({"error": "No valuation models available: insufficient EPS/BV/EBITDA/FCFF coverage from source."}, status_code=200)

        # ---- Train valuation weights on past (avoid future) ----
        # We want a stable anchor V_t, not perfect fit. Train weights by minimizing MAPE vs price on training window.
        # Use only dates where at least one model exists.
        # We'll fit on last TRAIN_WINDOW_DAYS trading days.
        y_all = close_series.values.astype(float)
        n = len(dates)
        train_start_idx = max(0, n - TRAIN_WINDOW_DAYS)
        y_train = y_all[train_start_idx:]
        X_train = X[:, train_start_idx:]

        # model gating per point: if a model is NaN at that date, it just won't contribute due to nansum
        # Still, we avoid selecting a model that never exists overall via avail above.
        try:
            w_val = optimize_weights_dirichlet(y_train, X_train, avail, n_samples=N_WEIGHT_SAMPLES)
        except Exception:
            # fallback: equal weight over available
            w_val = np.zeros(4, dtype=float)
            idxs = np.where(avail)[0]
            w_val[idxs] = 1.0 / len(idxs)

        # valuation anchor V_t (spot)
        V_anchor = np.nansum((X.T * w_val), axis=1)
        V_anchor = pd.Series(index=dates, data=V_anchor, dtype=float)

        # =========================================================
        # SPREAD ENGINE: forecast delta% 1M ahead
        # =========================================================
        df_core = pd.DataFrame(index=dates)
        df_core["Close"] = close_series
        df_core["MktClose"] = pd.Series(index=dates, data=mkt_close.reindex(dates).values, dtype=float)
        df_core["V_anchor"] = V_anchor
        if vol_series is not None:
            df_core["Volume"] = vol_series

        feat = build_spread_features(df_core, shares_daily)

        # target is delta% at realized date (t) predicted from features at (t-h)
        delta_pct = (df_core["Close"] - df_core["V_anchor"]) / df_core["V_anchor"].replace(0.0, np.nan)
        y_target = delta_pct.shift(-0)  # realized delta% at the realized date index
        # Walk-forward returns predictions aligned to realized dates using features from (t-horizon)
        preds_delta, backtest_meta = walk_forward_spread_forecast(
            df_feat=feat,
            y_target=y_target,
            horizon=SPREAD_HORIZON_DAYS,
            train_days=TRAIN_WINDOW_DAYS,
            test_days=TEST_WINDOW_DAYS,
        )

        # Build predicted 1M-ahead price series aligned to realized dates:
        # For a realized date t, we use V at (t-h) with predicted delta at t
        V_lag = V_anchor.shift(SPREAD_HORIZON_DAYS)
        P_hat_realized = V_lag * (1.0 + preds_delta)

        # Today's forecast (for today+1M): use latest available features to predict delta at future date,
        # but our preds_delta is aligned to realized dates. For a simple "forward" forecast,
        # we use the latest trained model implicitly via last available preds_delta at last index.
        # Here we approximate: fair_value = V_today * (1 + delta_hat_next), where delta_hat_next is last preds_delta (learned mapping).
        # More strictly: produce forecast for t+H by training on latest aligned and predicting y at t+H isn't computed above.
        # We approximate delta_hat_next by the most recent fitted delta prediction (stable, not perfect).
        delta_hat_now = float(preds_delta.dropna().iloc[-1]) if preds_delta.dropna().size else float(delta_pct.dropna().iloc[-1])
        V_now = float(V_anchor.iloc[-1]) if np.isfinite(V_anchor.iloc[-1]) else float(current_price)
        fair_value_1m = V_now * (1.0 + delta_hat_now)

        # Backtest rows: compare P_hat_realized vs actual Close at same realized date
        backtest = []
        for row in backtest_meta:
            d = row["date"]
            actual = float(df_core.loc[d, "Close"]) if d in df_core.index else np.nan
            modelv = float(P_hat_realized.loc[d]) if d in P_hat_realized.index else np.nan
            if np.isfinite(actual) and np.isfinite(modelv):
                backtest.append({"period": row["period"], "actual": actual, "model": modelv})

        # Historical data for chart: use the overlapping non-NaN region
        # We plot actual vs P_hat_realized. If P_hat_realized is sparse early, that is expected.
        fair_series_for_chart = P_hat_realized.reindex(dates).astype(float)
        fair_values_list = fair_series_for_chart.fillna(method="ffill").fillna(method="bfill").tolist()

        # Returns
        def pct_return(series: pd.Series, days: int) -> Optional[float]:
            if series.dropna().size < days + 1:
                return None
            a = float(series.iloc[-1])
            b = float(series.iloc[-(days + 1)])
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

        # Present-day headline multiples (from last available)
        eps_now = float(eps_ttm_daily.iloc[-1]) if np.isfinite(eps_ttm_daily.iloc[-1]) else np.nan
        bvps_now = float(bvps_daily.iloc[-1]) if np.isfinite(bvps_daily.iloc[-1]) else np.nan
        pe_now = safe_div(current_price, eps_now) if np.isfinite(eps_now) and eps_now != 0 else np.nan
        book_value_now = (bvps_now * float(shares_daily.iloc[-1])) if np.isfinite(bvps_now) else np.nan

        # Model breakdown at "now" (spot models, not spread forecast)
        model_breakdown = {
            "dcf": float(dcf_model.iloc[-1]) if np.isfinite(dcf_model.iloc[-1]) else None,
            "pe_model": float(pe_model.iloc[-1]) if np.isfinite(pe_model.iloc[-1]) else None,
            "pb_model": float(pb_model.iloc[-1]) if np.isfinite(pb_model.iloc[-1]) else None,
            "ev_ebitda_model": float(ev_ebitda_model.iloc[-1]) if np.isfinite(ev_ebitda_model.iloc[-1]) else None,
        }

        # DCF projections (display only): using latest fcff and growth
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
            if upside > 8:
                verdict = "Undervalued"
            elif upside < -8:
                verdict = "Overvalued"
            else:
                verdict = "Fairly Valued"
        else:
            verdict = "Fairly Valued"
            upside = 0.0

        response = {
            "valuation_summary": {
                "company_name": company_name,
                "sector": sector,
                "current_price": current_price,
                "fair_value": float(fair_value_1m) if np.isfinite(fair_value_1m) else None,
                "upside_percent": float(upside) if np.isfinite(upside) else None,
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

    except Exception as e:
        return JSONResponse({"error": f"Unhandled error: {str(e)}"}, status_code=200)
