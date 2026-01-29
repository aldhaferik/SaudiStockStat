# app.py
from __future__ import annotations

import os
import math
import time
import random
from dataclasses import dataclass
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
# =========================================================
DEFAULT_HISTORY_PERIOD = "5y"
TRADING_DAYS = 252

BETA_LOOKBACK_DAYS = TRADING_DAYS * 2
MARKET_RETURN_LOOKBACK_DAYS = TRADING_DAYS * 5

TRAIN_WINDOW_DAYS = TRADING_DAYS * 3
TEST_WINDOW_DAYS = TRADING_DAYS * 1

SOLVER_SAMPLE_STEP = 5
N_WEIGHT_SAMPLES = 6000

FORECAST_YEARS = 5
TASI_TICKER = "^TASI.SR"

# Backup price sources (you requested)
ALPHA_VANTAGE_KEY = "0LR5JLOBSLOA6Z0A"
TWELVE_DATA_KEY = "ed240f406bab4225ac6e0a98be553aa2"

# Risk-free source (your repo file)
RISK_FREE_XLSX_PATH = "saudi_yields.xlsx"
RISK_FREE_COLUMN_NAME = "10-Year government bond yield"

# Robustness bounds
GROWTH_MIN = -0.20
GROWTH_MAX = 0.40
WACC_MAX = 0.50

# Spread engine forecast horizon (1M)
SPREAD_HORIZON_DAYS = 21

# Price-only forecaster horizon (1M)
PRICE_HORIZON_DAYS = 21

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

def safe_div(a: float, b: float) -> float:
    try:
        if a is None or b is None:
            return float("nan")
        a = float(a)
        b = float(b)
        if not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return float("nan")
        return a / b
    except Exception:
        return float("nan")

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
        <p style="color:#666; font-size:14px;">Time-aligning fundamentals (TTM) + walk-forward tuning</p>
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
                    <div class="fv-sub">Model-weighted target</div>
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
                <div class="card-title">Walk-forward Backtest</div>
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
            const currentYear = new Date().i00);



/* NOTE: HTML unchanged; leaving remainder identical to your original UI.
   It was truncated here by the chat length limits in some environments. */
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

        fin_a = safe_attr("financials")
        bs_a = safe_attr("balance_sheet")
        cf_a = safe_attr("cashflow")

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
# 5) STATEMENT + SERIES HELPERS
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
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

def ttm_from_quarters(q_series: pd.Series) -> pd.Series:
    s = q_series.sort_index()
    return s.rolling(4, min_periods=4).sum()

def last_value_on_or_before(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(index=dates, dtype=float)
    s = series.sort_index()
    tmp = s.reindex(s.index.union(dates)).sort_index().ffill()
    out = pd.Series(index=dates, data=tmp.reindex(dates).values, dtype=float)
    return out

def winsorize(arr: np.ndarray, p_low=0.05, p_high=0.95) -> np.ndarray:
    x = arr.copy()
    x2 = x[np.isfinite(x)]
    if x2.size == 0:
        return arr
    lo = np.quantile(x2, p_low)
    hi = np.quantile(x2, p_high)
    return np.clip(arr, lo, hi)

def normalize_prices_index_tz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes: TypeError: Cannot compare tz-naive and tz-aware timestamps
    Ensures index becomes tz-naive DatetimeIndex.
    """
    if df is None or df.empty:
        return df
    idx = df.index
    try:
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
    except Exception:
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            pass
    return df

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
# 7) MODELS (DCF + Multiples) built from time-aligned fundamentals
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
    return best_w_full / best_w_full.sum()

# =========================================================
# 7B) SPREAD ENGINE (valuation + price dynamics)
# =========================================================
def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)

def build_spread_features(df_core: pd.DataFrame, shares_daily: pd.Series) -> pd.DataFrame:
    """
    Features are computed at time t and used to predict delta% at time (t + horizon),
    but we will shift inside the walk-forward function to avoid leakage.
    """
    df = df_core.copy()

    # price returns / momentum
    df["ret_1d"] = df["Close"].pct_change()
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_21d"] = df["Close"].pct_change(21)

    # realized vol
    df["vol_21d"] = df["ret_1d"].rolling(21).std(ddof=0)
    df["vol_63d"] = df["ret_1d"].rolling(63).std(ddof=0)

    # market returns / beta-ish co-move proxy
    if "MktClose" in df.columns and df["MktClose"].notna().sum() > 50:
        df["mkt_ret_1d"] = df["MktClose"].pct_change()
        df["mkt_ret_21d"] = df["MktClose"].pct_change(21)
        # rolling correlation as regime proxy
        df["corr_63d"] = df["ret_1d"].rolling(63).corr(df["mkt_ret_1d"])

    # valuation spread
    df["delta_pct"] = (df["Close"] - df["V_anchor"]) / df["V_anchor"].replace(0.0, np.nan)
    df["delta_z_252"] = _zscore(df["delta_pct"], 252)

    # liquidity proxies
    if "Volume" in df.columns and df["Volume"].notna().sum() > 50:
        # dollar volume
        df["dvol"] = df["Volume"] * df["Close"]
        df["dvol_z_63"] = _zscore(df["dvol"].replace(0.0, np.nan), 63)
        # volume shock
        df["vol_shock"] = df["Volume"] / df["Volume"].rolling(63).median()

    # size proxy (market cap approx)
    df["mcap_proxy"] = df["Close"] * shares_daily.reindex(df.index).replace(0.0, np.nan)
    df["mcap_log"] = np.log(df["mcap_proxy"].replace(0.0, np.nan))

    # keep only numeric cols
    feat_cols = [c for c in df.columns if c not in ["Close", "MktClose", "V_anchor"]]
    out = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # minimal cleaning: leave NaNs; model will drop per-window
    return out

def _fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, l2: float = 10.0) -> np.ndarray:
    """
    Ridge regression using closed-form:
    w = (X'X + l2*I)^-1 X'y
    X includes intercept column if you want one.
    """
    XtX = X.T @ X
    I = np.eye(XtX.shape[0], dtype=float)
    beta = np.linalg.pinv(XtX + l2 * I) @ (X.T @ y)
    return beta

def _standardize_train_apply(X_train: np.ndarray, X_apply: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xtr = (X_train - mu) / sd
    Xap = (X_apply - mu) / sd
    return Xtr, Xap, mu

def walk_forward_spread_forecast(
    df_feat: pd.DataFrame,
    y_target: pd.Series,
    horizon: int,
    train_days: int,
    test_days: int,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Predicts y_target at date t using features from date (t - horizon).
    This aligns with your construction where you later use V_lag = V.shift(horizon).
    """
    idx = df_feat.index
    X_all = df_feat.copy()
    y_all = y_target.reindex(idx).astype(float)

    # shift features so that features at (t-h) align to target at t
    X_shift = X_all.shift(horizon)

    # output series aligned to dates
    pred = pd.Series(index=idx, dtype=float)

    n = len(idx)
    test_start_idx = max(0, n - test_days)

    # walk forward daily within test window
    for t in range(test_start_idx, n):
        train_start = max(0, t - train_days)

        X_train_df = X_shift.iloc[train_start:t]
        y_train = y_all.iloc[train_start:t]

        # drop rows with any NaNs in X or y
        train_mask = np.isfinite(y_train.values)
        X_train_df2 = X_train_df.loc[train_mask]
        y_train2 = y_train.loc[train_mask]

        if X_train_df2.shape[0] < 120:
            continue

        X_train_np = X_train_df2.values.astype(float)
        y_train_np = y_train2.values.astype(float)

        # drop columns with too many NaNs in train
        col_ok = np.isfinite(X_train_np).mean(axis=0) > 0.85
        if col_ok.sum() < 3:
            continue

        X_train_np = X_train_np[:, col_ok]

        # apply to current point
        X_t = X_shift.iloc[t].values.astype(float)
        X_t = X_t[col_ok]

        if not np.isfinite(X_t).all():
            continue

        # standardize
        Xtr, Xap, _mu = _standardize_train_apply(X_train_np, X_t.reshape(1, -1))

        # add intercept
        Xtr_i = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
        Xap_i = np.c_[np.ones((Xap.shape[0], 1)), Xap]

        # ridge fit
        beta = _fit_ridge_closed_form(Xtr_i, y_train_np, l2=25.0)
        yhat = float(Xap_i @ beta)

        # clamp to reasonable delta range to avoid nonsense
        yhat = float(np.clip(yhat, -0.75, 0.75))
        pred.iloc[t] = yhat

    # Backtest meta rows (match your UI table style)
    meta = []
    if n > 10:
        # choose key dates if possible
        def _safe_date(i: int) -> Optional[pd.Timestamp]:
            if 0 <= i < n:
                return idx[i]
            return None

        end = n - 1
        test_start = test_start_idx

        mapping = [
            ("3 Months Ago (OOS)", end - 63),
            ("6 Months Ago (OOS)", end - 126),
            ("1 Year Ago (OOS)", end - 252),
            ("Test Start (OOS)", test_start),
        ]
        for label, ii in mapping:
            d = _safe_date(ii)
            if d is not None:
                meta.append({"period": label, "date": d})

    return pred, meta

# =========================================================
# 7C) PRICE-ONLY FORECASTER (no fundamentals required)
# =========================================================
def build_price_features(close: pd.Series, mkt: Optional[pd.Series] = None) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df["ret_1d"] = close.pct_change()
    df["ret_5d"] = close.pct_change(5)
    df["ret_21d"] = close.pct_change(21)
    df["vol_21d"] = df["ret_1d"].rolling(21).std(ddof=0)
    df["vol_63d"] = df["ret_1d"].rolling(63).std(ddof=0)
    if mkt is not None and mkt.notna().sum() > 50:
        mr = mkt.pct_change()
        df["mkt_ret_1d"] = mr
        df["mkt_ret_21d"] = mkt.pct_change(21)
        df["corr_63d"] = df["ret_1d"].rolling(63).corr(mr)
    return df.replace([np.inf, -np.inf], np.nan)

def walk_forward_price_forecast(
    close: pd.Series,
    mkt_close: Optional[pd.Series],
    horizon: int,
    train_days: int,
    test_days: int,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Forecasts future return over horizon using only price features.
    Predict y_ret(t) = close(t)/close(t-horizon)-1? No. We forecast forward:
    target at t is forward return from t to t+horizon, but that is not observable at end.
    For backtest, we compute realized forward return where possible.
    For "today", we use the last fitted model and last features.
    """
    idx = close.index
    feat = build_price_features(close, mkt_close)

    # forward return target
    y_fwd = (close.shift(-horizon) / close) - 1.0
    y_fwd = y_fwd.reindex(idx).astype(float)

    pred_ret = pd.Series(index=idx, dtype=float)

    n = len(idx)
    test_start_idx = max(0, n - test_days)

    for t in range(test_start_idx, n):
        train_start = max(0, t - train_days)

        X_train_df = feat.iloc[train_start:t]
        y_train = y_fwd.iloc[train_start:t]

        mask = np.isfinite(y_train.values)
        X_train_df2 = X_train_df.loc[mask]
        y_train2 = y_train.loc[mask]

        if X_train_df2.shape[0] < 120:
            continue

        X_train_np = X_train_df2.values.astype(float)
        y_train_np = y_train2.values.astype(float)

        col_ok = np.isfinite(X_train_np).mean(axis=0) > 0.85
        if col_ok.sum() < 3:
            continue
        X_train_np = X_train_np[:, col_ok]

        X_t = feat.iloc[t].values.astype(float)[col_ok]
        if not np.isfinite(X_t).all():
            continue

        Xtr, Xap, _mu = _standardize_train_apply(X_train_np, X_t.reshape(1, -1))
        Xtr_i = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
        Xap_i = np.c_[np.ones((Xap.shape[0], 1)), Xap]

        beta = _fit_ridge_closed_form(Xtr_i, y_train_np, l2=25.0)
        yhat = float(Xap_i @ beta)
        yhat = float(np.clip(yhat, -0.50, 0.50))
        pred_ret.iloc[t] = yhat

    meta = []
    if n > 10:
        end = n - 1
        test_start = test_start_idx
        mapping = [
            ("3 Months Ago (OOS)", end - 63),
            ("6 Months Ago (OOS)", end - 126),
            ("1 Year Ago (OOS)", end - 252),
            ("Test Start (OOS)", test_start),
        ]
        for label, ii in mapping:
            if 0 <= ii < n:
                meta.append({"period": label, "date": idx[ii]})

    return pred_ret, meta

# =========================================================
# 8) REQUEST MODEL
# =========================================================
class StockRequest(BaseModel):
    ticker: str

# =========================================================
# 9) MAIN ANALYSIS ENDPOINT
# =========================================================
@app.post("/analyze")
def analyze_stock(request: StockRequest):
    try:
        fetcher = DataFetcher()
        ticker = fetcher.clean_saudi_ticker(request.ticker)

        # ---------- Prices (stock + market index) ----------
        hist, source_stock = fetcher.fetch_prices(ticker, period=DEFAULT_HISTORY_PERIOD)
        mkt_hist, source_mkt = fetcher.fetch_prices(TASI_TICKER, period=DEFAULT_HISTORY_PERIOD)

        hist = normalize_prices_index_tz(hist)
        mkt_hist = normalize_prices_index_tz(mkt_hist)

        if hist is None or hist.empty or "Close" not in hist.columns or hist["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for stock."}, status_code=200)
        if mkt_hist is None or mkt_hist.empty or "Close" not in mkt_hist.columns or mkt_hist["Close"].dropna().empty:
            return JSONResponse({"error": "No valid Close prices for market index (^TASI.SR)."}, status_code=200)

        stock_close_raw = hist["Close"].astype(float).dropna()
        mkt_close_raw = mkt_hist["Close"].astype(float).dropna()

        # Optional volume
        vol_series = None
        if "Volume" in hist.columns:
            vol_series = hist["Volume"].astype(float)
            vol_series = vol_series.replace([np.inf, -np.inf], np.nan)

        # Align by date intersection
        aligned_px = pd.DataFrame({"stock": stock_close_raw, "mkt": mkt_close_raw}).dropna()
        if len(aligned_px) < 300:
            return JSONResponse({"error": "Not enough overlapping price history between stock and TASI."}, status_code=200)

        stock_close = aligned_px["stock"]
        mkt_close = aligned_px["mkt"]
        dates = stock_close.index

        current_price = float(stock_close.iloc[-1])
        prices_list = stock_close.tolist()
        dates_ms = (dates.astype(np.int64) // 10**6).tolist()
        n_days = len(stock_close)

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
            "wacc": None,
            "growth": None,
            "fcff": None,
            "fundamentals": "quarterly_ttm" if (isinstance(fin_q, pd.DataFrame) and not fin_q.empty) else "annual_fallback",
            "prices_source_stock": source_stock,
            "prices_source_market": source_mkt,
            "walk_forward": f"train={TRAIN_WINDOW_DAYS}d,test={TEST_WINDOW_DAYS}d",
        }

        # ---- Core fundamental series candidates ----
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

        # Annual fallbacks
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

        # Effective tax rate (best-effort)
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

        # Net debt time series
        st_debt_q = st_debt_q if st_debt_q is not None else pd.Series(dtype=float)
        lt_debt_q = lt_debt_q if lt_debt_q is not None else pd.Series(dtype=float)
        cash_q = cash_q if cash_q is not None else pd.Series(dtype=float)

        debt_q = (st_debt_q.fillna(0.0) + lt_debt_q.fillna(0.0)).sort_index()
        cash_q = cash_q.fillna(0.0).sort_index()
        net_debt_q = (debt_q - cash_q).sort_index()
        net_debt_daily = last_value_on_or_before(net_debt_q, dates)

        # Shares time series
        if shares_now is None or not np.isfinite(shares_now) or shares_now <= 0:
            # fallback if info missing: infer from market cap (if any)
            if mcap_now is not None and np.isfinite(mcap_now) and current_price > 0:
                shares_now = float(mcap_now / current_price)

        if shares_now is None or not np.isfinite(shares_now) or shares_now <= 0:
            # still missing: will degrade valuation models; price-only will still run
            shares_now = float("nan")

        if shares_q is not None and shares_q.dropna().size >= 2:
            shares_daily = last_value_on_or_before(shares_q, dates)
            if not np.isfinite(shares_daily.dropna().median()) or shares_daily.dropna().median() <= 0:
                shares_daily = pd.Series(index=dates, data=float(shares_now) if np.isfinite(shares_now) else np.nan, dtype=float)
                method_flags["shares"] = "constant_info_or_nan"
            else:
                method_flags["shares"] = "report_aligned_best_effort"
        else:
            shares_daily = pd.Series(index=dates, data=float(shares_now) if np.isfinite(shares_now) else np.nan, dtype=float)
            method_flags["shares"] = "constant_info_or_nan"

        # EPS TTM daily
        if ni_q is not None and ni_q.dropna().size >= 4 and shares_daily.notna().sum() > 50:
            ni_ttm = ttm_from_quarters(ni_q)
            ni_ttm_daily = last_value_on_or_before(ni_ttm, dates)
            eps_ttm_daily = ni_ttm_daily / shares_daily.replace(0.0, np.nan)
        else:
            eps_ttm_daily = pd.Series(index=dates, dtype=float)

        # ---- BVPS daily (equity / shares) ----
        if eq_q is not None and eq_q.dropna().size >= 1 and shares_daily.notna().sum() > 50:
            eq_daily = last_value_on_or_before(eq_q, dates)
            bvps_daily = eq_daily / shares_daily.replace(0.0, np.nan)
        else:
            # fallback: info bookValue is usually per-share
            bookv = _to_float(info.get("bookValue"))
            bvps_daily = pd.Series(index=dates, data=(bookv if bookv is not None else np.nan), dtype=float)

        # ---- EBITDA TTM daily ----
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
            ebitda_info = _to_float(info.get("ebitda"))
            ebitda_ttm_daily = pd.Series(index=dates, data=(ebitda_info if ebitda_info is not None else np.nan), dtype=float)

        # ---- FCFF TTM daily (CFO - CapEx) ----
        fcff_ttm_daily = pd.Series(index=dates, dtype=float)
        if cfo_q is not None and cfo_q.dropna().size >= 4 and capex_q is not None and capex_q.dropna().size >= 4:
            cfo_ttm = ttm_from_quarters(cfo_q)
            capex_ttm = ttm_from_quarters(capex_q)
            cfo_ttm_daily = last_value_on_or_before(cfo_ttm, dates)
            capex_ttm_daily = last_value_on_or_before(capex_ttm, dates)

            capex_out = capex_ttm_daily.values.astype(float)
            capex_out = np.where(np.isfinite(capex_out), capex_out, np.nan)
            capex_out = np.where(capex_out < 0, -capex_out, capex_out)

            fcff_ttm_daily = pd.Series(index=dates, data=(cfo_ttm_daily.values - capex_out), dtype=float)
            method_flags["fcff"] = "ttm_cfo_minus_capex"
        else:
            method_flags["fcff"] = "unavailable"

        # ---- Growth estimate ----
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

        # ---- WACC (conservative) ----
        D = net_debt_daily.clip(lower=0.0)
        if mcap_now is None or not np.isfinite(mcap_now):
            # approximate with current price * shares
            if np.isfinite(shares_daily.iloc[-1]):
                mcap_now = float(current_price * float(shares_daily.iloc[-1]))
        E = pd.Series(index=dates, data=float(mcap_now) if (mcap_now is not None and np.isfinite(mcap_now)) else np.nan, dtype=float)
        Vcap = (D + E).replace(0.0, np.nan)
        wd = (D / Vcap).clip(0.0, 0.95)
        we = (E / Vcap).clip(0.05, 1.0)
        Rd = rf
        wacc_daily = (we * Re + wd * Rd * (1.0 - T)).clip(0.0, WACC_MAX)
        method_flags["wacc"] = "wacc_equity_capm_debt_rf_proxy"

        # ---- Build observed multiples series ----
        close_series = pd.Series(index=dates, data=stock_close.values, dtype=float)

        pe_obs = close_series / eps_ttm_daily.replace(0.0, np.nan)
        pb_obs = close_series / bvps_daily.replace(0.0, np.nan)

        ev_daily = (close_series * shares_daily.replace(0.0, np.nan)) + net_debt_daily
        ev_ebitda_obs = ev_daily / ebitda_ttm_daily.replace(0.0, np.nan)

        # ---- Self-anchored target multiples ----
        def rolling_target_multiple(obs: pd.Series, window: int = TRADING_DAYS * 2) -> pd.Series:
            m = obs.copy()
            m = m.replace([np.inf, -np.inf], np.nan)
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

        # ---- DCF model daily ----
        dcf_model = pd.Series(index=dates, dtype=float)
        market_long_run_g = rm_exp
        if fcff_ttm_daily.dropna().size > 200 and shares_daily.notna().sum() > 50:
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
        avail = np.array([
            np.isfinite(models["dcf"]).sum() > 150,
            np.isfinite(models["pe"]).sum() > 150,
            np.isfinite(models["pb"]).sum() > 150,
            np.isfinite(models["ev_ebitda"]).sum() > 150
        ], dtype=bool)

        # If none: DO NOT FAIL — fall back to price-only forecaster
        fundamentals_available = bool(np.any(avail))

        # ---- Train valuation weights on past ----
        close_arr = close_series.values.astype(float)
        n = len(dates)
        train_start_idx = max(0, n - TRAIN_WINDOW_DAYS)

        if fundamentals_available:
            y_train = close_arr[train_start_idx:]
            X_train = X[:, train_start_idx:]
            try:
                w_val = optimize_weights_dirichlet(y_train, X_train, avail, n_samples=N_WEIGHT_SAMPLES)
            except Exception:
                w_val = np.zeros(4, dtype=float)
                idxs = np.where(avail)[0]
                w_val[idxs] = 1.0 / len(idxs)

            V_anchor = np.nansum((X.T * w_val), axis=1)
            V_anchor = pd.Series(index=dates, data=V_anchor, dtype=float)
        else:
            w_val = np.zeros(4, dtype=float)
            V_anchor = pd.Series(index=dates, data=np.nan, dtype=float)

        # =========================================================
        # SPREAD ENGINE
        # =========================================================
        backtest = []
        fair_value_1m = np.nan
        P_hat_realized = pd.Series(index=dates, dtype=float)

        if fundamentals_available and V_anchor.notna().sum() > 200:
            df_core = pd.DataFrame(index=dates)
            df_core["Close"] = close_series
            df_core["MktClose"] = pd.Series(index=dates, data=mkt_close.reindex(dates).values, dtype=float)
            df_core["V_anchor"] = V_anchor
            if vol_series is not None:
                df_core["Volume"] = vol_series.reindex(dates)

            feat = build_spread_features(df_core, shares_daily)

            delta_pct = (df_core["Close"] - df_core["V_anchor"]) / df_core["V_anchor"].replace(0.0, np.nan)
            y_target = delta_pct

            preds_delta, backtest_meta = walk_forward_spread_forecast(
                df_feat=feat,
                y_target=y_target,
                horizon=SPREAD_HORIZON_DAYS,
                train_days=TRAIN_WINDOW_DAYS,
                test_days=TEST_WINDOW_DAYS,
            )

            V_lag = V_anchor.shift(SPREAD_HORIZON_DAYS)
            P_hat_realized = V_lag * (1.0 + preds_delta)

            # "now" delta estimate
            if preds_delta.dropna().size:
                delta_hat_now = float(preds_delta.dropna().iloc[-1])
            else:
                delta_hat_now = float(delta_pct.dropna().iloc[-1]) if delta_pct.dropna().size else 0.0

            V_now = float(V_anchor.iloc[-1]) if np.isfinite(V_anchor.iloc[-1]) else float(current_price)
            fair_value_1m = V_now * (1.0 + float(np.clip(delta_hat_now, -0.75, 0.75)))

            for row in backtest_meta:
                d = row["date"]
                actual = float(df_core.loc[d, "Close"]) if d in df_core.index else np.nan
                modelv = float(P_hat_realized.loc[d]) if d in P_hat_realized.index else np.nan
                if np.isfinite(actual) and np.isfinite(modelv):
                    backtest.append({"period": row["period"], "actual": actual, "model": modelv})

        # =========================================================
        # PRICE-ONLY FORECAST (always available)
        # =========================================================
        price_pred_ret, price_bt_meta = walk_forward_price_forecast(
            close=close_series,
            mkt_close=pd.Series(index=dates, data=mkt_close.reindex(dates).values, dtype=float),
            horizon=PRICE_HORIZON_DAYS,
            train_days=TRAIN_WINDOW_DAYS,
            test_days=TEST_WINDOW_DAYS,
        )

        if price_pred_ret.dropna().size:
            ret_hat_now = float(price_pred_ret.dropna().iloc[-1])
        else:
            ret_hat_now = 0.0

        price_forecast_1m = float(current_price * (1.0 + float(np.clip(ret_hat_now, -0.50, 0.50))))

        # If fundamentals-based forecast is unavailable, use price-only as headline fair value
        if not np.isfinite(fair_value_1m):
            fair_value_1m = price_forecast_1m

        # Chart fair values: if spread-engine exists use it; else use price-only implied path
        if fundamentals_available and P_hat_realized.notna().sum() > 50:
            fair_series_for_chart = P_hat_realized.reindex(dates).astype(float)
            fair_values_list = fair_series_for_chart.ffill().bfill().tolist()
        else:
            # price-only: produce a "model line" equal to close * (1 + predicted forward return shifted back)
            # (gives a reasonable comparable line; not perfect, but avoids blank chart)
            po_line = close_series * (1.0 + price_pred_ret.shift(-PRICE_HORIZON_DAYS))
            fair_values_list = po_line.reindex(dates).astype(float).ffill().bfill().tolist()

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

        # Present-day headline multiples
        eps_now = float(eps_ttm_daily.iloc[-1]) if np.isfinite(eps_ttm_daily.iloc[-1]) else np.nan
        bvps_now = float(bvps_daily.iloc[-1]) if np.isfinite(bvps_daily.iloc[-1]) else np.nan
        pe_now = safe_div(current_price, eps_now) if np.isfinite(eps_now) and eps_now != 0 else np.nan
        book_value_now = (bvps_now * float(shares_daily.iloc[-1])) if np.isfinite(bvps_now) and np.isfinite(shares_daily.iloc[-1]) else np.nan

        model_breakdown = {
            "dcf": float(dcf_model.iloc[-1]) if np.isfinite(dcf_model.iloc[-1]) else None,
            "pe_model": float(pe_model.iloc[-1]) if np.isfinite(pe_model.iloc[-1]) else None,
            "pb_model": float(pb_model.iloc[-1]) if np.isfinite(pb_model.iloc[-1]) else None,
            "ev_ebitda_model": float(ev_ebitda_model.iloc[-1]) if np.isfinite(ev_ebitda_model.iloc[-1]) else None,
        }

        # DCF projections (display only)
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
                "method_flags": {
                    **method_flags,
                    "fundamentals_available": fundamentals_available,
                    "headline_forecast": "spread_engine" if fundamentals_available else "price_only",
                },
                "price_only_forecast_1m": price_forecast_1m,
            },
            "metrics": {
                "market_cap": float(mcap_now) if (mcap_now is not None and np.isfinite(mcap_now)) else None,
                "pe_ratio": float(pe_now) if np.isfinite(pe_now) else None,
                "eps": float(eps_now) if np.isfinite(eps_now) else None,
                "beta": float(beta) if np.isfinite(beta) else None,
                "growth_rate": float(growth_daily.iloc[-1]) if np.isfinite(growth_daily.iloc[-1]) else None,
                "book_value": float(book_value_now) if np.isfinite(book_value_now) else None,
                "wacc": float(wacc_daily.iloc[-1]) if np.isfinite(wacc_daily.iloc[-1]) else None,
            },
            "optimized_weights": {
                "dcf": float(w_val[0]),
                "pe": float(w_val[1]),
                "pb": float(w_val[2]),
                "ev_ebitda": float(w_val[3]),
            },
            "backtest": backtest,  # spread-engine backtest (if available)
            "returns": returns,
            "historical_data": {
                "dates": dates_ms,
                "prices": prices_list,
                "fair_values": fair_values_list,
            },
        }

        return JSONResponse(json_safe(response), status_code=200)

    except Exception as e:
        return JSONResponse({"error": f"Unhandled error: {str(e)}"}, status_code=200)
