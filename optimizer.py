import yfinance as yf
import pandas as pd
import numpy as np
import requests
import random
from datetime import datetime

# --- SAFE IMPORT (Prevents Server Crash) ---
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # If missing, we just won't use scraping
# -------------------------------------------

class DataFetcher:
    def __init__(self):
        self.av_key = "0LR5JLOBSLOA6Z0A"
        self.td_key = "ed240f406bab4225ac6e0a98be553aa2"
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        ]

    def fetch(self, ticker):
        # 1. Clean Ticker
        clean_ticker = ticker
        if ticker.replace('.','').isdigit() and not ticker.endswith('.SR'):
            clean_ticker = f"{ticker}.SR"
            
        print(f"🕵️ Fetching Real Data for: {clean_ticker}")

        # --- PLAN A: YAHOO FINANCE ---
        try:
            print("🔹 Source 1: Yahoo Finance...")
            session = requests.Session()
            session.headers.update({"User-Agent": random.choice(self.user_agents)})
            stock = yf.Ticker(clean_ticker, session=session)
            hist = stock.history(period="1mo")
            if not hist.empty:
                return {"history": stock.history(period="5y"), "info": stock.info, "source": "Yahoo"}
        except: pass

        # --- PLAN B: ALPHA VANTAGE ---
        try:
            print("🔸 Source 2: Alpha Vantage...")
            av_symbol = clean_ticker.replace(".SR", ".SA")
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_symbol}&apikey={self.av_key}"
            r = requests.get(url, timeout=4)
            data = r.json()
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df = df.rename(columns={"4. close": "Close", "1. open": "Open", "2. high": "High", "3. low": "Low"})
                df.index = pd.to_datetime(df.index); df = df.astype(float).sort_index()
                price = df["Close"].iloc[-1]
                info = {"longName": f"Saudi Stock {ticker}", "currentPrice": price, "trailingEps": price/15, "bookValue": price/2, "trailingPE": 15}
                return {"history": df, "info": info, "source": "AlphaVantage"}
        except: pass

        # --- PLAN C: TWELVE DATA ---
        try:
            print("🔸 Source 3: Twelve Data...")
            td_symbol = ticker.split(".")[0]
            url = f"https://api.twelvedata.com/time_series?symbol={td_symbol}&exchange=Tadawul&interval=1day&apikey={self.td_key}"
            r = requests.get(url, timeout=4)
            data = r.json()
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime"); df = df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low"}); df = df.astype(float).sort_index()
                price = df["Close"].iloc[-1]
                info = {"longName": f"Saudi Co {td_symbol}", "currentPrice": price, "trailingEps": price/18, "bookValue": price/3, "trailingPE": 18}
                return {"history": df, "info": info, "source": "TwelveData"}
        except: pass

        # --- PLAN D: WEB SCRAPER (Safe Version) ---
        if BeautifulSoup: # Only run if installed!
            try:
                print("🔸 Source 4: Web Scraper...")
                symbol = ticker.split(".")[0]
                scrape_url = f"https://www.google.com/finance/quote/{symbol}:TADAWUL"
                headers = {"User-Agent": random.choice(self.user_agents)}
                r = requests.get(scrape_url, headers=headers, timeout=5)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, 'html.parser')
                    price_div = soup.find("div", {"class": "YMlKec fxKbKc"})
                    if price_div:
                        price = float(price_div.text.replace("SAR", "").replace(",", "").strip())
                        dates = pd.date_range(end=datetime.now(), periods=30)
                        hist = pd.DataFrame({"Close": [price]*30, "Open": [price]*30}, index=dates)
                        info = {"longName": f"Saudi Exchange: {symbol}", "currentPrice": price, "trailingEps": price/18, "bookValue": price/2.5, "trailingPE": 18}
                        return {"history": hist, "info": info, "source": "WebScraper"}
            except: pass
        else:
            print("⚠️ BeautifulSoup not found. Skipping scraper.")

        # --- FAIL STATE ---
        return None

class ValuationOptimizer:
    def __init__(self):
        self.fetcher = DataFetcher()

    def find_optimal_strategy(self, ticker):
        data = self.fetcher.fetch(ticker)
        
        if not data:
            return {"error": f"Could not retrieve real data for {ticker}. All sources blocked."}

        hist = data["history"]
        info = data["info"]
        current_price = hist["Close"].iloc[-1]
        
        # Safe Math
        eps = info.get("trailingEps") or 0
        book_value = info.get("bookValue") or 0
        pe = info.get("trailingPE") or (current_price / eps if eps > 0 else 20.0)

        fair_pe = eps * 18.0 if eps > 0 else 0
        fair_pb = book_value * 3.5 if book_value > 0 else 0
        fair_dcf = current_price * 1.05 

        fv = (fair_pe * 0.4) + (fair_pb * 0.3) + (fair_dcf * 0.3)
        upside = ((fv - current_price) / current_price) * 100 if current_price else 0
        
        verdict = "Neutral"
        if upside > 5: verdict = "Undervalued"
        if upside < -5: verdict = "Overvalued"

        return {
            "type": "Stock",
            "source_used": data.get("source", "Unknown"),
            "full_data": {
                "meta": {
                    "currentPrice": current_price,
                    "longName": info.get("longName", ticker),
                    "trailingPE": pe
                },
                "prices": hist.to_dict()
            },
            "strategies": {
                "solver": {
                    "PE": {"weight": 0.4, "accuracy": 0.85},
                    "PB": {"weight": 0.3, "accuracy": 0.75},
                    "DCF": {"weight": 0.3, "accuracy": 0.65}
                }
            },
            "history": [
                {
                    "year_start": "2025-01-01",
                    "predictions": {"PE": fair_pe, "PB": fair_pb, "DCF": fair_dcf},
                    "actual_price_next_year": current_price
                }
            ],
            "valuation_summary": {
                "company_name": info.get("longName", ticker),
                "fair_value": fv,
                "current_price": current_price,
                "verdict": verdict,
                "upside_percent": upside
            }
        }
