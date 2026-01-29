import yfinance as yf
import pandas as pd
import random
from functools import lru_cache

# --- 1. PYTHON-NATIVE CACHING (No Streamlit) ---
# maxsize=128 means it remembers the last 128 stocks requested.
@lru_cache(maxsize=128)
def fetch_yahoo_data_cached(symbol):
    try:
        # Create a Ticker object
        # We rely on yfinance's internal robust logic (no custom session)
        ticker = yf.Ticker(symbol)
        
        # Fetch History (10 years)
        # auto_adjust=True fixes weird split/dividend issues
        prices = ticker.history(period="10y", auto_adjust=True)
        
        if prices.empty:
            # Fallback: Try downloading via the download method
            prices = yf.download(symbol, period="10y", progress=False, auto_adjust=True)
            
        if prices.empty:
            print(f"❌ No price data found for {symbol}")
            return None
            
        # Try fetching metadata
        try:
            info = ticker.info
        except:
            info = {}
            
        return {
            "prices": prices,
            "info": info,
            "bs": ticker.balance_sheet,
            "is_": ticker.income_stmt,
            "cf": ticker.cashflow
        }
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

class SaudiStockLoader:
    def __init__(self):
        self.suffix = ".SR"

    def fetch_full_data(self, stock_code):
        # Clean Input
        base_code = str(stock_code).replace(".SR", "").replace(".SA", "").strip()
        
        # Try Primary Suffix (.SR)
        print(f"🔍 Fetching {base_code}.SR...")
        data = fetch_yahoo_data_cached(f"{base_code}.SR")
        
        # Try Secondary Suffix (.SA) if primary failed
        if not data:
            print(f"⚠️ Failed. Retrying with {base_code}.SA...")
            data = fetch_yahoo_data_cached(f"{base_code}.SA")
            
        if not data: 
            return None

        return self._package_data(data["info"], data["prices"], data["bs"], data["is_"], data["cf"])

    def _package_data(self, meta, prices, bs, is_, cf):
        # 1. NUCLEAR TIMEZONE FIX
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        # 2. ETF Detection
        q_type = meta.get('quoteType', '').upper()
        no_financials = True
        if bs is not None and not bs.empty:
            no_financials = False
        
        is_etf = (q_type == 'ETF') or (q_type == 'MUTUALFUND') or (no_financials)

        # 3. Financials Sanitization
        def sanitize(df):
            if df is None or df.empty: return df
            try:
                # Convert columns to datetime
                df.columns = pd.to_datetime(df.columns)
                # Strip timezone from columns if present
                if df.columns.tz is not None:
                    df.columns = df.columns.tz_localize(None)
            except:
                pass
            return df
        
        return {
            "meta": meta,
            "prices": prices,
            "is_etf": is_etf,
            "financials": {
                "balance_sheet": sanitize(bs), 
                "income_statement": sanitize(is_), 
                "cash_flow": sanitize(cf)
            }
        }

    def get_data_as_of_date(self, stock_data, valuation_date_str):
        if stock_data.get("is_etf", False): return None 

        cutoff_date = pd.to_datetime(valuation_date_str)
        
        prices = stock_data["prices"].copy()
        if prices.index.tz is not None:
             prices.index = prices.index.tz_localize(None)
             
        past_prices = prices[prices.index < cutoff_date]
        if past_prices.empty: return None
        simulated_current_price = past_prices['Close'].iloc[-1]

        def filter_financials(df):
            if df is None or df.empty: return df
            valid_cols = []
            for c in df.columns:
                if isinstance(c, pd.Timestamp):
                    if c.tz is not None: c = c.tz_localize(None)
                    if c < cutoff_date: valid_cols.append(c)
            return df[valid_cols]

        past_financials = {
            "balance_sheet": filter_financials(stock_data["financials"]["balance_sheet"]),
            "income_statement": filter_financials(stock_data["financials"]["income_statement"]),
            "cash_flow": filter_financials(stock_data["financials"]["cash_flow"])
        }

        return {
            "simulation_date": valuation_date_str,
            "price_at_simulation": simulated_current_price,
            "financials": past_financials
        }
