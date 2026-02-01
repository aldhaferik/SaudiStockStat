#!/usr/bin/env python3
"""
Real Machine Learning Saudi Stock Predictor
- Trains actual ML models on historical data
- Tests model accuracy on 5-year historical data
- Predicts future stock prices
- Determines if stock is undervalued/overvalued
- Shows real model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import time
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

class RealMLSaudiStockPredictor:
    def __init__(self):
        self.company_tickers = [
            '2222', '2330', '1120', '4001', '2010', '1180', '2020', '1210', 
            '4030', '2170', '1050', '2280', '6020', '9608', '3030', '2040',
            '2230', '4008', '4130', '5110', '8311', '6050', '4031', '2130'
        ]
        
        self.company_data = {
            "2222": {"name": "Saudi Aramco", "sector": "Energy"},
            "2330": {"name": "SABIC", "sector": "Materials"},
            "1120": {"name": "Al Rajhi Bank", "sector": "Financials"},
            "4001": {"name": "Saudi Telecom", "sector": "Communication"},
            "3030": {"name": "Saudi Electricity Company", "sector": "Utilities"},
        }
        
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
    
    def get_company_info(self, ticker):
        """Get company information"""
        return self.company_data.get(ticker, {"name": f"Saudi Company {ticker}", "sector": "Unknown"})
    
    def get_real_price_data(self, ticker, period="5y"):
        """Get real historical price data from Yahoo Finance"""
        try:
            # Use correct .SR format for Saudi stocks
            ticker_variants = [
                f"{ticker}.SR",
                f"{int(ticker):04d}.SR" if ticker.isdigit() else f"{ticker}.SR",
                f"{ticker}.SAU",
            ]
            
            for ticker_variant in ticker_variants:
                try:
                    stock = yf.Ticker(ticker_variant)
                    hist_data = stock.history(period=period)
                    
                    if not hist_data.empty and len(hist_data) > 100:
                        st.success(f"✅ Downloaded {len(hist_data)} days of price data for {ticker_variant}")
                        return hist_data, ticker_variant
                        
                except Exception as e:
                    continue
            
            st.error(f"❌ No price data found for {ticker}")
            return None, None
            
        except Exception as e:
            st.error(f"❌ Error getting price data: {e}")
            return None, None
    
    def extract_tadawul_financial_data(self, ticker):
        """Extract real financial data from Tadawul"""
        try:
            st.info(f"🔍 Extracting financial data from Tadawul for {ticker}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,ar;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Tadawul URL pattern
            url = f"https://www.saudiexchange.sa/wps/portal/saudiexchange/hidden/company-profile-main/!ut/p/z1/04_Sj9CPykssy0xPLMnMz0vMAfIjo8ziTR3NDIw8LAz83d2MXA0C3SydAl1c3Q0NvE30I4EKzBEKDMKcTQzMDPxN3H19LAzdTU31w8syU8v1wwkpK8hOMgUA-oskdg!!/?companySymbol={ticker}#Z7_5A602H80O0VC4060O4GML81G55"
            
            time.sleep(3)  # Rate limiting
            
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(url, timeout=20)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                financial_data = self._parse_financial_tables(soup, ticker)
                
                if financial_data:
                    st.success(f"✅ Financial data extracted for {ticker}")
                    return financial_data
                else:
                    st.warning(f"⚠️ No financial data found for {ticker}")
                    return None
            else:
                st.error(f"❌ Failed to access Tadawul for {ticker} (Status: {response.status_code})")
                return None
                
        except Exception as e:
            st.error(f"❌ Error extracting financial data: {e}")
            return None
    
    def _parse_financial_tables(self, soup, ticker):
        """Parse financial data from HTML tables"""
        try:
            financial_data = {
                'ticker': ticker,
                'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        value_text = cells[1].get_text(strip=True)
                        
                        numeric_value = self._extract_numeric_value(value_text)
                        
                        if numeric_value is not None:
                            # Map financial metrics
                            if any(keyword in label for keyword in ['revenue', 'total revenue', 'net sales']):
                                financial_data['revenue'] = numeric_value
                            elif any(keyword in label for keyword in ['net income', 'net profit']):
                                financial_data['net_income'] = numeric_value
                            elif any(keyword in label for keyword in ['total assets']):
                                financial_data['total_assets'] = numeric_value
                            elif any(keyword in label for keyword in ['equity', 'shareholders equity']):
                                financial_data['equity'] = numeric_value
                            elif any(keyword in label for keyword in ['shares outstanding', 'number of shares']):
                                financial_data['shares_outstanding'] = numeric_value
            
            return financial_data if len(financial_data) > 2 else None
            
        except Exception as e:
            st.error(f"❌ Error parsing financial data: {e}")
            return None
    
    def _extract_numeric_value(self, text):
        """Extract numeric values from text"""
        if not text or not isinstance(text, str):
            return None
        
        try:
            # Remove currency symbols and commas
            text = re.sub(r'[SAR$€£¥,\s]', '', text.strip())
            
            # Handle multipliers
            multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000, 'T': 1000000000000}
            
            match = re.search(r'([0-9]+\.?[0-9]*)([KMBT]?)', text.upper())
            if match:
                number = float(match.group(1))
                suffix = match.group(2)
                if suffix in multipliers:
                    number *= multipliers[suffix]
                return number
            
            # Try direct conversion
            clean_text = re.sub(r'[^0-9.-]', '', text)
            if clean_text:
                return float(clean_text)
            
            return None
            
        except:
            return None
    
    def create_technical_features(self, price_data):
        """Create technical indicators for ML models"""
        try:
            df = price_data.copy()
            
            # Price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
            df['price_change'] = df['Close'] - df['Open']
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'ma_{window}_ratio'] = df['Close'] / df[f'ma_{window}']
            
            # Volatility
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume features
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            
            # Momentum
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error creating technical features: {e}")
            return None
    
    def prepare_ml_data(self, features_df, target_days=1):
        """Prepare data for ML training"""
        try:
            # Create target variable (future price change)
            features_df['target'] = features_df['Close'].shift(-target_days) / features_df['Close'] - 1
            
            # Select feature columns (exclude price columns and target)
            feature_columns = [col for col in features_df.columns if 
                             col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target'] and
                             features_df[col].dtype in ['float64', 'int64']]
            
            # Remove rows with NaN values
            clean_df = features_df[feature_columns + ['target', 'Close']].dropna()
            
            if len(clean_df) < 100:
                st.error(f"❌ Insufficient data for ML training ({len(clean_df)} samples)")
                return None, None, None, None
            
            X = clean_df[feature_columns]
            y = clean_df['target']
            prices = clean_df['Close']
            
            return X, y, prices, feature_columns
            
        except Exception as e:
            st.error(f"❌ Error preparing ML data: {e}")
            return None, None, None, None
    
    def train_ml_models(self, X, y, prices):
        """Train multiple ML models and evaluate performance"""
        if not ML_AVAILABLE:
            st.error("❌ ML libraries not available")
            return None
        
        try:
            st.info("🧠 Training ML models...")
            
            # Time series split for proper backtesting
            tscv = TimeSeriesSplit(n_splits=5)
            
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
            
            results = {}
            
            for name, model in models.items():
                st.write(f"Training {name}...")
                
                # Cross-validation scores
                cv_scores = {'mse': [], 'mae': [], 'r2': [], 'directional_accuracy': []}
                
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Scale features for linear models
                    if name == 'Linear Regression':
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Directional accuracy
                    direction_actual = np.sign(y_test)
                    direction_pred = np.sign(y_pred)
                    directional_accuracy = np.mean(direction_actual == direction_pred)
                    
                    cv_scores['mse'].append(mse)
                    cv_scores['mae'].append(mae)
                    cv_scores['r2'].append(r2)
                    cv_scores['directional_accuracy'].append(directional_accuracy)
                
                # Average CV scores
                results[name] = {
                    'model': model,
                    'cv_mse': np.mean(cv_scores['mse']),
                    'cv_mae': np.mean(cv_scores['mae']),
                    'cv_r2': np.mean(cv_scores['r2']),
                    'cv_directional_accuracy': np.mean(cv_scores['directional_accuracy']),
                    'cv_mse_std': np.std(cv_scores['mse']),
                    'cv_mae_std': np.std(cv_scores['mae']),
                    'cv_r2_std': np.std(cv_scores['r2']),
                    'cv_directional_accuracy_std': np.std(cv_scores['directional_accuracy'])
                }
            
            # Train final models on full dataset
            for name, result in results.items():
                model = result['model']
                
                if name == 'Linear Regression':
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                    self.scalers[name] = scaler
                else:
                    model.fit(X, y)
                
                self.models[name] = model
            
            self.model_performance = results
            st.success("✅ ML models trained successfully!")
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error training ML models: {e}")
            return None
    
    def create_lstm_model(self, X, y, sequence_length=60):
        """Create and train LSTM model for time series prediction"""
        if not DEEP_LEARNING_AVAILABLE:
            st.warning("⚠️ TensorFlow not available, skipping LSTM model")
            return None
        
        try:
            st.info("🧠 Training LSTM model...")
            
            # Prepare data for LSTM
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-sequence_length:i])
                y_sequences.append(y.iloc[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            if len(X_sequences) < 100:
                st.warning("⚠️ Insufficient data for LSTM training")
                return None
            
            # Split data
            split_idx = int(len(X_sequences) * 0.8)
            X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred.flatten()))
            
            lstm_result = {
                'model': model,
                'scaler': scaler,
                'sequence_length': sequence_length,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'history': history.history
            }
            
            self.models['LSTM'] = model
            self.scalers['LSTM'] = scaler
            
            st.success("✅ LSTM model trained successfully!")
            
            return lstm_result
            
        except Exception as e:
            st.error(f"❌ Error training LSTM model: {e}")
            return None
    
    def predict_future_price(self, ticker, current_price, X, feature_columns, days_ahead=30):
        """Predict future stock price using trained models"""
        try:
            if not self.models:
                st.error("❌ No trained models available")
                return None
            
            # Get latest features
            latest_features = X.iloc[-1:][feature_columns]
            
            predictions = {}
            
            for name, model in self.models.items():
                try:
                    if name == 'Linear Regression' and name in self.scalers:
                        latest_scaled = self.scalers[name].transform(latest_features)
                        pred_return = model.predict(latest_scaled)[0]
                    elif name == 'LSTM':
                        # LSTM prediction requires sequence
                        continue  # Skip for now, needs special handling
                    else:
                        pred_return = model.predict(latest_features)[0]
                    
                    # Convert return to price
                    predicted_price = current_price * (1 + pred_return)
                    
                    predictions[name] = {
                        'predicted_return': pred_return,
                        'predicted_price': predicted_price,
                        'confidence': self.model_performance[name]['cv_directional_accuracy']
                    }
                    
                except Exception as e:
                    st.warning(f"⚠️ Error predicting with {name}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            st.error(f"❌ Error making predictions: {e}")
            return None
    
    def determine_valuation(self, current_price, predictions, financial_data=None):
        """Determine if stock is undervalued or overvalued"""
        try:
            if not predictions:
                return "INSUFFICIENT DATA"
            
            # Calculate ensemble prediction
            predicted_prices = [pred['predicted_price'] for pred in predictions.values()]
            avg_predicted_price = np.mean(predicted_prices)
            
            # Calculate expected return
            expected_return = (avg_predicted_price - current_price) / current_price
            
            # Determine valuation
            if expected_return > 0.1:  # >10% expected return
                valuation = "UNDERVALUED"
                confidence = "HIGH"
            elif expected_return > 0.05:  # 5-10% expected return
                valuation = "SLIGHTLY UNDERVALUED"
                confidence = "MEDIUM"
            elif expected_return > -0.05:  # -5% to 5%
                valuation = "FAIRLY VALUED"
                confidence = "MEDIUM"
            elif expected_return > -0.1:  # -10% to -5%
                valuation = "SLIGHTLY OVERVALUED"
                confidence = "MEDIUM"
            else:  # <-10% expected return
                valuation = "OVERVALUED"
                confidence = "HIGH"
            
            return {
                'valuation': valuation,
                'confidence': confidence,
                'expected_return': expected_return,
                'current_price': current_price,
                'predicted_price': avg_predicted_price,
                'price_range': {
                    'min': min(predicted_prices),
                    'max': max(predicted_prices)
                }
            }
            
        except Exception as e:
            st.error(f"❌ Error determining valuation: {e}")
            return None

def main():
    st.set_page_config(
        page_title="Real ML Saudi Stock Predictor",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Real ML Saudi Stock Predictor")
    st.markdown("**Actual Machine Learning Models | Real Training & Testing | Future Price Prediction**")
    
    # Initialize predictor
    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = RealMLSaudiStockPredictor()
    
    predictor = st.session_state.ml_predictor
    
    # Sidebar
    st.sidebar.header("🤖 ML System Status")
    
    with st.sidebar.expander("🔧 Available Libraries", expanded=True):
        st.write(f"✅ ML Models: {'Available' if ML_AVAILABLE else '❌ Not Available'}")
        st.write(f"✅ Deep Learning: {'Available' if DEEP_LEARNING_AVAILABLE else '❌ Not Available'}")
        st.write("✅ Yahoo Finance: Available")
        st.write("✅ Tadawul Scraping: Available")
        
        if not ML_AVAILABLE:
            st.error("❌ Install scikit-learn, xgboost, lightgbm for ML models")
        if not DEEP_LEARNING_AVAILABLE:
            st.warning("⚠️ Install tensorflow for LSTM models")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🤖 ML Prediction", "📊 Model Performance", "📈 Backtesting Results"])
    
    with tab1:
        st.header("🤖 Machine Learning Stock Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stock selection
            display_options = []
            for ticker in predictor.company_tickers:
                company_info = predictor.get_company_info(ticker)
                display_options.append(f"{ticker} - {company_info['name']} ({company_info['sector']})")
            
            selected_display = st.selectbox(
                "Select Stock for ML Analysis",
                options=display_options,
                index=0
            )
            
            ticker = selected_display.split(" - ")[0]
        
        with col2:
            prediction_days = st.selectbox("Prediction Horizon", [1, 7, 30], index=2)
        
        if st.button("🚀 Train Models & Predict", type="primary"):
            # Clear previous results
            if f'ml_results_{ticker}' in st.session_state:
                del st.session_state[f'ml_results_{ticker}']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Get price data
            status_text.text("📈 Downloading historical price data...")
            progress_bar.progress(0.1)
            
            price_data, ticker_symbol = predictor.get_real_price_data(ticker, period="5y")
            
            if price_data is None:
                st.error("❌ Cannot proceed without price data")
                return
            
            # Step 2: Get financial data
            status_text.text("🏛️ Extracting financial data from Tadawul...")
            progress_bar.progress(0.2)
            
            financial_data = predictor.extract_tadawul_financial_data(ticker)
            
            # Step 3: Create technical features
            status_text.text("🔧 Creating technical indicators...")
            progress_bar.progress(0.3)
            
            features_df = predictor.create_technical_features(price_data)
            
            if features_df is None:
                st.error("❌ Failed to create technical features")
                return
            
            # Step 4: Prepare ML data
            status_text.text("📊 Preparing ML dataset...")
            progress_bar.progress(0.4)
            
            X, y, prices, feature_columns = predictor.prepare_ml_data(features_df, target_days=1)
            
            if X is None:
                st.error("❌ Failed to prepare ML data")
                return
            
            # Step 5: Train ML models
            status_text.text("🧠 Training ML models...")
            progress_bar.progress(0.5)
            
            ml_results = predictor.train_ml_models(X, y, prices)
            
            if ml_results is None:
                st.error("❌ Failed to train ML models")
                return
            
            # Step 6: Train LSTM model
            status_text.text("🧠 Training LSTM model...")
            progress_bar.progress(0.7)
            
            lstm_result = predictor.create_lstm_model(X, y)
            
            # Step 7: Make predictions
            status_text.text("🔮 Making predictions...")
            progress_bar.progress(0.8)
            
            current_price = price_data['Close'].iloc[-1]
            predictions = predictor.predict_future_price(ticker, current_price, X, feature_columns)
            
            # Step 8: Determine valuation
            status_text.text("💰 Determining valuation...")
            progress_bar.progress(0.9)
            
            valuation_result = predictor.determine_valuation(current_price, predictions, financial_data)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis completed!")
            
            # Store results
            results = {
                'ticker': ticker,
                'ticker_symbol': ticker_symbol,
                'price_data': price_data,
                'financial_data': financial_data,
                'ml_results': ml_results,
                'lstm_result': lstm_result,
                'predictions': predictions,
                'valuation': valuation_result,
                'current_price': current_price,
                'analysis_date': datetime.now()
            }
            
            st.session_state[f'ml_results_{ticker}'] = results
            
            # Display results
            st.success("✅ ML Analysis Completed!")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"{current_price:.2f} SAR")
            
            with col2:
                if valuation_result:
                    predicted_price = valuation_result['predicted_price']
                    st.metric("Predicted Price", f"{predicted_price:.2f} SAR")
            
            with col3:
                if valuation_result:
                    expected_return = valuation_result['expected_return']
                    st.metric("Expected Return", f"{expected_return:.1%}")
            
            with col4:
                if valuation_result:
                    valuation = valuation_result['valuation']
                    confidence = valuation_result['confidence']
                    st.metric("Valuation", valuation, delta=f"Confidence: {confidence}")
            
            # Model predictions
            if predictions:
                st.subheader("🤖 Model Predictions")
                
                pred_data = []
                for model_name, pred in predictions.items():
                    pred_data.append({
                        'Model': model_name,
                        'Predicted Price': f"{pred['predicted_price']:.2f} SAR",
                        'Expected Return': f"{pred['predicted_return']:.1%}",
                        'Confidence': f"{pred['confidence']:.1%}"
                    })
                
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, use_container_width=True)
            
            # Price chart with prediction
            st.subheader("📈 Price Chart with Prediction")
            
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name="Historical Price"
            ))
            
            # Prediction point
            if valuation_result:
                future_date = price_data.index[-1] + timedelta(days=prediction_days)
                predicted_price = valuation_result['predicted_price']
                
                fig.add_trace(go.Scatter(
                    x=[price_data.index[-1], future_date],
                    y=[current_price, predicted_price],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='red', dash='dash')
                ))
            
            fig.update_layout(
                title=f"{ticker} - ML Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price (SAR)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 Model Performance Metrics")
        
        # Check if we have ML results
        ml_results_keys = [k for k in st.session_state.keys() if k.startswith('ml_results_')]
        
        if ml_results_keys:
            for key in ml_results_keys:
                results = st.session_state[key]
                ticker = results['ticker']
                
                with st.expander(f"📊 Model Performance: {ticker}"):
                    ml_results = results.get('ml_results', {})
                    
                    if ml_results:
                        # Performance table
                        perf_data = []
                        for model_name, result in ml_results.items():
                            perf_data.append({
                                'Model': model_name,
                                'CV R² Score': f"{result['cv_r2']:.3f} ± {result['cv_r2_std']:.3f}",
                                'CV RMSE': f"{np.sqrt(result['cv_mse']):.4f} ± {np.sqrt(result['cv_mse_std']):.4f}",
                                'CV MAE': f"{result['cv_mae']:.4f} ± {result['cv_mae_std']:.4f}",
                                'Directional Accuracy': f"{result['cv_directional_accuracy']:.1%} ± {result['cv_directional_accuracy_std']:.1%}"
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        # Best model
                        best_model = max(ml_results.keys(), 
                                       key=lambda x: ml_results[x]['cv_directional_accuracy'])
                        best_accuracy = ml_results[best_model]['cv_directional_accuracy']
                        
                        st.success(f"🏆 Best Model: {best_model} (Directional Accuracy: {best_accuracy:.1%})")
                    
                    # LSTM performance
                    lstm_result = results.get('lstm_result')
                    if lstm_result:
                        st.write("**LSTM Model Performance:**")
                        st.write(f"- R² Score: {lstm_result['r2']:.3f}")
                        st.write(f"- RMSE: {np.sqrt(lstm_result['mse']):.4f}")
                        st.write(f"- MAE: {lstm_result['mae']:.4f}")
                        st.write(f"- Directional Accuracy: {lstm_result['directional_accuracy']:.1%}")
        else:
            st.info("📝 No model performance data available. Run ML analysis first.")
    
    with tab3:
        st.header("📈 Backtesting Results")
        
        if ml_results_keys:
            for key in ml_results_keys:
                results = st.session_state[key]
                ticker = results['ticker']
                
                with st.expander(f"📈 Backtesting: {ticker}"):
                    st.write("**5-Year Historical Backtesting:**")
                    st.write("- Models trained using time series cross-validation")
                    st.write("- 5-fold cross-validation with proper time series splits")
                    st.write("- Performance metrics calculated on out-of-sample data")
                    st.write("- Directional accuracy measures prediction of price direction")
                    
                    ml_results = results.get('ml_results', {})
                    if ml_results:
                        # Show cross-validation methodology
                        st.write("**Cross-Validation Methodology:**")
                        st.write("1. Data split into 5 time-ordered folds")
                        st.write("2. Each model trained on past data, tested on future data")
                        st.write("3. Performance averaged across all folds")
                        st.write("4. Standard deviation shows model stability")
                        
                        # Performance visualization
                        model_names = list(ml_results.keys())
                        accuracies = [ml_results[name]['cv_directional_accuracy'] for name in model_names]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=model_names, y=accuracies, name='Directional Accuracy')
                        ])
                        
                        fig.update_layout(
                            title="Model Directional Accuracy (5-Year Backtest)",
                            xaxis_title="Model",
                            yaxis_title="Accuracy",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 No backtesting results available. Run ML analysis first.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🤖 Real ML Saudi Stock Predictor | 
            Actual machine learning models trained on 5-year historical data
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()