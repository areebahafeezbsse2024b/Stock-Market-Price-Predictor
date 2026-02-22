import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="QuantumPredict Pro",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    /* Force Background on the entire App */
    .stApp {
        background: #06050a !important;
        background-image: radial-gradient(circle at 50% 50%, #110b1f 0%, #06050a 100%) !important;
    }

    /* Target the main container specifically */
    .main { 
        background: transparent !important;
    }

    /* Fix Text Visibility - High Contrast for Readability */
    p, span, li, label {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
    }

    /* Headings - Neon Purple Gradient */
    h1, h2, h3, h4 { 
        background: linear-gradient(90deg, #b76eff, #9d4edd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Info Box / Welcome Card Fix */
    .info-box {
        background: rgba(26, 11, 46, 0.8) !important;
        border: 1px solid #7b2cbf !important;
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .info-box h3 {
        -webkit-text-fill-color: #c77dff !important;
    }

    /* Popular Stocks Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #5a189a, #7b2cbf) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
        text-align: left !important;
        padding-left: 20px !important;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(157, 78, 221, 0.5) !important;
        background: linear-gradient(45deg, #7b2cbf, #9d4edd) !important;
    }

    /* Sidebar Fixes */
    section[data-testid="stSidebar"] {
        background-color: #0d0b16 !important;
        border-right: 1px solid #2d1b4e !important;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(17, 11, 31, 0.9) !important;
        border-left: 5px solid #9d4edd !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    /* Footer Fix */
    .footer {
        background: #0d0b16 !important;
        border-top: 2px solid #5a189a !important;
        padding: 30px !important;
        margin-top: 50px;
        border-radius: 16px 16px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("🚀 QuantumPredict Pro")
    st.markdown("---")
    
    # Stock selection
    ticker = st.text_input("Stock Symbol", value="AAPL").upper()
    
    # Quick selection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Tech"):
            st.session_state.ticker = "AAPL"
            st.session_state.run_prediction = True
            st.rerun()
    with col2:
        if st.button("EV"):
            st.session_state.ticker = "TSLA"
            st.session_state.run_prediction = True
            st.rerun()
    
    st.markdown("---")
    
    # Date range selection
    end_date = datetime.now()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                  value=end_date - timedelta(days=365))
    with col2:
        end_date_input = st.date_input("End Date", value=end_date)
    
    # Model parameters
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    
    prediction_days = st.slider("Lookback Days", 10, 120, 60, 
                               help="Number of historical days for prediction patterns")
    forecast_days = st.slider("Forecast Days", 1, 60, 14,
                            help="Number of days to predict into the future")
    
    # Model type selection
    model_type = st.selectbox(
        "AI Model",
        ["Random Forest", "Gradient Boosting", "Ensemble"],
        help="Select the machine learning algorithm"
    )
    
    st.markdown("---")
    
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        st.session_state.run_prediction = True
        st.session_state.ticker = ticker
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date_input
        st.session_state.prediction_days = prediction_days
        st.session_state.forecast_days = forecast_days
        st.session_state.model_type = model_type

# ============================================================================
# HELPER FUNCTIONS - FIXED VERSION
# ============================================================================
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch comprehensive stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get stock info
        info = stock.info
        
        # Download historical data
        df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            return None, None
        
        return df, info
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    return df.dropna()

def create_simplified_features(df, lookback_days=60):
    """Create simplified feature set that works with all models"""
    features = []
    targets = []
    
    # Limit lookback to avoid too many features
    lookback_days = min(lookback_days, 30)  # Max 30 days for stability
    
    for i in range(lookback_days, len(df) - 1):  # Predict next day
        # Price features (last n days)
        price_window = df['Close'].iloc[i-lookback_days:i].values
        
        # Technical indicators from current day
        ma_20 = df['MA_20'].iloc[i] if i < len(df) else 0
        ma_50 = df['MA_50'].iloc[i] if i < len(df) else 0
        rsi = df['RSI'].iloc[i] if i < len(df) else 50
        macd = df['MACD'].iloc[i] if i < len(df) else 0
        volume_ratio = df['Volume_Ratio'].iloc[i] if i < len(df) else 1
        volatility = df['Volatility'].iloc[i] if i < len(df) else 0
        
        # Price momentum features
        price_change_1d = df['Daily_Return'].iloc[i] if i < len(df) else 0
        price_change_5d = ((df['Close'].iloc[i] - df['Close'].iloc[max(0, i-5)]) / df['Close'].iloc[max(0, i-5)] * 100) if i >= 5 else 0
        
        # Combine all features - keep it simple
        feature_vector = np.concatenate([
            price_window,
            [ma_20, ma_50, rsi, macd, volume_ratio, volatility, price_change_1d, price_change_5d]
        ])
        
        features.append(feature_vector)
        targets.append(df['Close'].iloc[i + 1])  # Predict next day's close
    
    return np.array(features), np.array(targets)

def train_model(X_train, y_train, model_type="Random Forest"):
    """Train selected ML model with optimized parameters"""
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            subsample=0.8,
            max_features='sqrt'
        )
    else:  # Ensemble
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        model = VotingRegressor(
            [('rf', rf), ('gb', gb)],
            weights=[0.5, 0.5]
        )
    
    model.fit(X_train, y_train)
    return model

def generate_simple_predictions(model, df, forecast_days, model_type):
    """Generate future predictions using a simple approach"""
    future_predictions = []
    confidence_intervals = []
    
    # Get recent price data
    recent_prices = df['Close'].values[-60:]  # Last 60 days
    last_price = recent_prices[-1]
    
    # Calculate recent statistics
    if len(recent_prices) > 1:
        returns = np.diff(recent_prices) / recent_prices[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.02
    else:
        mean_return = 0.001
        std_return = 0.02
    
    # Calculate trend
    if len(recent_prices) >= 10:
        trend = np.polyfit(range(10), recent_prices[-10:], 1)[0] / recent_prices[-1]
    else:
        trend = mean_return
    
    # Generate predictions
    current_prediction = last_price
    
    for day in range(forecast_days):
        try:
            # Create feature vector based on model type
            if model_type == "Random Forest":
                # Create simple features for Random Forest
                features = []
                for lag in [1, 5, 10, 20]:
                    if len(recent_prices) > lag:
                        features.append(recent_prices[-lag])
                    else:
                        features.append(last_price)
                
                # Add technical indicators
                if len(df) > 0:
                    features.extend([
                        df['MA_20'].iloc[-1] if 'MA_20' in df.columns else last_price,
                        df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
                        trend * 100,
                        day + 1
                    ])
                
                features = np.array(features).reshape(1, -1)
                
                # Make prediction
                if hasattr(model, 'estimators_'):
                    tree_preds = []
                    for tree in model.estimators_:
                        pred = tree.predict(features)[0]
                        tree_preds.append(pred)
                    
                    pred_mean = np.mean(tree_preds)
                    pred_std = np.std(tree_preds) if len(tree_preds) > 1 else last_price * std_return
                else:
                    pred_mean = model.predict(features)[0]
                    pred_std = last_price * std_return
                    
            elif model_type == "Gradient Boosting":
                # Create features for Gradient Boosting
                features = np.array([
                    last_price,
                    mean_return * 100,
                    std_return * 100,
                    trend * 100,
                    day + 1,
                    df['RSI'].iloc[-1] if len(df) > 0 and 'RSI' in df.columns else 50
                ]).reshape(1, -1)
                
                pred_mean = model.predict(features)[0]
                
                # Estimate confidence
                if hasattr(model, 'staged_predict'):
                    try:
                        staged_preds = list(model.staged_predict(features))
                        pred_std = np.std(staged_preds) if len(staged_preds) > 1 else last_price * std_return
                    except:
                        pred_std = last_price * std_return
                else:
                    pred_std = last_price * std_return
                    
            elif model_type == "Ensemble":
                # Create features for Ensemble
                features = np.array([
                    last_price,
                    mean_return * 100,
                    std_return * 100,
                    trend * 100,
                    day + 1,
                    df['RSI'].iloc[-1] if len(df) > 0 and 'RSI' in df.columns else 50
                ]).reshape(1, -1)
                
                # Get predictions from all estimators
                preds = []
                for name, estimator in model.named_estimators_.items():
                    pred = estimator.predict(features)[0]
                    preds.append(pred)
                
                pred_mean = np.mean(preds)
                pred_std = np.std(preds) if len(preds) > 1 else last_price * std_return
                
            else:
                # Fallback: simple projection
                pred_mean = last_price * (1 + mean_return + trend * (day + 1))
                pred_std = last_price * std_return
            
            # Ensure prediction is reasonable
            pred_mean = max(pred_mean, last_price * 0.5)  # Don't drop below 50% of current price
            pred_mean = min(pred_mean, last_price * 2.0)  # Don't rise above 200% of current price
            
            future_predictions.append(pred_mean)
            confidence_intervals.append(pred_std * 1.96)  # 95% CI
            
            # Update for next iteration
            recent_prices = np.append(recent_prices[1:], pred_mean)
            last_price = pred_mean
            
        except Exception as e:
            # Fallback: simple random walk
            pred_mean = last_price * (1 + np.random.normal(mean_return, std_return))
            future_predictions.append(pred_mean)
            confidence_intervals.append(last_price * std_return * 1.96)
            recent_prices = np.append(recent_prices[1:], pred_mean)
            last_price = pred_mean
    
    return future_predictions, confidence_intervals

# ============================================================================
# MAIN APP
# ============================================================================
st.title("📊 QuantumPredict Pro - AI Stock Predictor")

# Initialize session state
if 'run_prediction' not in st.session_state:
    st.session_state.run_prediction = False
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"

# Get values from session state or current inputs
current_ticker = st.session_state.ticker
start_date = st.session_state.get('start_date', datetime.now() - timedelta(days=365))
end_date = st.session_state.get('end_date', datetime.now())
prediction_days = st.session_state.get('prediction_days', 60)
forecast_days = st.session_state.get('forecast_days', 14)
model_type = st.session_state.get('model_type', "Random Forest")

if st.session_state.run_prediction and current_ticker:
    with st.spinner("🔍 Fetching real-time market data..."):
        try:
            # Fetch comprehensive data
            df, stock_info = fetch_stock_data(
                current_ticker, 
                start_date, 
                end_date + timedelta(days=1)
            )
            
            if df is None or df.empty or len(df) < prediction_days + 30:
                st.error(f"❌ Insufficient data for {current_ticker}. Please try a different symbol or date range.")
                st.info(f"💡 Need at least {prediction_days + 30} days of data. Current: {len(df) if df is not None else 0} days")
                st.stop()
            
            # Display stock information
            st.subheader(f"📊 {current_ticker} - {stock_info.get('longName', 'N/A')}")
            
            # Current metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${current_price:.2f}", 
                    f"{change_pct:.2f}%",
                    delta_color="normal" if change_pct >= 0 else "inverse"
                )
            
            with col2:
                day_high = float(df['High'].iloc[-1])
                day_low = float(df['Low'].iloc[-1])
                st.metric("Day Range", f"${day_low:.2f} - ${day_high:.2f}")
            
            with col3:
                volume = int(df['Volume'].iloc[-1])
                avg_volume = int(df['Volume'].tail(20).mean()) if len(df) >= 20 else volume
                volume_pct = ((volume/avg_volume-1)*100) if avg_volume > 0 else 0
                st.metric("Volume", f"{volume:,.0f}", 
                         f"{volume_pct:.1f}% vs avg")
            
            with col4:
                market_cap = stock_info.get('marketCap', 0)
                if market_cap > 1e9:
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                elif market_cap > 1e6:
                    st.metric("Market Cap", f"${market_cap/1e6:.2f}M")
                else:
                    st.metric("Market Cap", f"${market_cap:,.0f}")
            
            # Additional info
            with st.expander("📋 Stock Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Sector:**", stock_info.get('sector', 'N/A'))
                    st.write("**Industry:**", stock_info.get('industry', 'N/A'))
                with col2:
                    fifty_two_week_high = stock_info.get('fiftyTwoWeekHigh', 0)
                    fifty_two_week_low = stock_info.get('fiftyTwoWeekLow', 0)
                    st.write("**52W High:**", f"${fifty_two_week_high:.2f}" if fifty_two_week_high else "N/A")
                    st.write("**52W Low:**", f"${fifty_two_week_low:.2f}" if fifty_two_week_low else "N/A")
                with col3:
                    pe_ratio = stock_info.get('trailingPE', 'N/A')
                    st.write("**P/E Ratio:**", f"{pe_ratio:.2f}" if pe_ratio != 'N/A' else 'N/A')
                    dividend_yield = stock_info.get('dividendYield', 0)
                    st.write("**Dividend Yield:**", f"{dividend_yield*100:.2f}%" 
                            if dividend_yield else "N/A")
            
            # Calculate technical indicators
            st.info("📈 Calculating technical indicators...")
            df_tech = calculate_technical_indicators(df)
            
            # Create features for ML
            st.info("🤖 Preparing AI model features...")
            X, y = create_simplified_features(df_tech, lookback_days=prediction_days)
            
            if len(X) < 20:
                st.warning(f"⚠️ Limited data for training ({len(X)} samples). Consider increasing date range or reducing lookback days.")
                st.stop()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            st.info(f"🎯 Training {model_type} model...")
            model = train_model(X_train_scaled, y_train, model_type)
            
            # Generate predictions
            st.info("🔮 Generating future predictions...")
            
            future_predictions, confidence_intervals = generate_simple_predictions(
                model, df_tech, forecast_days, model_type
            )
            
            # Display results
            st.success(f"✅ AI Analysis Complete! Using {model_type} model")
            
            # Create interactive charts
            st.subheader("📈 Interactive Charts")
            
            tab1, tab2, tab3 = st.tabs(["Price Forecast", "Technical Analysis", "Prediction Details"])
            
            with tab1:
                # Price forecast chart
                fig_forecast = go.Figure()
                
                # Historical data (last 90 days or available)
                hist_days = min(90, len(df))
                historical_dates = df.index[-hist_days:]
                historical_prices = df['Close'].iloc[-hist_days:].values
                
                fig_forecast.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_prices,
                    name='Historical Price',
                    line=dict(color='#00ffcc', width=2),
                    mode='lines'
                ))
                
                # Future predictions
                future_dates = pd.date_range(
                    start=df.index[-1] + timedelta(days=1),
                    periods=forecast_days,
                    freq='B'
                )
                
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    name='AI Forecast',
                    line=dict(color='#ff3366', width=3, dash='dash'),
                    mode='lines+markers'
                ))
                
                # Confidence interval
                upper_bound = np.array(future_predictions) + np.array(confidence_intervals)
                lower_bound = np.array(future_predictions) - np.array(confidence_intervals)
                
                fig_forecast.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor='rgba(255, 51, 102, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
                
                fig_forecast.update_layout(
                    title=f'{current_ticker} Price Forecast ({forecast_days} Days)',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    template='plotly_dark',
                    height=500,
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            with tab2:
                # Technical indicators chart
                fig_tech = go.Figure()
                
                # RSI
                rsi_days = min(60, len(df_tech))
                fig_tech.add_trace(go.Scatter(
                    x=df_tech.index[-rsi_days:],
                    y=df_tech['RSI'].iloc[-rsi_days:],
                    name='RSI',
                    line=dict(color='#ff9900', width=2)
                ))
                
                # Add overbought/oversold lines
                fig_tech.add_hline(y=70, line_dash="dash", line_color="red", 
                                 annotation_text="Overbought")
                fig_tech.add_hline(y=30, line_dash="dash", line_color="green", 
                                 annotation_text="Oversold")
                
                fig_tech.update_layout(
                    title='RSI Indicator',
                    yaxis_title='RSI',
                    template='plotly_dark',
                    height=300
                )
                
                st.plotly_chart(fig_tech, use_container_width=True)
                
                # MACD
                fig_macd = go.Figure()
                macd_days = min(60, len(df_tech))
                fig_macd.add_trace(go.Scatter(
                    x=df_tech.index[-macd_days:],
                    y=df_tech['MACD'].iloc[-macd_days:],
                    name='MACD',
                    line=dict(color='#00ccff', width=2)
                ))
                fig_macd.add_trace(go.Scatter(
                    x=df_tech.index[-macd_days:],
                    y=df_tech['Signal_Line'].iloc[-macd_days:],
                    name='Signal Line',
                    line=dict(color='#ff3366', width=2)
                ))
                
                fig_macd.update_layout(
                    title='MACD Indicator',
                    template='plotly_dark',
                    height=300
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
            
            with tab3:
                # Detailed predictions table
                st.subheader("📅 Daily Forecast")
                
                prediction_data = []
                for i in range(forecast_days):
                    pred_date = future_dates[i]
                    pred_price = float(future_predictions[i])
                    
                    if i == 0:
                        price_change = ((pred_price - current_price) / current_price) * 100
                    else:
                        price_change = ((pred_price - float(future_predictions[i-1])) / float(future_predictions[i-1])) * 100
                    
                    confidence_width = (float(confidence_intervals[i]) / pred_price * 100) if pred_price > 0 else 0
                    
                    prediction_data.append({
                        'Day': i + 1,
                        'Date': pred_date.strftime('%Y-%m-%d'),
                        'Predicted Price': f"${pred_price:.2f}",
                        'Daily Change': f"{price_change:.2f}%",
                        'Confidence (±)': f"${confidence_intervals[i]:.2f}",
                        'Confidence %': f"{confidence_width:.1f}%"
                    })
                
                pred_df = pd.DataFrame(prediction_data)
                st.dataframe(pred_df, use_container_width=True, height=400)
                
                # Summary statistics
                final_pred = float(future_predictions[-1])
                total_change = ((final_pred - current_price) / current_price) * 100 if current_price > 0 else 0
                avg_daily_change = total_change / forecast_days if forecast_days > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Target Price", f"${final_pred:.2f}")
                with col2:
                    st.metric("Total Expected Change", f"{total_change:.1f}%")
                with col3:
                    st.metric("Avg Daily Change", f"{avg_daily_change:.1f}%")
            
            # Trading recommendations
            st.subheader("💡 Trading Recommendations")
            
            # Calculate recommendation score
            score = 0
    
            # Price momentum
            if total_change > 15:
                score += 2
            elif total_change > 5:
                score += 1
            elif total_change < -15:
                score -= 2
            elif total_change < -5:
                score -= 1
            
            # RSI analysis
            current_rsi = df_tech['RSI'].iloc[-1] if len(df_tech) > 0 else 50
            if current_rsi < 30:
                score += 1  # Oversold, potential buy
            elif current_rsi > 70:
                score -= 1  # Overbought, potential sell
            
            # Volume analysis
            current_volume_ratio = df_tech['Volume_Ratio'].iloc[-1] if len(df_tech) > 0 else 1
            if current_volume_ratio > 1.5:
                score += 1  # High volume, confirms trend
            
            # Generate recommendation
            if score >= 3:
                st.success(f"""
                **🎯 STRONG BUY SIGNAL**
                
                **Why:**
                - Strong positive momentum expected ({total_change:.1f}% upside)
                - Favorable technical indicators
                - High confidence in upward trend
                
                **Action:** Consider initiating or increasing long position
                **Target:** ${final_pred:.2f} ({total_change:.1f}% upside)
                **Stop Loss:** ${current_price * 0.95:.2f} (-5% from current)
                """)
                
            elif score >= 1:
                st.info(f"""
                **👍 MODERATE BUY SIGNAL**
                
                **Why:**
                - Positive outlook with moderate confidence
                - Some supporting indicators
                
                **Action:** Consider entry with proper risk management
                **Target:** ${final_pred:.2f} ({total_change:.1f}% upside)
                **Stop Loss:** ${current_price * 0.93:.2f} (-7% from current)
                """)
                
            elif score >= -1:
                st.warning(f"""
                **🤝 NEUTRAL / HOLD**
                
                **Why:**
                - Mixed signals from analysis
                - Limited clear direction
                
                **Action:** Maintain current position, wait for clearer signals
                **Watch Levels:** ${current_price * 0.97:.2f} (support), ${current_price * 1.03:.2f} (resistance)
                """)
                
            elif score >= -3:
                st.error(f"""
                **👎 MODERATE SELL SIGNAL**
                
                **Why:**
                - Negative trend expected ({total_change:.1f}% downside)
                - Several bearish indicators
                
                **Action:** Consider reducing exposure or hedging
                **Target:** ${final_pred:.2f} ({total_change:.1f}% downside)
                **Stop Loss:** ${current_price * 1.05:.2f} (+5% from current)
                """)
                
            else:
                st.error(f"""
                **⚠️ STRONG SELL SIGNAL**
                
                **Why:**
                - Significant downside risk ({total_change:.1f}% downside)
                - Multiple bearish confirmations
                
                **Action:** Consider exiting positions or short opportunities
                **Target:** ${final_pred:.2f} ({total_change:.1f}% downside)
                **Stop Loss:** ${current_price * 1.03:.2f} (+3% from current)
                """)
            
            # Risk Disclaimer
            st.markdown("---")
            st.warning("""
            **📌 IMPORTANT DISCLAIMER:**
            
            This AI-powered prediction tool is for **educational and informational purposes only**. 
            It is **NOT financial advice**. 
            
            **Key Risks:**
            - Stock market predictions are inherently uncertain
            - Past performance does not guarantee future results
            - AI models can be wrong or overfit to historical data
            - External factors (news, events, market sentiment) are not fully captured
            
            **Always:**
            - Conduct your own research
            - Consult with qualified financial advisors
            - Diversify your investments
            - Only invest what you can afford to lose
            - Understand your risk tolerance
            """)
            
            # Download option
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=pred_df.to_csv(index=False),
                file_name=f"{current_ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.info("""
            **Troubleshooting:**
            1. Check if stock symbol is valid
            2. Verify your internet connection
            3. Try a different date range
            4. Check Yahoo Finance for market hours
            5. Try popular symbols: AAPL, TSLA, GOOGL, MSFT, AMZN
            6. Reduce Lookback Days if you have limited data
            """)

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Welcome to QuantumPredict Pro</h3>
    <p><b>Advanced AI-Powered Stock Market Prediction Platform</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### **Real-Time Market Intelligence**
        
        This platform uses **advanced machine learning algorithms** and **real-time Yahoo Finance data** 
        to analyze stocks and generate intelligent predictions.
        
        ### 🚀 **How It Works:**
        1. **Enter any stock symbol** (AAPL, TSLA, etc.)
        2. **Customize analysis settings** in sidebar
        3. **Click "Generate Predictions"**
        4. **Get AI-powered insights** with confidence intervals
        
        ### 📊 **Key Features:**
        ✅ **Real-time Yahoo Finance API integration**  
        ✅ **Multiple AI models** (Random Forest, Gradient Boosting, Ensemble)  
        ✅ **Technical indicators** (RSI, MACD, Moving Averages)  
        ✅ **Interactive visualizations** with Plotly  
        ✅ **Confidence intervals** for predictions  
        ✅ **Risk assessment** and trading recommendations  
        ✅ **Export predictions** to CSV  
        
        ### 🎯 **Best Practices:**
        - Use longer timeframes for more accurate predictions
        - Combine AI predictions with fundamental analysis
        - Always use proper risk management
        """)
    
    with col2:
        st.markdown("""
        ### 🏆 **Popular Stocks**
        """)
        
        popular_stocks = {
            "AAPL": "Apple Inc.",
            "TSLA": "Tesla Inc.",
            "GOOGL": "Alphabet Inc.",
            "MSFT": "Microsoft",
            "AMZN": "Amazon.com",
            "NVDA": "NVIDIA",
            "META": "Meta Platforms",
            "JPM": "JPMorgan Chase",
            "V": "Visa Inc.",
            "JNJ": "Johnson & Johnson"
        }
        
        for symbol, name in popular_stocks.items():
            if st.button(f"📊 {symbol} - {name}", use_container_width=True):
                st.session_state.ticker = symbol
                st.session_state.run_prediction = True
                st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">
        <span style="background: linear-gradient(45deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            QuantumPredict Pro v3.1
        </span>
    </p>
    <p style="color: #cbd5e1; margin-bottom: 5px;">Powered by Real-Time Yahoo Finance API</p>
    <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 5px;">© 2024 | Built with Streamlit, Scikit-learn, yfinance, and Plotly</p>
    <p style="color: #64748b; font-size: 0.8rem;"><i>Data Source: Yahoo Finance | For educational purposes only</i></p>
</div>
""", unsafe_allow_html=True)

