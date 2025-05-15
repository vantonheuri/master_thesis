import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta

######################
## DATA FUNCTIONS ##
######################

def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()

    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, group_by='ticker')
    else:
        data = yf.download(ticker, period=period, interval=interval, group_by='ticker')

    # ✅ Si columnas tienen MultiIndex como (Price, Ticker), reorganizar correctamente
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Extrae columnas del ticker
            data = data.xs(key=ticker, level='Ticker', axis=1)
            data.columns.name = None  # elimina el nombre 'Price'
        except Exception as e:
            st.warning(f"⚠️ Could not extract {ticker}: {e}")
            return pd.DataFrame()

    # Verificación final
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(data.columns):
        st.warning(f"⚠️ Missing columns for {ticker}. Found: {data.columns}")
        return pd.DataFrame()

    return data

def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def calculate_metrics(data):
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data):
    # Forzar 'Close' a una serie válida
    if isinstance(data['Close'], pd.DataFrame):
        data['Close'] = data['Close'].squeeze()

    if isinstance(data['Close'], pd.Series) and data['Close'].ndim == 1:
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    else:
        st.error("❌ 'Close' column is not 1-dimensional. Cannot calculate SMA/EMA.")
        st.dataframe(data.head())
        st.stop()

    return data



######################
## APP LAYOUT ##
######################

st.set_page_config(layout="wide")
st.title('Real Time Stock Dashboard')

######## SIDEBAR ########

st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'ADBE')
time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])

interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

######## MAIN SECTION ########

if st.sidebar.button('Update'):
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    if data.empty:
        st.warning(f"⚠️ No data found for {ticker} with time period '{time_period}'. Try another.")
        st.stop()

    data = process_data(data)
    data = add_technical_indicators(data)

    if 'Datetime' not in data.columns or 'Close' not in data.columns:
        st.error("❌ Required columns missing after processing.")
        st.dataframe(data.head())
        st.stop()

    last_close, change, pct_change, high, low, volume = calculate_metrics(data)

    st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", delta=f"{change:.2f} ({pct_change:.2f}%)")

    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{high:.2f} USD")
    col2.metric("Low", f"{low:.2f} USD")
    col3.metric("Volume", f"{volume:,}")

    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data['Datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        ))
    else:
        fig = px.line(data, x='Datetime', y='Close', title=f"{ticker} Price Over Time")

    for indicator in indicators:
        if indicator == 'SMA 20':
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
        elif indicator == 'EMA 20':
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))

    fig.update_layout(
        title=f'{ticker} {time_period.upper()} Chart',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader('📊 Data Preview')
    st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10))

    st.subheader('📈 Technical Indicators')
    st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']].tail(10))


######## REAL-TIME STOCK PRICES ########

st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']

for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '1m')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        try:
            last_price = float(real_time_data['Close'].iloc[-1])
            open_price = float(real_time_data['Open'].iloc[0])
            change = last_price - open_price
            pct_change = (change / open_price) * 100
            st.sidebar.metric(
                label=f"{symbol}",
                value=f"{last_price:.2f} USD",
                delta=f"{change:.2f} ({pct_change:.2f}%)"
            )
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load {symbol}: {e}")
    else:
        st.sidebar.warning(f"⚠️ No intraday data for {symbol}.")

st.sidebar.subheader('About')
st.sidebar.info(
    'This dashboard provides live stock data and technical analysis. '
    'Use the sidebar to select tickers, timeframes, and technical indicators.'
)
