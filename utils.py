import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import mplfinance as mpf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional

# Base directory for extracted data (customize via env var if needed)
BASE_DIR = os.getenv("MASTER_THESIS_DIR", "data/extract")

# --- Data Loading Functions ---

def load_prices(
    ticker: str,
    frequency: str = 'daily',
    folder: str = 'stocks'
) -> pd.DataFrame:
    """
    Load adjusted stock price data for a given ticker from standard file structure.

    Returns DataFrame indexed by date with column 'adjusted_close'.
    """
    freq_map = {'daily': 'daily_adjusted', 'weekly': 'weekly_adjusted'}
    if frequency not in freq_map:
        raise ValueError("frequency must be 'daily' or 'weekly'")
    pattern = os.path.join(
        BASE_DIR, folder, ticker, 'prices', f"{ticker}_{freq_map[frequency]}.csv"
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No price file found for {ticker} ({frequency}) at {pattern}")
    df = pd.read_csv(files[0])
    # parse date
    date_col = next((c for c in df.columns if c.lower() in ('date','timestamp')), None)
    if date_col is None:
        raise ValueError("No date or timestamp column found in price file")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # detect adjusted close
    adj_col = next((c for c in df.columns if 'adjusted' in c.lower()), None)
    if adj_col:
        df.rename(columns={adj_col:'adjusted_close'}, inplace=True)
    elif 'close' in df.columns:
        df.rename(columns={'close':'adjusted_close'}, inplace=True)
    else:
        raise ValueError("No adjusted or close column found in price file")
    return df

# --- Sentiment Processing ---

def load_news(
    ticker: str,
    folder: str = 'news'
) -> Tuple[str, pd.DataFrame]:
    """
    Locate the latest news CSV for a ticker and load raw DataFrame.
    Returns tuple(path, raw DataFrame).
    """
    pattern = os.path.join(BASE_DIR, folder, f"{ticker}_news_*_to_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No news file found for {ticker} at {pattern}")
    path = sorted(files)[-1]
    df = pd.read_csv(path, sep=';')
    return path, df


from typing import Optional, Tuple
import pandas as pd
import json

def explode_sentiment_data(
    ticker: str,
    folder: str = 'news',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sentiment_lower: Optional[float] = None,
    sentiment_upper: Optional[float] = None,
    min_relevance_score: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explode JSON sentiment for a ticker, filter by date range and sentiment/relevance thresholds.
    sentiment_lower: keep entries with score <= sentiment_lower (e.g. -0.3)
    sentiment_upper: keep entries with score >= sentiment_upper (e.g. 0.3)
    """
    _, df_raw = load_news(ticker, folder=folder)
    if 'time_published' not in df_raw.columns:
        raise ValueError("Column 'time_published' not found in news data")

    df_raw['time_published'] = pd.to_datetime(df_raw['time_published'])

    rows = []
    for _, r in df_raw.iterrows():
        try:
            sentiments = json.loads(r['ticker_sentiment'].replace("'", '"'))
        except:
            continue
        for sent in sentiments:
            if sent.get('ticker') == ticker:
                tss = float(sent.get('ticker_sentiment_score', 0))
                trs = float(sent.get('relevance_score', 0))
                rows.append({
                    'time_published': r['time_published'],
                    'day_date': r['time_published'].date(),
                    'week_date': pd.to_datetime(r['time_published']).to_period('W').start_time.date(),
                    'title': r.get('title') or r.get('title_raw', ''),
                    'summary': r.get('summary', ''),
                    'ticker_sentiment_score': tss,
                    'ticker_relevance_score': trs
                })

    exploded_df = pd.DataFrame(rows)
    if exploded_df.empty:
        raise ValueError(f"No sentiment entries found for {ticker}")

    # Filter logic
    filtered = exploded_df.copy()
    if start_date:
        sd = pd.to_datetime(start_date).date()
        filtered = filtered[filtered['day_date'] >= sd]
    if end_date:
        ed = pd.to_datetime(end_date).date()
        filtered = filtered[filtered['day_date'] <= ed]
    if sentiment_lower is not None and sentiment_upper is not None:
        filtered = filtered[
            (filtered['ticker_sentiment_score'] <= sentiment_lower) |
            (filtered['ticker_sentiment_score'] >= sentiment_upper)
        ]
    elif sentiment_lower is not None:
        filtered = filtered[filtered['ticker_sentiment_score'] <= sentiment_lower]
    elif sentiment_upper is not None:
        filtered = filtered[filtered['ticker_sentiment_score'] >= sentiment_upper]
    if min_relevance_score is not None:
        filtered = filtered[filtered['ticker_relevance_score'] >= min_relevance_score]

    return exploded_df, filtered



def aggregate_sentiment(
    exploded_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        exploded_df
        .groupby('day_date')[['ticker_sentiment_score','ticker_relevance_score']]
        .mean().reset_index()
        .rename(columns={'ticker_sentiment_score':'daily_tss','ticker_relevance_score':'daily_trs'})
    )
    weekly = (
        exploded_df
        .groupby('week_date')[['ticker_sentiment_score','ticker_relevance_score']]
        .mean().reset_index()
        .rename(columns={'ticker_sentiment_score':'weekly_tss','ticker_relevance_score':'weekly_trs'})
    )
    return daily, weekly

# --- Analysis & Plotting Functions ---

def plot_multi_prices(
    tickers: List[str],
    frequency: str = 'daily',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    months_back: Optional[int] = None,
    sma_windows: Optional[List[int]] = None,
    ema_spans: Optional[List[int]] = None,
    candlestick: bool = False,
    title: Optional[str] = None
) -> None:
    """
    Plot price series for multiple tickers with optional SMA/EMA.
    If candlestick=True, draws OHLC candlesticks with overlays.
    Otherwise, draws line charts with SMA/EMA overlays.
    """
    # Determine date range
    if months_back is not None:
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            df0 = load_prices(tickers[0], frequency)
            end = df0.index.max()
        start = end - pd.DateOffset(months=months_back)
    else:
        start = pd.to_datetime(start_date) if start_date else None
        end   = pd.to_datetime(end_date)   if end_date   else None

    if candlestick:
        import mplfinance as mpf
        n = len(tickers)
        fig, axes = plt.subplots(n, 1, figsize=(12, 5*n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, ticker in zip(axes, tickers):
            df = load_prices(ticker, frequency)
            df = df.loc[
                (start or df.index.min()):
                (end   or df.index.max())
            ]
            # detect OHLC columns
            open_col  = next(c for c in df.columns if 'open'  in c.lower())
            high_col  = next(c for c in df.columns if 'high'  in c.lower())
            low_col   = next(c for c in df.columns if 'low'   in c.lower())
            close_col = next(c for c in df.columns if 'close' in c.lower() and 'adjusted' not in c.lower())
            ohlc = df[[open_col, high_col, low_col, close_col]].copy()
            ohlc.columns = ['Open','High','Low','Close']
            addplots = []
            if sma_windows:
                for w in sma_windows:
                    sma = ohlc['Close'].rolling(w).mean()
                    addplots.append(mpf.make_addplot(sma, ax=ax, linestyle='--', label=f'SMA{w}'))
            if ema_spans:
                for span in ema_spans:
                    ema = ohlc['Close'].ewm(span=span, adjust=False).mean()
                    addplots.append(mpf.make_addplot(ema, ax=ax, linestyle=':', label=f'EMA{span}'))
            mpf.plot(
                ohlc,
                type='candle',
                ax=ax,
                addplot=addplots or None,
                show_nontrading=False,
                style='yahoo',
                title=title or f"{ticker} Candlestick",
                ylabel='Price',
                volume=False
            )
        plt.tight_layout()
        plt.show()
        return

    # fallback to line plots
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        df = load_prices(ticker, frequency)
        df = df.loc[
            (start or df.index.min()):
            (end   or df.index.max())
        ]
        plt.plot(df.index, df['adjusted_close'], label=f'{ticker} Price')
        if sma_windows:
            for w in sma_windows:
                sma = df['adjusted_close'].rolling(w).mean()
                plt.plot(df.index, sma, '--', label=f'{ticker} SMA{w}')
        if ema_spans:
            for span in ema_spans:
                ema = df['adjusted_close'].ewm(span=span, adjust=False).mean()
                plt.plot(df.index, ema, ':', label=f'{ticker} EMA{span}')
    plt.title(title or 'Price Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_stationary_with_sentiment(
    tickers: List[str],
    frequency: str = 'daily',
    detrend_method: str = 'log_return',
    sentiment_agg: str = 'daily',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    months_back: Optional[int] = None,
    sentiment_lower: Optional[float] = None,
    sentiment_upper: Optional[float] = None,
    min_relevance_score: Optional[float] = None,
    title: str = 'Stationary vs Sentiment'
) -> None:
    """
    Plot detrended price series, sentiment aggregates, combined normalized plot, and correlations.

    Includes sentiment filtering by thresholds.
    """
    if months_back is not None and (start_date or end_date):
        raise ValueError("Specify either months_back or start_date/end_date, not both")

    price_data = {}
    sentiment_data = {}
    for ticker in tickers:
        df = load_prices(ticker, frequency=frequency)
        # date slicing
        if months_back is not None:
            last = df.index.max()
            first = last - pd.DateOffset(months=months_back)
        else:
            first = pd.to_datetime(start_date) if start_date else df.index.min()
            last = pd.to_datetime(end_date) if end_date else df.index.max()
        df_slice = df.loc[first:last]
        # detrend
        if detrend_method == 'difference':
            ts = df_slice['adjusted_close'].diff().dropna()
        elif detrend_method == 'log_return':
            ts = np.log(df_slice['adjusted_close'] / df_slice['adjusted_close'].shift(1)).dropna()
        elif detrend_method == 'demean':
            window = 30
            ts = df_slice['adjusted_close'] - df_slice['adjusted_close'].rolling(window).mean()
            ts = ts.dropna()
        else:
            raise ValueError("Invalid detrend_method")
        price_data[ticker] = ts
        # sentiment
        _, filt = explode_sentiment_data(
            ticker,
            start_date=first.date().isoformat(),
            end_date=last.date().isoformat(),
            sentiment_lower=sentiment_lower,
            sentiment_upper=sentiment_upper,
            min_relevance_score=min_relevance_score
        )
        daily, weekly = aggregate_sentiment(filt)
        if sentiment_agg == 'daily':
            ser = daily.set_index('day_date')['daily_tss']
        else:
            ser = weekly.set_index('week_date')['weekly_tss']
        ser = ser.reindex(ts.index.date, method='nearest')
        sentiment_data[ticker] = pd.Series(ser.values, index=ts.index)

    # plot separate
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8),sharex=True)
    for t, series in price_data.items(): ax1.plot(series.index, series, label=t)
    ax1.set_title(f"Detrended Prices ({detrend_method})"); ax1.legend(); ax1.grid(True)
    for t, series in sentiment_data.items(): ax2.plot(series.index, series, label=t)
    ax2.set_title(f"Sentiment ({sentiment_agg})"); ax2.legend(); ax2.grid(True)
    fig.suptitle(title); plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show()

    # combined normalized + correlation
    fig2, ax3 = plt.subplots(figsize=(12,4))
    for t in tickers:
        pd_norm = (price_data[t] - price_data[t].mean()) / price_data[t].std()
        sent_norm = (sentiment_data[t] - sentiment_data[t].mean()) / sentiment_data[t].std()
        ax3.plot(pd_norm.index, pd_norm, label=f'{t} price (norm)')
        ax3.plot(sent_norm.index, sent_norm, '--', label=f'{t} sentiment (norm)')
        corr = pd_norm.corr(sent_norm)
        print(f'Correlation {t}: {corr:.4f}')
    ax3.set_title(f'{t} - Normalized Price vs Sentiment - Alpha Vantage- (Last 6 Months, Pearson r={corr:.3f})'); ax3.set_xlabel('Date'); ax3.set_ylabel('Normalized Value'); ax3.legend(); ax3.grid(True); plt.show()

def plot_stationary_with_sentiment_df(
    price_ticker: str,
    df_news: pd.DataFrame,
    frequency: str = 'daily',
    detrend_method: str = 'log_return',
    sentiment_agg: str = 'daily',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    title: str = 'Stationary vs Sentiment'
) -> None:
    """
    Plot detrended price series of price_ticker with aggregated sentiment from df_news.

    df_news must contain 'day_date', 'weekly_tss', 'daily_tss' columns.
    """
    # 1. Cargar precios SPY
    df_price = load_prices(price_ticker, frequency=frequency)

    # 2. Filtrado por fechas
    first = pd.to_datetime(start_date) if start_date else df_price.index.min()
    last = pd.to_datetime(end_date) if end_date else df_price.index.max()
    df_slice = df_price.loc[first:last]

    # 3. Estacionarizar precios
    if detrend_method == 'difference':
        ts_price = df_slice['adjusted_close'].diff().dropna()
    elif detrend_method == 'log_return':
        ts_price = np.log(df_slice['adjusted_close'] / df_slice['adjusted_close'].shift(1)).dropna()
    elif detrend_method == 'demean':
        window = 30
        ts_price = df_slice['adjusted_close'] - df_slice['adjusted_close'].rolling(window).mean()
        ts_price = ts_price.dropna()
    else:
        raise ValueError("Invalid detrend_method")

    # 4. Agregar sentimiento desde df_news (ya filtrado o full)
    daily_sentiment, weekly_sentiment = aggregate_sentiment(df_news)

    if sentiment_agg == 'daily':
        sentiment_series = daily_sentiment.set_index('day_date')['daily_tss']
    else:
        sentiment_series = weekly_sentiment.set_index('week_date')['weekly_tss']

    # 5. Alinear sentimiento con fechas de precios
    sentiment_series = sentiment_series.reindex(ts_price.index.date, method='nearest')
    ts_sentiment = pd.Series(sentiment_series.values, index=ts_price.index)

    # 6. Plot series separadas
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8),sharex=True)

    ax1.plot(ts_price.index, ts_price, label=f'{price_ticker} Price')
    ax1.set_title(f"Detrended {price_ticker} Prices ({detrend_method})")
    ax1.grid(True)

    ax2.plot(ts_sentiment.index, ts_sentiment, label='Sentiment Aggregate', color='orange')
    ax2.set_title(f"Sentiment Aggregate ({sentiment_agg})")
    ax2.grid(True)

    fig.suptitle(title)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

    # 7. Plot normalizado + correlación
    fig2, ax3 = plt.subplots(figsize=(12,4))

    price_norm = (ts_price - ts_price.mean()) / ts_price.std()
    sent_norm = (ts_sentiment - ts_sentiment.mean()) / ts_sentiment.std()

    ax3.plot(price_norm.index, price_norm, label=f'{price_ticker} Price (norm)')
    ax3.plot(sent_norm.index, sent_norm, '--', label='Sentiment (norm)')

    corr = price_norm.corr(sent_norm)
    print(f'Correlation {price_ticker}: {corr:.4f}')

    ax3.set_title(f'{price_ticker} - Normalized Price vs Sentiment')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Normalized Value')
    ax3.legend()
    ax3.grid(True)

    plt.show()

