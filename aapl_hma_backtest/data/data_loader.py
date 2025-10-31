"""
Data Loader for Stock Prices

Fetches historical stock data using Alpaca
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Alpaca

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # Get API credentials from environment - try both live and paper
        api_key = os.environ.get('ALPACA_API_KEY') or os.environ.get('ALPACA_PAPER_API_KEY')
        api_secret = os.environ.get('ALPACA_API_SECRET') or os.environ.get('ALPACA_PAPER_API_SECRET')

        if not api_key or not api_secret:
            raise ValueError("Alpaca API credentials not found")

        # Create client for historical data
        client = StockHistoricalDataClient(api_key, api_secret)

        # Create request
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d')
        )

        # Fetch data
        bars = client.get_stock_bars(request)

        # Convert to DataFrame
        df = bars.df

        # If multi-index (symbol, timestamp), reset to use timestamp as index
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        # Rename columns to match expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Keep only OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        return df

    except ImportError:
        print("alpaca-py not installed. Using synthetic data for demonstration.")
        return generate_synthetic_data(start_date, end_date)
    except Exception as e:
        print(f"Error fetching data from Alpaca: {e}")
        print("Using synthetic data for demonstration.")
        return generate_synthetic_data(start_date, end_date)


def generate_synthetic_data(start_date: str, end_date: str,
                           initial_price: float = 75.0) -> pd.DataFrame:
    """
    Generate synthetic stock data mimicking AAPL's behavior

    Args:
        start_date: Start date
        end_date: End date
        initial_price: Starting price (AAPL was ~$75 in Jan 2020)

    Returns:
        DataFrame with synthetic OHLCV data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate price using geometric Brownian motion
    np.random.seed(42)
    n_days = len(dates)

    # AAPL-like parameters (went from ~$75 to ~$230 over 2020-2024)
    mu = 0.00075  # Annualized ~20% growth
    sigma = 0.018  # ~28% annualized volatility

    # Generate returns
    returns = np.random.normal(mu, sigma, n_days)

    # Add regime changes (bull/bear periods)
    regime = np.zeros(n_days)
    regime[:int(n_days*0.3)] = 0.0005  # Strong bull 2020-2021
    regime[int(n_days*0.3):int(n_days*0.5)] = -0.0002  # Bear 2022
    regime[int(n_days*0.5):] = 0.0003  # Recovery 2023-2024

    returns = returns + regime

    # Generate price path
    price = initial_price * np.exp(np.cumsum(returns))

    # Add some realistic noise
    noise = np.random.normal(0, 0.5, n_days)
    price = price + noise

    # Generate OHLC from close prices
    df = pd.DataFrame({
        'Close': price,
        'Open': price * (1 + np.random.normal(0, 0.003, n_days)),
        'High': price * (1 + np.abs(np.random.normal(0.008, 0.004, n_days))),
        'Low': price * (1 - np.abs(np.random.normal(0.008, 0.004, n_days))),
        'Volume': np.random.randint(60000000, 120000000, n_days)
    }, index=dates)

    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

    return df


def load_and_prepare_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load and prepare data for backtesting

    Args:
        ticker: Stock ticker
        start_date: Start date
        end_date: End date

    Returns:
        Prepared DataFrame
    """
    df = fetch_stock_data(ticker, start_date, end_date)

    # Remove any NaN values
    df = df.dropna()

    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


if __name__ == '__main__':
    # Test data loading
    df = load_and_prepare_data('AAPL', '2023-01-01', '2024-01-01')
    print(f"Loaded {len(df)} days of data")
    print(df.head())
    print(df.tail())
