import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

def _build_synthetic_ohlcv(start_date, end_date, seed=42):
    """Create deterministic OHLCV data for offline smoke tests."""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.015, size=len(dates))
    close = 100 * np.cumprod(1 + returns)
    open_prices = close * (1 + rng.normal(0, 0.003, size=len(dates)))
    high = np.maximum(open_prices, close) * (1 + rng.uniform(0.001, 0.01, size=len(dates)))
    low = np.minimum(open_prices, close) * (1 - rng.uniform(0.001, 0.01, size=len(dates)))
    volume = rng.integers(1_000_000, 5_000_000, size=len(dates))

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=pd.Index(dates, name="Date"),
    )

def _save_stock_data(data, ticker, start_date, end_date, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{ticker}_{start_date}_{end_date}.parquet"
    filepath = os.path.join(output_dir, filename)
    data.to_parquet(filepath)
    return filepath

def fetch_stock_data(ticker, start_date, end_date, output_dir="data/raw", allow_synthetic_fallback=False):
    """
    Fetches historical OHLCV data for a given ticker from Yahoo Finance.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            if not allow_synthetic_fallback:
                print(f"No data found for {ticker}.")
                return None

            print(f"No live data found for {ticker}; using deterministic synthetic data for verification.")
            data = _build_synthetic_ohlcv(start_date, end_date)

        filepath = _save_stock_data(data, ticker, start_date, end_date, output_dir)
        
        print(f"Successfully saved data to {filepath}")
        return filepath
    except Exception as e:
        if allow_synthetic_fallback:
            print(f"Error fetching data for {ticker}: {e}")
            print("Using deterministic synthetic data for verification.")
            data = _build_synthetic_ohlcv(start_date, end_date)
            filepath = _save_stock_data(data, ticker, start_date, end_date, output_dir)
            print(f"Successfully saved data to {filepath}")
            return filepath

        print(f"Error fetching data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    ticker_symbol = "AAPL"
    start = "2020-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    fetch_stock_data(ticker_symbol, start, end)
