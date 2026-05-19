import os
import pandas as pd
import numpy as np
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import triple_barrier_labeling, prepare_features_and_labels
from src.data_ingestion import fetch_stock_data
from src.train_benchmark import run_benchmarking

def test_pipeline_smoke():
    """
    Smoke test to verify the end-to-end flow.
    """
    ticker = "AAPL"
    start = "2023-01-01"
    end = "2023-12-31" # Use a fixed period for testing
    
    print("--- Starting Verification Smoke Test ---")
    
    # 1. Test Ingestion
    print("\n1. Testing Data Ingestion...")
    raw_file = fetch_stock_data(ticker, start, end, allow_synthetic_fallback=True)
    if raw_file and os.path.exists(raw_file):
        print("PASS: Data Ingestion")
    else:
        print("FAIL: Data Ingestion")
        return

    # 2. Test Feature Engineering
    print("\n2. Testing Feature Engineering...")
    df = pd.read_parquet(raw_file)
    df_feat = add_technical_indicators(df)
    required_cols = ['RSI', 'MACD', 'BB_Upper', 'Volatility']
    if all(col in df_feat.columns for col in required_cols):
        print("PASS: Feature Engineering")
    else:
        print("FAIL: Feature Engineering")
        return

    # 3. Test Preprocessing & Benchmarking
    print("\n3. Testing Benchmarking Pipeline...")
    results = run_benchmarking(ticker, start, end)
    if results is not None and not results.empty:
        print("PASS: Benchmarking Pipeline")
    else:
        print("FAIL: Benchmarking Pipeline")
        return

    print("\n--- Smoke Test Completed Successfully ---")

if __name__ == "__main__":
    test_pipeline_smoke()
