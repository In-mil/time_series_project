#!/usr/bin/env python3
"""Make a real prediction using data from the dataset."""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Configuration
API_URL = "https://time-series-api-jgqkhpmk5q-ey.a.run.app"
DATA_PATH = Path(__file__).parent.parent / "data" / "final_data" / "20251115_dataset_crp.csv"
LOOK_BACK = 20


def load_and_prepare_data(ticker="BTC", n_samples=LOOK_BACK):
    """Load real data from CSV and prepare a sequence for prediction.

    Args:
        ticker: Crypto ticker symbol (e.g., 'BTC', 'ETH', 'btcusd', 'ethusd')
        n_samples: Number of timesteps to use (default: 20)

    Returns:
        Tuple of (sequence, metadata)
    """
    print(f"Loading data from: {DATA_PATH}")

    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please ensure the dataset is available.")
        sys.exit(1)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} rows")

    # Normalize ticker format (convert BTC -> btcusd, ETH -> ethusd)
    ticker_normalized = ticker.lower()
    if not ticker_normalized.endswith('usd'):
        ticker_normalized = ticker_normalized + 'usd'

    print(f"  Looking for ticker: {ticker_normalized}")

    # Filter for specific ticker
    df_ticker = df[df['ticker'] == ticker_normalized].copy()

    if len(df_ticker) == 0:
        available_tickers = df['ticker'].unique()
        print(f"ERROR: Ticker '{ticker_normalized}' not found in dataset")
        print(f"Available tickers: {', '.join(available_tickers[:10])}")
        sys.exit(1)

    print(f"  Found {len(df_ticker):,} rows for {ticker_normalized}")

    # Sort by date
    df_ticker = df_ticker.sort_values('date')

    # Get the latest n_samples rows
    df_sequence = df_ticker.tail(n_samples)

    if len(df_sequence) < n_samples:
        print(f"WARNING: Only {len(df_sequence)} samples available, need {n_samples}")
        print("Using all available samples...")

    # Define feature columns (exclude targets and metadata)
    target_columns = [
        'future_5_close_higher_than_today',
        'future_10_close_higher_than_today',
        'future_5_close_lower_than_today',
        'future_10_close_lower_than_today',
        'higher_close_today_vs_future_5_close',
        'higher_close_today_vs_future_10_close',
        'lower_close_today_vs_future_5_close',
        'lower_close_today_vs_future_10_close',
        'date',
        'ticker'
    ]

    feature_cols = [col for col in df_sequence.columns if col not in target_columns]

    # Extract features
    X = df_sequence[feature_cols].values

    # Convert to list for JSON serialization
    sequence = X.tolist()

    # Metadata
    metadata = {
        'ticker': ticker_normalized,
        'n_features': len(feature_cols),
        'n_timesteps': len(sequence),
        'date_range': f"{df_sequence['date'].iloc[0]} to {df_sequence['date'].iloc[-1]}",
        'feature_columns': feature_cols
    }

    return sequence, metadata


def make_prediction(sequence, api_url=API_URL):
    """Send prediction request to API.

    Args:
        sequence: List of timesteps with features
        api_url: API endpoint URL

    Returns:
        API response (dict) or None on error
    """
    payload = {"sequence": sequence}

    try:
        print(f"\nSending prediction request to API...")
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        return data

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def display_prediction(prediction, metadata):
    """Display prediction results in a nice format.

    Args:
        prediction: API response with predictions
        metadata: Data metadata
    """
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    print(f"\nInput Data:")
    print(f"  Ticker: {metadata['ticker']}")
    print(f"  Timesteps: {metadata['n_timesteps']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Date Range: {metadata['date_range']}")

    print(f"\n{'Ensemble Prediction':<20}: {prediction['prediction']:.4f}")

    print(f"\nComponent Models:")
    for model, pred in prediction['components'].items():
        bar_length = int(abs(pred) * 5)  # Scale for visualization
        bar = '█' * bar_length
        sign = '+' if pred > 0 else '-'
        print(f"  {model:<12}: {pred:>8.4f} {sign}{bar}")

    # Interpretation
    ensemble_pred = prediction['prediction']
    print(f"\nInterpretation:")
    if ensemble_pred > 0.5:
        print(f"  📈 Bullish signal (confidence: {ensemble_pred:.2f})")
        print(f"     Price likely to increase in next 5 days")
    elif ensemble_pred < -0.5:
        print(f"  📉 Bearish signal (confidence: {abs(ensemble_pred):.2f})")
        print(f"     Price likely to decrease in next 5 days")
    else:
        print(f"  ➡️  Neutral signal")
        print(f"     Uncertain price movement")

    print("=" * 60)


def main():
    """Main function."""
    # Parse command line arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    api_url = sys.argv[2] if len(sys.argv) > 2 else API_URL

    print("=" * 60)
    print("CRYPTO PRICE PREDICTION")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"API URL: {api_url}")
    print("=" * 60)

    # Load and prepare data
    sequence, metadata = load_and_prepare_data(ticker=ticker)

    print(f"\nPrepared sequence: {metadata['n_timesteps']} timesteps × {metadata['n_features']} features")

    # Make prediction
    prediction = make_prediction(sequence, api_url=api_url)

    if prediction is None:
        print("\nERROR: Prediction failed")
        return 1

    # Display results
    display_prediction(prediction, metadata)

    return 0


if __name__ == "__main__":
    sys.exit(main())
