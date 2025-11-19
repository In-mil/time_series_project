#!/usr/bin/env python3
"""Test script for the deployed time-series prediction API."""

import requests
import json
import numpy as np
from pathlib import Path
import sys

# Configuration
API_URL = "https://time-series-api-jgqkhpmk5q-ey.a.run.app"
LOOK_BACK = 20


def test_health_check():
    """Test the API health endpoint."""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{API_URL}/")
        response.raise_for_status()
        data = response.json()

        print(f"✓ Health check passed")
        print(f"  Status: {data.get('status')}")
        print(f"  Database enabled: {data.get('database_enabled')}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def create_sample_sequence():
    """Create a sample sequence for testing.

    In production, you would load real data from your CSV and preprocess it.
    This creates random data with the correct shape for demonstration.
    """
    # Get feature count from the data
    data_path = Path(__file__).parent.parent / "data" / "final_data" / "20251115_dataset_crp.csv"

    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path, nrows=1)

        # Exclude target columns and date/ticker
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
        feature_cols = [col for col in df.columns if col not in target_columns]
        n_features = len(feature_cols)

        print(f"  Using {n_features} features from dataset")
    else:
        # Default if data not available (must match model's expected features)
        n_features = 68
        print(f"  Using {n_features} features (default)")

    # Create random sequence
    # In production, replace this with real preprocessed data
    sequence = np.random.randn(LOOK_BACK, n_features).tolist()

    return sequence


def test_prediction():
    """Test the prediction endpoint."""
    print("\n" + "=" * 60)
    print("Testing Prediction Endpoint")
    print("=" * 60)

    # Create sample data
    sequence = create_sample_sequence()

    print(f"  Sequence shape: {len(sequence)} timesteps × {len(sequence[0])} features")

    # Make request
    payload = {"sequence": sequence}

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        print(f"\n✓ Prediction successful!")
        print(f"\n  Ensemble Prediction: {data['prediction']:.4f}")
        print(f"\n  Component Predictions:")
        for model, pred in data['components'].items():
            print(f"    {model:<12}: {pred:.4f}")

        return True
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"  Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_metrics_endpoint():
    """Test the Prometheus metrics endpoint."""
    print("\n" + "=" * 60)
    print("Testing Metrics Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{API_URL}/metrics")
        response.raise_for_status()

        # Parse Prometheus metrics (simple count)
        lines = response.text.split('\n')
        metric_count = len([l for l in lines if l and not l.startswith('#')])

        print(f"✓ Metrics endpoint accessible")
        print(f"  Found {metric_count} metrics")
        return True
    except Exception as e:
        print(f"✗ Metrics check failed: {e}")
        return False


def test_analytics_endpoints():
    """Test the analytics endpoints."""
    print("\n" + "=" * 60)
    print("Testing Analytics Endpoints")
    print("=" * 60)

    try:
        # Test recent predictions
        response = requests.get(f"{API_URL}/analytics/recent?limit=5")
        response.raise_for_status()
        data = response.json()

        print(f"✓ Recent predictions: {data.get('count', 0)} found")

        # Test performance metrics
        response = requests.get(f"{API_URL}/analytics/performance")
        response.raise_for_status()
        data = response.json()

        if 'error' in data:
            print(f"  Performance metrics: {data['error']}")
        else:
            print(f"✓ Performance metrics available")

        return True
    except Exception as e:
        print(f"✗ Analytics check failed: {e}")
        return False


def main():
    """Run all tests."""
    # Update URL from command line if provided
    global API_URL
    if len(sys.argv) > 1:
        API_URL = sys.argv[1].rstrip('/')

    print("\n" + "=" * 60)
    print("TIME SERIES API TEST SUITE")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Health Check", test_health_check()))
    results.append(("Prediction", test_prediction()))
    results.append(("Metrics", test_metrics_endpoint()))
    results.append(("Analytics", test_analytics_endpoints()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<20}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
