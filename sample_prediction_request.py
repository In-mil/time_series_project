#!/usr/bin/env python3
"""
Sample prediction requests for presentation demo.
Creates realistic-looking prediction requests with proper shape.
"""

import numpy as np
import json
import requests

# Configuration
API_URL = "http://localhost:8000"
LOOK_BACK = 20
N_FEATURES = 68


def create_realistic_sequence():
    """
    Create a realistic-looking sequence for demo.
    Simulates normalized cryptocurrency features.
    """
    np.random.seed(42)  # For reproducibility in demo

    # Create trending pattern (simulates crypto price movement)
    base_trend = np.linspace(0, 0.5, LOOK_BACK)

    # Create features with some correlation
    sequence = []
    for t in range(LOOK_BACK):
        # Create feature vector
        features = []

        # Add trend-based features (first 10)
        for i in range(10):
            noise = np.random.randn() * 0.2
            features.append(base_trend[t] + noise)

        # Add random features (next 58)
        for i in range(58):
            features.append(np.random.randn() * 0.5)

        sequence.append(features)

    return sequence


def make_prediction(sequence, show_details=True):
    """Make a prediction and display results."""
    payload = {"sequence": sequence}

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        if show_details:
            print("=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            print(f"\nEnsemble Prediction: {result['prediction']:.4f}")
            print(f"\nIndividual Model Predictions:")
            for model, value in result['components'].items():
                print(f"  {model:<12}: {value:>8.4f}")
            print("\n" + "=" * 60)

        return result

    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def demo_1_single_prediction():
    """Demo 1: Single prediction with detailed output."""
    print("\n" + "=" * 60)
    print("DEMO 1: Single Prediction")
    print("=" * 60)

    sequence = create_realistic_sequence()
    print(f"Input shape: {len(sequence)} timesteps × {len(sequence[0])} features")

    result = make_prediction(sequence)
    return result


def demo_2_multiple_predictions():
    """Demo 2: Multiple predictions to show consistency."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multiple Predictions")
    print("=" * 60)

    ensemble_preds = []

    for i in range(5):
        # Create different sequences
        np.random.seed(42 + i)
        sequence = create_realistic_sequence()

        result = make_prediction(sequence, show_details=False)
        if result:
            ensemble_preds.append(result['prediction'])
            print(f"Prediction {i+1}: {result['prediction']:.4f}")

    print(f"\nAverage: {np.mean(ensemble_preds):.4f}")
    print(f"Std Dev: {np.std(ensemble_preds):.4f}")
    print("=" * 60)


def demo_3_export_curl_command():
    """Demo 3: Generate cURL command for manual testing."""
    print("\n" + "=" * 60)
    print("DEMO 3: cURL Command for Manual Testing")
    print("=" * 60)

    sequence = create_realistic_sequence()
    payload = {"sequence": sequence}

    # Save to file
    with open('sample_request.json', 'w') as f:
        json.dump(payload, f, indent=2)

    print("\nSaved sample request to: sample_request.json")
    print("\nTo make prediction using cURL:")
    print("-" * 60)
    print(f"curl -X POST {API_URL}/predict \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d @sample_request.json \\")
    print(f"  | python -m json.tool")
    print("-" * 60)

    # Also print compact version
    compact_sequence = [[round(val, 2) for val in timestep[:10]] + ["..."]
                       for timestep in sequence[:3]] + ["..."]

    print("\nCompact representation (first 3 timesteps, first 10 features):")
    print(json.dumps(compact_sequence, indent=2))
    print("=" * 60)


def demo_4_analytics():
    """Demo 4: Show recent predictions from database."""
    print("\n" + "=" * 60)
    print("DEMO 4: Recent Predictions from Database")
    print("=" * 60)

    try:
        response = requests.get(f"{API_URL}/analytics/recent?limit=5")
        response.raise_for_status()
        data = response.json()

        print(f"\nTotal predictions in database: {data['count']}")
        print(f"\nLast 5 predictions (Ensemble model):")
        print("-" * 80)
        print(f"{'Timestamp':<30} {'Ensemble':<12} {'ANN':<12} {'GRU':<12} {'LSTM':<12}")
        print("-" * 80)

        for pred in data['predictions'][:5]:
            timestamp = pred.get('timestamp', 'N/A')
            ensemble = pred.get('prediction_ensemble', 0)
            ann = pred.get('prediction_ann', 0)
            gru = pred.get('prediction_gru', 0)
            lstm = pred.get('prediction_lstm', 0)
            print(f"{timestamp:<30} {ensemble:<12.4f} {ann:<12.4f} {gru:<12.4f} {lstm:<12.4f}")

        print("=" * 60)

    except Exception as e:
        print(f"Error fetching analytics: {e}")


def demo_5_performance_metrics():
    """Demo 5: Show API performance metrics."""
    print("\n" + "=" * 60)
    print("DEMO 5: API Performance Metrics")
    print("=" * 60)

    try:
        response = requests.get(f"{API_URL}/analytics/performance")
        response.raise_for_status()
        data = response.json()

        if 'error' in data:
            print(f"\nNo performance data available yet: {data['error']}")
        elif isinstance(data, dict):
            print("\nPerformance Summary:")
            print("-" * 60)

            # Handle different response structures
            if 'total_predictions' in data:
                print(f"Total predictions: {data.get('total_predictions', 0)}")
                print(f"Avg ensemble:      {data.get('avg_ensemble', 0):.4f}")
                print(f"Avg latency:       {data.get('avg_latency_ms', 0):.2f} ms")
            else:
                # If it's a nested structure
                for key, value in data.items():
                    if isinstance(value, dict):
                        print(f"\n{key}:")
                        for metric, val in value.items():
                            print(f"  {metric}: {val}")
                    else:
                        print(f"{key}: {value}")
        else:
            print(f"\nUnexpected response format: {type(data)}")
            print(data)

        print("=" * 60)

    except Exception as e:
        print(f"Error fetching performance metrics: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION REQUESTS FOR PRESENTATION")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print("=" * 60)

    # Check if API is running
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        print("✓ API is running and healthy\n")
    except Exception as e:
        print(f"✗ API is not accessible: {e}")
        print("\nPlease start the API first:")
        print("  docker-compose -f docker-compose.monitoring.yml up -d")
        return

    # Run demos
    demo_1_single_prediction()
    demo_2_multiple_predictions()
    demo_3_export_curl_command()
    demo_4_analytics()
    demo_5_performance_metrics()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
