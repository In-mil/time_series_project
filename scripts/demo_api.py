#!/usr/bin/env python3
"""
API Demo Script for Presentation

Calls the Prediction API and displays results in a nice format.

Usage:
    python scripts/demo_api.py
"""

import requests

# API Configuration
API_URL = "https://prediction-api-101264457040.europe-west3.run.app"


def check_health():
    """Check if API is running."""
    print("Checking API health...")
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        print("API is running!\n")
        return True
    else:
        print(f"API error: {response.status_code}")
        return False


def get_batch_predictions():
    """Get batch predictions for all cryptocurrencies."""
    print("Fetching predictions from Cloud Run API...")
    print(f"URL: {API_URL}/predict/batch\n")

    response = requests.get(f"{API_URL}/predict/batch")

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    return response.json()


def display_results(data):
    """Display predictions in a nice table format."""
    print("=" * 60)
    print("  CRYPTO PRICE PREDICTIONS (5-Day % Change)")
    print("=" * 60)
    print(f"  Date: {data['date']}")
    print(f"  Total Cryptocurrencies: {data['count']}")
    print("-" * 60)
    print(f"  {'#':<4} {'Ticker':<10} {'Predicted Change':<18} {''}")
    print("-" * 60)

    for i, pred in enumerate(data['predictions'], 1):
        ticker = pred['ticker'].upper()
        change = pred['predicted_5d_change']

        # Add emoji indicator
        if change > 2:
            indicator = "UP"
        elif change > 0:
            indicator = "up"
        elif change > -2:
            indicator = "down"
        else:
            indicator = "DOWN"

        print(f"  {i:<4} {ticker:<10} {change:+.2f}%{'':<12} {indicator}")

    print("-" * 60)

    # Summary statistics
    changes = [p['predicted_5d_change'] for p in data['predictions']]
    print(f"\n  Summary:")
    print(f"    Best:    {data['predictions'][0]['ticker'].upper()} ({changes[0]:+.2f}%)")
    print(f"    Worst:   {data['predictions'][-1]['ticker'].upper()} ({changes[-1]:+.2f}%)")
    print(f"    Average: {sum(changes)/len(changes):+.2f}%")
    print("=" * 60)


def main():
    print("\n")
    print("*" * 60)
    print("*  THE SHOWDOWNN - Crypto Price Prediction Demo  *")
    print("*" * 60)
    print("\n")

    # Health check
    if not check_health():
        return

    # Get predictions
    data = get_batch_predictions()
    if data:
        display_results(data)

    print("\nAPI Response received from Google Cloud Run!")
    print(f"Endpoint: {API_URL}/predict/batch")


if __name__ == "__main__":
    main()
