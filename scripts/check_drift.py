#!/usr/bin/env python3
"""
Drift Monitoring Script

This script checks for data drift and generates reports.
Can be run manually or scheduled via cron/GitHub Actions.
"""

import requests
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color


def check_drift_status(api_url: str):
    """Check drift detection status."""
    print(f"\n{YELLOW}Checking drift detection status...{NC}")

    try:
        response = requests.get(f"{api_url}/drift/status", timeout=10)
        response.raise_for_status()
        status = response.json()

        print(f"{GREEN}✓ Drift Detection Status:{NC}")
        print(f"  Reference samples: {status.get('reference_samples', 0)}")
        print(f"  Current window size: {status.get('current_window_size', 0)}")
        print(f"  Prediction window size: {status.get('prediction_window_size', 0)}")
        print(f"  Last drift check: {status.get('last_drift_check', 'N/A')}")
        print(f"  Drift threshold: {status.get('drift_threshold', 0.3)}")

        return status

    except requests.exceptions.RequestException as e:
        print(f"{RED}✗ Failed to get drift status: {e}{NC}")
        return None


def trigger_drift_check(api_url: str):
    """Trigger a drift detection check."""
    print(f"\n{YELLOW}Triggering drift detection check...{NC}")

    try:
        response = requests.post(f"{api_url}/drift/check", timeout=30)
        response.raise_for_status()
        results = response.json()

        status = results.get('status', 'unknown')

        if status == 'success':
            drift_detected = results.get('dataset_drift', False)
            drift_score = results.get('drift_score', 0.0)
            pred_drift_score = results.get('prediction_drift_score', 0.0)
            samples = results.get('samples_analyzed', 0)

            if drift_detected:
                print(f"{RED}⚠ DRIFT DETECTED!{NC}")
                print(f"  Dataset drift score: {drift_score:.3f}")
                print(f"  Prediction drift score: {pred_drift_score:.3f}")
                print(f"  Samples analyzed: {samples}")
                print(f"\n{YELLOW}Recommendation: Consider retraining the model.{NC}")
                return 1  # Return error code
            else:
                print(f"{GREEN}✓ No significant drift detected{NC}")
                print(f"  Dataset drift score: {drift_score:.3f}")
                print(f"  Prediction drift score: {pred_drift_score:.3f}")
                print(f"  Samples analyzed: {samples}")
                return 0

        elif status == 'insufficient_data':
            print(f"{YELLOW}⚠ Insufficient data for drift check{NC}")
            return 0

        else:
            print(f"{RED}✗ Drift check failed: {results.get('error', 'Unknown error')}{NC}")
            return 1

    except requests.exceptions.RequestException as e:
        print(f"{RED}✗ Failed to trigger drift check: {e}{NC}")
        return 1


def generate_drift_report(api_url: str, output_dir: Path):
    """Generate detailed drift report."""
    print(f"\n{YELLOW}Generating drift report...{NC}")

    try:
        response = requests.get(f"{api_url}/drift/report", timeout=60)
        response.raise_for_status()
        result = response.json()

        status = result.get('status', 'unknown')

        if status == 'success':
            report_path = result.get('report_path', 'N/A')
            print(f"{GREEN}✓ Drift report generated{NC}")
            print(f"  Report saved to: {report_path}")
            return 0
        elif status == 'insufficient_data':
            print(f"{YELLOW}⚠ Insufficient data for drift report{NC}")
            return 0
        else:
            print(f"{RED}✗ Failed to generate report: {result.get('message', 'Unknown error')}{NC}")
            return 1

    except requests.exceptions.RequestException as e:
        print(f"{RED}✗ Failed to generate drift report: {e}{NC}")
        return 1


def check_prometheus_metrics(prometheus_url: str):
    """Check drift metrics from Prometheus."""
    print(f"\n{YELLOW}Checking Prometheus drift metrics...{NC}")

    try:
        # Query dataset drift score
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={'query': 'model_drift_score{drift_type="dataset"}'},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        results = data.get('data', {}).get('result', [])
        if results:
            drift_score = float(results[0]['value'][1])
            print(f"{GREEN}✓ Dataset Drift Score:{NC} {drift_score:.3f}")

            if drift_score > 0.5:
                print(f"  {RED}⚠ Critical drift level!{NC}")
            elif drift_score > 0.3:
                print(f"  {YELLOW}⚠ Warning: Elevated drift{NC}")
            else:
                print(f"  {GREEN}✓ Drift within normal range{NC}")
        else:
            print(f"{YELLOW}  No drift metrics available yet{NC}")

        # Query prediction drift score
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={'query': 'prediction_drift_score'},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        results = data.get('data', {}).get('result', [])
        if results:
            pred_drift_score = float(results[0]['value'][1])
            print(f"{GREEN}✓ Prediction Drift Score:{NC} {pred_drift_score:.3f}")

        return 0

    except requests.exceptions.RequestException as e:
        print(f"{YELLOW}⚠ Failed to get Prometheus metrics: {e}{NC}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Monitor model drift')
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--prometheus-url',
        default='http://localhost:9090',
        help='Prometheus URL (default: http://localhost:9090)'
    )
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip generating detailed HTML report'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'artifacts' / 'drift_detection',
        help='Output directory for reports'
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Drift Detection Monitoring")
    print("=" * 50)
    print(f"API URL: {args.api_url}")
    print(f"Prometheus URL: {args.prometheus_url}")
    print("=" * 50)

    # Check drift status
    status = check_drift_status(args.api_url)
    if status is None:
        sys.exit(1)

    # Trigger drift check
    drift_result = trigger_drift_check(args.api_url)

    # Check Prometheus metrics
    check_prometheus_metrics(args.prometheus_url)

    # Generate detailed report
    if not args.skip_report:
        generate_drift_report(args.api_url, args.output_dir)

    print(f"\n{'=' * 50}")
    print("Drift Monitoring Complete")
    print("=" * 50)

    # Exit with error code if drift was detected
    sys.exit(drift_result)


if __name__ == "__main__":
    main()
