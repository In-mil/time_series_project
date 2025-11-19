#!/usr/bin/env python3
"""
Query MLflow experiments and runs with sorting by metrics.
"""

import mlflow
import pandas as pd
from pathlib import Path

# Set tracking URI (use database if available, otherwise mlruns)
if Path("mlflow.db").exists():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
else:
    mlflow.set_tracking_uri("file:./mlruns")

def list_experiments():
    """List all experiments."""
    print("\n" + "="*80)
    print("MLFLOW EXPERIMENTS")
    print("="*80)

    experiments = mlflow.search_experiments()

    for exp in experiments:
        print(f"\nExperiment ID: {exp.experiment_id}")
        print(f"Name: {exp.name}")
        print(f"Artifact Location: {exp.artifact_location}")
        print(f"Lifecycle Stage: {exp.lifecycle_stage}")

def list_runs(experiment_id="0", order_by="metrics.test_mae ASC", max_results=10):
    """List runs with sorting."""
    print("\n" + "="*80)
    print(f"TOP {max_results} RUNS - Sorted by {order_by}")
    print("="*80 + "\n")

    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[order_by],
            max_results=max_results
        )

        if runs.empty:
            print("No runs found in this experiment.")
            return

        # Select key columns
        display_cols = [
            'run_id',
            'start_time',
            'tags.mlflow.runName',
            'params.model_type',
            'metrics.test_mae',
            'metrics.test_rmse',
            'metrics.test_r2'
        ]

        # Filter to only existing columns
        existing_cols = [col for col in display_cols if col in runs.columns]

        # Display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)

        print(runs[existing_cols].to_string(index=False))

        print(f"\n\nTotal runs: {len(runs)}")

    except Exception as e:
        print(f"Error searching runs: {e}")

def get_best_run(experiment_id="0", metric="test_mae"):
    """Get the best run by a specific metric."""
    print("\n" + "="*80)
    print(f"BEST RUN by {metric}")
    print("="*80 + "\n")

    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )

        if runs.empty:
            print("No runs found.")
            return

        best_run = runs.iloc[0]

        print(f"Run ID: {best_run['run_id']}")
        if 'tags.mlflow.runName' in best_run:
            print(f"Run Name: {best_run['tags.mlflow.runName']}")
        if 'params.model_type' in best_run:
            print(f"Model Type: {best_run['params.model_type']}")
        print(f"\nMetrics:")
        for col in best_run.index:
            if col.startswith('metrics.'):
                metric_name = col.replace('metrics.', '')
                print(f"  {metric_name}: {best_run[col]:.6f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys

    # List experiments
    list_experiments()

    # Get experiment ID from args or use default
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "0"

    # List all runs sorted by test_mae
    list_runs(experiment_id=exp_id, order_by="metrics.test_mae ASC", max_results=10)

    # Show best run
    get_best_run(experiment_id=exp_id, metric="test_mae")
