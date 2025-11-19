#!/usr/bin/env python3
"""
Comprehensive MLflow Experiment Summary
Shows all model performances and rankings
"""

import mlflow
import pandas as pd
from pathlib import Path

# Set tracking URI
if Path("mlflow.db").exists():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
else:
    mlflow.set_tracking_uri("file:./mlruns")

def get_all_model_performance():
    """Get performance metrics from all model experiments."""
    print("\n" + "="*100)
    print(" "*35 + "MODEL PERFORMANCE SUMMARY")
    print("="*100)

    # Get model evaluation experiment (has all model comparisons)
    eval_runs = mlflow.search_runs(
        experiment_ids=["6"],  # model_evaluation
        order_by=["start_time DESC"],
        max_results=1
    )

    if not eval_runs.empty:
        print("\n📊 LATEST EVALUATION RUN")
        print("-" * 100)

        latest = eval_runs.iloc[0]
        print(f"Run ID: {latest['run_id']}")
        print(f"Time: {latest['start_time']}")

        # Extract model metrics
        models_data = []
        for col in latest.index:
            if col.startswith('metrics.') and 'MAE_original' in col:
                model_name = col.replace('metrics.', '').replace('_MAE_original', '')
                mae_col = f'metrics.{model_name}_MAE_original'
                mse_col = f'metrics.{model_name}_MSE_original'

                if mae_col in latest.index and mse_col in latest.index:
                    models_data.append({
                        'Model': model_name,
                        'MAE': latest[mae_col],
                        'MSE': latest[mse_col],
                        'RMSE': latest[mse_col] ** 0.5
                    })

        if models_data:
            df = pd.DataFrame(models_data).sort_values('MAE')

            print("\n" + "="*100)
            print(" "*40 + "MODELS RANKED BY MAE")
            print("="*100)
            print()

            # Format table
            print(f"{'Rank':<6} {'Model':<15} {'MAE':>12} {'MSE':>12} {'RMSE':>12}")
            print("-" * 60)

            for idx, row in df.iterrows():
                rank = df.index.get_loc(idx) + 1
                best_marker = " 🏆" if rank == 1 else ""
                print(f"{rank:<6} {row['Model']:<15} {row['MAE']:>12.4f} {row['MSE']:>12.4f} {row['RMSE']:>12.4f}{best_marker}")

            print()
            print("="*100)

            # Show improvement
            best = df.iloc[0]
            worst = df.iloc[-1]
            improvement = ((worst['MAE'] - best['MAE']) / worst['MAE']) * 100

            print(f"\n✨ Best Model: {best['Model']} (MAE: {best['MAE']:.4f})")
            print(f"📈 Improvement over worst: {improvement:.1f}%")

            return df

    print("\nNo evaluation runs found.")
    return None

def get_individual_experiment_stats():
    """Get stats from individual model experiments."""
    print("\n\n" + "="*100)
    print(" "*30 + "INDIVIDUAL EXPERIMENT STATISTICS")
    print("="*100)

    experiments = {
        "2": "ANN",
        "3": "GRU",
        "4": "LSTM",
        "5": "Transformer",
        "1": "Ensemble"
    }

    for exp_id, name in experiments.items():
        runs = mlflow.search_runs(
            experiment_ids=[exp_id],
            max_results=1000
        )

        if not runs.empty:
            print(f"\n{name} Experiment (ID: {exp_id})")
            print("-" * 60)
            print(f"  Total runs: {len(runs)}")
            print(f"  Latest run: {runs.iloc[0]['start_time']}")

            # Show metric columns available
            metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
            if metric_cols:
                print(f"  Tracked metrics: {len(metric_cols)}")

def show_quick_commands():
    """Show useful MLflow commands."""
    print("\n\n" + "="*100)
    print(" "*35 + "USEFUL COMMANDS")
    print("="*100)

    commands = [
        ("View all experiments", "mlflow experiments search"),
        ("List runs in experiment", "mlflow runs list --experiment-id <ID>"),
        ("Query with Python script", "python scripts/query_mlflow.py <experiment_id>"),
        ("This summary", "python scripts/mlflow_summary.py"),
        ("Start MLflow UI", "mlflow ui --port 5000"),
    ]

    print()
    for desc, cmd in commands:
        print(f"  {desc:.<40} {cmd}")

    print("\n" + "="*100)

if __name__ == "__main__":
    # Get model performance summary
    df = get_all_model_performance()

    # Get individual experiment stats
    get_individual_experiment_stats()

    # Show commands
    show_quick_commands()

    print()
