#!/usr/bin/env python3
"""
Prepare reference dataset for drift detection.

This script creates a representative sample from the training data
to serve as the baseline for drift detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "final_data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ensemble"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "drift_detection"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_reference_data(
    input_file: Path,
    output_file: Path,
    sample_size: int = 10000,
    random_state: int = 42
):
    """
    Create reference dataset from training data.

    Args:
        input_file: Path to full training dataset
        output_file: Path to save reference dataset
        sample_size: Number of samples to include
        random_state: Random seed for reproducibility
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Original dataset shape: {df.shape}")

    # Remove target columns (we only want features for drift detection)
    target_cols = [
        'higher_close_today_vs_future_5_close',
        'future_5_close_higher_than_today',
        'higher_close_today_vs_future_10_close',
        'future_10_close_higher_than_today',
        'lower_close_today_vs_future_5_close',
        'future_5_close_lower_than_today',
        'lower_close_today_vs_future_10_close',
        'future_10_close_lower_than_today',
    ]

    # Keep only feature columns and metadata
    feature_df = df.drop(columns=[col for col in target_cols if col in df.columns], errors='ignore')

    logger.info(f"Feature dataset shape: {feature_df.shape}")

    # Sample data
    if len(feature_df) > sample_size:
        reference_df = feature_df.sample(n=sample_size, random_state=random_state)
        logger.info(f"Sampled {sample_size} rows")
    else:
        reference_df = feature_df
        logger.info(f"Using all {len(feature_df)} rows (less than requested sample size)")

    # Save reference data
    reference_df.to_csv(output_file, index=False)
    logger.info(f"Reference data saved to {output_file}")

    # Print statistics
    logger.info("\n=== Reference Data Statistics ===")
    logger.info(f"Shape: {reference_df.shape}")
    logger.info(f"Date range: {reference_df['date'].min()} to {reference_df['date'].max()}")

    if 'ticker' in reference_df.columns:
        logger.info(f"Tickers: {reference_df['ticker'].unique().tolist()}")
        logger.info(f"Ticker distribution:\n{reference_df['ticker'].value_counts()}")

    # Check for missing values
    missing = reference_df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")

    return reference_df


if __name__ == "__main__":
    input_file = DATA_DIR / "20251115_dataset_crp.csv"
    output_file = OUTPUT_DIR / "reference_data.csv"

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        exit(1)

    reference_df = prepare_reference_data(
        input_file=input_file,
        output_file=output_file,
        sample_size=10000
    )

    logger.info("\n✓ Reference dataset created successfully!")
    logger.info(f"  Location: {output_file}")
    logger.info(f"  Samples: {len(reference_df)}")
    logger.info(f"  Features: {len(reference_df.columns)}")
