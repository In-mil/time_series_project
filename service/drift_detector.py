"""
Drift Detection Module using Evidently AI

This module monitors:
1. Data drift: Changes in input feature distributions
2. Prediction drift: Changes in model output distributions
3. Concept drift: Changes in the relationship between features and target

Reference: https://github.com/evidentlyai/evidently
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging
from datetime import datetime, timedelta
import joblib
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# Prometheus metrics for drift detection
DRIFT_SCORE = Gauge(
    'model_drift_score',
    'Data drift score (0-1, higher = more drift)',
    ['drift_type']
)

DRIFT_DETECTED = Counter(
    'model_drift_detected_total',
    'Number of times drift was detected',
    ['drift_type', 'severity']
)

FEATURE_DRIFT = Gauge(
    'feature_drift_score',
    'Per-feature drift score',
    ['feature_name']
)

PREDICTION_DRIFT_SCORE = Gauge(
    'prediction_drift_score',
    'Drift score for model predictions'
)


class DriftDetector:
    """
    Monitors data and prediction drift using Evidently AI.

    Attributes:
        reference_data: Baseline data from training set
        current_window: Rolling window of recent predictions
        window_size: Number of samples to keep in current window
        drift_threshold: Threshold for drift detection (0-1)
    """

    def __init__(
        self,
        reference_data_path: Optional[Path] = None,
        window_size: int = 1000,
        drift_threshold: float = 0.3,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize drift detector.

        Args:
            reference_data_path: Path to reference dataset CSV
            window_size: Size of rolling window for current data
            drift_threshold: Threshold for drift alerts (0-1)
            feature_names: List of feature names (if None, will be inferred)
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.feature_names = feature_names or []

        # Storage for current window
        self.current_data: List[Dict] = []
        self.prediction_window: List[float] = []

        # Load reference data
        if reference_data_path and reference_data_path.exists():
            self.reference_data = self._load_reference_data(reference_data_path)
            logger.info(f"Loaded reference data: {len(self.reference_data)} samples")
        else:
            self.reference_data = None
            logger.warning("No reference data loaded - drift detection disabled")

        self.last_drift_check = datetime.now()
        self.drift_check_interval = timedelta(hours=1)  # Check drift every hour

    def _load_reference_data(self, path: Path) -> pd.DataFrame:
        """Load and prepare reference dataset."""
        try:
            df = pd.read_csv(path)
            logger.info(f"Reference data shape: {df.shape}")
            logger.info(f"Reference data columns: {list(df.columns)[:10]}...")
            return df
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            return None

    def add_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        metadata: Optional[Dict] = None
    ):
        """
        Add a new prediction to the current window.

        Args:
            features: Input features (flattened array or sequence)
            prediction: Model prediction value
            metadata: Optional metadata (timestamp, ticker, etc.)
        """
        # Flatten features if needed
        if len(features.shape) > 1:
            features_flat = features.flatten()
        else:
            features_flat = features

        # Create data record
        record = {
            'timestamp': metadata.get('timestamp', datetime.now()) if metadata else datetime.now(),
            'prediction': prediction,
        }

        # Add features
        for i, val in enumerate(features_flat):
            feature_name = f'feature_{i}' if i >= len(self.feature_names) else self.feature_names[i]
            record[feature_name] = val

        self.current_data.append(record)
        self.prediction_window.append(prediction)

        # Maintain window size
        if len(self.current_data) > self.window_size:
            self.current_data.pop(0)
        if len(self.prediction_window) > self.window_size:
            self.prediction_window.pop(0)

        # Check if it's time to run drift detection
        if self._should_check_drift():
            self.check_drift()

    def _should_check_drift(self) -> bool:
        """Determine if it's time to check for drift."""
        return (
            len(self.current_data) >= min(100, self.window_size // 2) and
            datetime.now() - self.last_drift_check > self.drift_check_interval
        )

    def check_drift(self) -> Dict:
        """
        Check for data and prediction drift.

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None or len(self.current_data) < 10:
            logger.warning("Insufficient data for drift detection")
            return {'status': 'insufficient_data'}

        try:
            # Convert current data to DataFrame
            current_df = pd.DataFrame(self.current_data)

            # Ensure we have common columns
            common_cols = list(set(self.reference_data.columns) & set(current_df.columns))
            if len(common_cols) < 5:
                logger.warning(f"Too few common columns: {len(common_cols)}")
                return {'status': 'schema_mismatch'}

            # Select subset for comparison
            ref_subset = self.reference_data[common_cols].sample(
                n=min(len(self.reference_data), self.window_size),
                random_state=42
            )
            curr_subset = current_df[common_cols]

            # Create Evidently report with new API
            report = Report(metrics=[
                DriftedColumnsCount(),
            ])

            report.run(
                reference_data=ref_subset,
                current_data=curr_subset
            )

            # Extract results - Evidently 0.7.x API
            # Access drift_share directly from the metric object
            metric = report.metrics[0]
            drift_score = metric.drift_share  # Share of drifted columns (0.0 to 1.0)

            # Update Prometheus metrics
            DRIFT_SCORE.labels(drift_type='dataset').set(drift_score)

            # Drift detected if share of drifted columns > threshold
            drift_detected = drift_score > self.drift_threshold
            if drift_detected:
                severity = 'critical' if drift_score > 0.5 else 'warning'
                DRIFT_DETECTED.labels(drift_type='dataset', severity=severity).inc()
                logger.warning(f"Data drift detected! Score: {drift_score:.3f}")
            else:
                logger.info(f"No significant drift detected. Score: {drift_score:.3f}")

            # Check prediction drift
            pred_drift_score = self._check_prediction_drift()

            self.last_drift_check = datetime.now()

            return {
                'status': 'success',
                'dataset_drift': drift_detected,
                'drift_score': drift_score,
                'prediction_drift_score': pred_drift_score,
                'timestamp': datetime.now().isoformat(),
                'samples_analyzed': len(curr_subset)
            }

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _check_prediction_drift(self) -> float:
        """Check if prediction distribution has drifted."""
        if len(self.prediction_window) < 50:
            return 0.0

        try:
            # Calculate statistics
            recent_preds = self.prediction_window[-100:]
            pred_mean = np.mean(recent_preds)
            pred_std = np.std(recent_preds)
            pred_min = np.min(recent_preds)
            pred_max = np.max(recent_preds)

            # Simple drift score based on variance
            # Higher variance might indicate drift
            if pred_std == 0:
                drift_score = 0.0
            else:
                # Normalize by range
                drift_score = min(1.0, pred_std / max(abs(pred_mean), 0.01))

            PREDICTION_DRIFT_SCORE.set(drift_score)

            logger.info(
                f"Prediction stats - Mean: {pred_mean:.3f}, "
                f"Std: {pred_std:.3f}, Range: [{pred_min:.3f}, {pred_max:.3f}]"
            )

            return drift_score

        except Exception as e:
            logger.error(f"Prediction drift check failed: {e}")
            return 0.0

    def get_drift_report(self) -> Optional[Report]:
        """
        Generate detailed drift report.

        Returns:
            Evidently Report object or None
        """
        if self.reference_data is None or len(self.current_data) < 10:
            return None

        try:
            current_df = pd.DataFrame(self.current_data)
            common_cols = list(set(self.reference_data.columns) & set(current_df.columns))

            if len(common_cols) < 5:
                return None

            ref_subset = self.reference_data[common_cols].sample(
                n=min(len(self.reference_data), 1000),
                random_state=42
            )
            curr_subset = current_df[common_cols]

            # Comprehensive report with multiple metrics
            report = Report(metrics=[
                DriftedColumnsCount(),
            ])

            report.run(
                reference_data=ref_subset,
                current_data=curr_subset
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate drift report: {e}")
            return None

    def get_status(self) -> Dict:
        """Get current drift detector status."""
        return {
            'reference_samples': len(self.reference_data) if self.reference_data is not None else 0,
            'current_window_size': len(self.current_data),
            'prediction_window_size': len(self.prediction_window),
            'last_drift_check': self.last_drift_check.isoformat(),
            'drift_threshold': self.drift_threshold,
            'window_size': self.window_size,
        }


# Global drift detector instance (initialized in app.py)
drift_detector: Optional[DriftDetector] = None


def initialize_drift_detector(
    reference_data_path: Optional[Path] = None,
    **kwargs
) -> DriftDetector:
    """Initialize global drift detector instance."""
    global drift_detector
    drift_detector = DriftDetector(reference_data_path=reference_data_path, **kwargs)
    logger.info("Drift detector initialized")
    return drift_detector


def get_drift_detector() -> Optional[DriftDetector]:
    """Get global drift detector instance."""
    return drift_detector
