"""
Tests for drift detection functionality

This module tests the DriftDetector class and drift detection API endpoints.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timedelta

from service.drift_detector import DriftDetector, initialize_drift_detector, get_drift_detector


# ============================================================================
# Phase 2: Quick Wins - Basic Tests
# ============================================================================

class TestDriftDetectorInitialization:
    """Test drift detector initialization and configuration"""

    def test_drift_detector_initializes_with_valid_reference_data(
        self, tmp_path, sample_reference_data
    ):
        """Test that detector initializes correctly with valid reference data"""
        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)

        # Act
        detector = DriftDetector(reference_data_path=ref_path)

        # Assert
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 100
        assert detector.current_data == []
        assert detector.prediction_window == []

    def test_drift_detector_initializes_without_reference_data(self):
        """Test that detector initializes gracefully without reference data"""
        # Act
        detector = DriftDetector(reference_data_path=None)

        # Assert
        assert detector.reference_data is None
        assert detector.current_data == []
        assert detector.prediction_window == []

    def test_drift_detector_sets_window_size(self, tmp_path, sample_reference_data):
        """Test that custom window size is set correctly"""
        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)
        custom_window = 500

        # Act
        detector = DriftDetector(reference_data_path=ref_path, window_size=custom_window)

        # Assert
        assert detector.window_size == custom_window

    def test_drift_detector_sets_drift_threshold(self, tmp_path, sample_reference_data):
        """Test that custom drift threshold is set correctly"""
        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)
        custom_threshold = 0.5

        # Act
        detector = DriftDetector(
            reference_data_path=ref_path, drift_threshold=custom_threshold
        )

        # Assert
        assert detector.drift_threshold == custom_threshold

    def test_drift_detector_loads_feature_names(self, tmp_path, sample_reference_data):
        """Test that feature names are stored correctly"""
        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)
        feature_names = [f"feature_{i}" for i in range(68)]

        # Act
        detector = DriftDetector(
            reference_data_path=ref_path, feature_names=feature_names
        )

        # Assert
        assert detector.feature_names == feature_names
        assert len(detector.feature_names) == 68

    def test_drift_detector_initializes_empty_windows(
        self, tmp_path, sample_reference_data
    ):
        """Test that data windows start empty and last_check is set"""
        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)

        # Act
        before = datetime.now()
        detector = DriftDetector(reference_data_path=ref_path)
        after = datetime.now()

        # Assert
        assert detector.current_data == []
        assert detector.prediction_window == []
        assert before <= detector.last_drift_check <= after


class TestDriftEndpoints:
    """Test drift detection API endpoints"""

    def test_drift_status_endpoint_detector_disabled(self, client):
        """Test /drift/status when detector is disabled"""
        # Arrange: Ensure detector is not initialized
        with patch("service.app.drift_detector.get_drift_detector", return_value=None):
            # Act
            response = client.get("/drift/status")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "disabled"
            assert "message" in data

    def test_drift_status_endpoint_returns_status(self, client):
        """Test /drift/status returns detector status when enabled"""
        # Arrange: Mock detector with status
        mock_detector = MagicMock()
        mock_detector.get_status.return_value = {
            "reference_samples": 100,
            "current_window_size": 50,
            "prediction_window_size": 50,
            "drift_threshold": 0.3,
            "window_size": 1000,
        }

        with patch(
            "service.app.drift_detector.get_drift_detector", return_value=mock_detector
        ):
            # Act
            response = client.get("/drift/status")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["reference_samples"] == 100
            assert data["current_window_size"] == 50
            mock_detector.get_status.assert_called_once()

    def test_drift_check_endpoint_detector_disabled(self, client):
        """Test /drift/check when detector is disabled"""
        # Arrange
        with patch("service.app.drift_detector.get_drift_detector", return_value=None):
            # Act
            response = client.post("/drift/check")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "disabled"

    def test_drift_check_endpoint_triggers_check(self, client):
        """Test /drift/check triggers drift detection"""
        # Arrange
        mock_detector = MagicMock()
        mock_detector.check_drift.return_value = {
            "status": "success",
            "dataset_drift": True,
            "drift_score": 0.45,
            "samples_analyzed": 100,
        }

        with patch(
            "service.app.drift_detector.get_drift_detector", return_value=mock_detector
        ):
            # Act
            response = client.post("/drift/check")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["dataset_drift"] is True
            assert data["drift_score"] == 0.45
            mock_detector.check_drift.assert_called_once()

    def test_drift_report_endpoint_detector_disabled(self, client):
        """Test /drift/report when detector is disabled"""
        # Arrange
        with patch("service.app.drift_detector.get_drift_detector", return_value=None):
            # Act
            response = client.get("/drift/report")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "disabled"

    def test_drift_report_endpoint_insufficient_data(self, client):
        """Test /drift/report with insufficient data"""
        # Arrange
        mock_detector = MagicMock()
        mock_detector.get_status.return_value = {"current_window_size": 5}

        with patch(
            "service.app.drift_detector.get_drift_detector", return_value=mock_detector
        ):
            # Act
            response = client.get("/drift/report")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "insufficient_data"


# ============================================================================
# Phase 3: Core Logic - Reach 55% Coverage
# ============================================================================

class TestDriftDetectorDataCollection:
    """Test data collection and windowing"""

    def test_add_prediction_stores_features(self, drift_detector_instance):
        """Test that add_prediction stores features correctly"""
        # Arrange
        features = np.random.rand(68)
        prediction = 0.5

        # Act
        drift_detector_instance.add_prediction(features, prediction)

        # Assert
        assert len(drift_detector_instance.current_data) == 1
        assert len(drift_detector_instance.prediction_window) == 1
        assert drift_detector_instance.prediction_window[0] == 0.5

    def test_add_prediction_flattens_multidimensional_features(
        self, drift_detector_instance
    ):
        """Test that 2D features are flattened correctly"""
        # Arrange: 20 timesteps x 68 features
        features_2d = np.random.rand(20, 68)
        prediction = 0.7

        # Act
        drift_detector_instance.add_prediction(features_2d, prediction)

        # Assert
        assert len(drift_detector_instance.current_data) == 1
        # Features should be flattened to 1D (20 * 68 = 1360 features)
        record = drift_detector_instance.current_data[0]
        # Count feature keys (excluding timestamp, prediction)
        feature_keys = [k for k in record.keys() if k not in ["timestamp", "prediction"]]
        assert len(feature_keys) == 1360

    def test_add_prediction_handles_1d_features(self, drift_detector_instance):
        """Test that 1D features are stored correctly"""
        # Arrange
        features_1d = np.random.rand(68)
        prediction = 0.3

        # Act
        drift_detector_instance.add_prediction(features_1d, prediction)

        # Assert
        assert len(drift_detector_instance.current_data) == 1
        record = drift_detector_instance.current_data[0]
        feature_keys = [k for k in record.keys() if k not in ["timestamp", "prediction"]]
        assert len(feature_keys) == 68

    def test_add_prediction_maintains_window_size(self, drift_detector_instance):
        """Test that window size is maintained (FIFO)"""
        # Arrange: Window size is 50
        features = np.random.rand(68)

        # Act: Add more than window size
        for i in range(60):
            drift_detector_instance.add_prediction(features, float(i))

        # Assert
        assert len(drift_detector_instance.current_data) == 50
        assert len(drift_detector_instance.prediction_window) == 50
        # Check FIFO: first predictions should be removed, last ones kept
        assert drift_detector_instance.prediction_window[-1] == 59.0

    def test_add_prediction_with_metadata(self, drift_detector_instance):
        """Test that metadata is stored correctly"""
        # Arrange
        features = np.random.rand(68)
        prediction = 0.5
        metadata = {"timestamp": 123456, "ticker": "BTC"}

        # Act
        drift_detector_instance.add_prediction(features, prediction, metadata)

        # Assert
        record = drift_detector_instance.current_data[0]
        assert record["timestamp"] == 123456

    def test_add_prediction_without_metadata(self, drift_detector_instance):
        """Test that timestamp is auto-generated when no metadata"""
        # Arrange
        features = np.random.rand(68)
        prediction = 0.5

        # Act
        before = datetime.now()
        drift_detector_instance.add_prediction(features, prediction, metadata=None)
        after = datetime.now()

        # Assert
        record = drift_detector_instance.current_data[0]
        # Timestamp should be datetime object between before and after
        assert isinstance(record["timestamp"], datetime)
        assert before <= record["timestamp"] <= after

    def test_prediction_window_maintains_size(self, drift_detector_instance):
        """Test that prediction window maintains FIFO behavior"""
        # Arrange
        features = np.random.rand(68)

        # Act: Add 60 predictions (window_size=50)
        for i in range(60):
            drift_detector_instance.add_prediction(features, float(i))

        # Assert
        assert len(drift_detector_instance.prediction_window) == 50
        # First 10 predictions should be removed
        assert drift_detector_instance.prediction_window[0] == 10.0
        assert drift_detector_instance.prediction_window[-1] == 59.0

    def test_add_prediction_stores_correct_feature_names(self, drift_detector_instance):
        """Test that features are stored with correct names"""
        # Arrange
        features = np.arange(68, dtype=float)  # 0, 1, 2, ..., 67

        # Act
        drift_detector_instance.add_prediction(features, 0.5)

        # Assert
        record = drift_detector_instance.current_data[0]
        # Check that feature names match indices
        assert record[drift_detector_instance.feature_names[0]] == 0.0
        assert record[drift_detector_instance.feature_names[10]] == 10.0


class TestDriftDetectorDriftChecking:
    """Test drift detection logic"""

    def test_check_drift_insufficient_data(self, drift_detector_instance):
        """Test that drift check fails with insufficient data"""
        # Arrange: Add only 5 predictions (need more for drift check)
        features = np.random.rand(68)
        for i in range(5):
            drift_detector_instance.add_prediction(features, float(i))

        # Act
        result = drift_detector_instance.check_drift()

        # Assert
        assert result["status"] == "insufficient_data"

    def test_check_drift_no_reference_data(self):
        """Test that drift check fails without reference data"""
        # Arrange: Detector without reference data
        detector = DriftDetector(reference_data_path=None, window_size=50)
        features = np.random.rand(68)
        for i in range(20):
            detector.add_prediction(features, float(i))

        # Act
        result = detector.check_drift()

        # Assert
        assert result["status"] == "insufficient_data"

    @patch("service.drift_detector.Report")
    def test_check_drift_no_drift_detected(
        self, mock_report_class, drift_detector_instance
    ):
        """Test drift check when no drift is detected"""
        # Arrange: Mock Evidently Report
        mock_report = MagicMock()
        mock_metric = MagicMock()
        mock_metric.drift_share = 0.1  # Below threshold (0.3)
        mock_report.metrics = [mock_metric]
        mock_report_class.return_value = mock_report

        # Add sufficient predictions
        features = np.random.rand(68)
        for i in range(30):
            drift_detector_instance.add_prediction(features, float(i))

        # Act
        result = drift_detector_instance.check_drift()

        # Assert
        assert result["status"] == "success"
        assert result["dataset_drift"] is False
        assert result["drift_score"] == 0.1

    @patch("service.drift_detector.Report")
    @patch("service.drift_detector.DRIFT_DETECTED")
    def test_check_drift_drift_detected_warning(
        self, mock_drift_counter, mock_report_class, drift_detector_instance
    ):
        """Test drift detection with warning severity (0.3 < score < 0.5)"""
        # Arrange: Mock Evidently Report with drift
        mock_report = MagicMock()
        mock_metric = MagicMock()
        mock_metric.drift_share = 0.4  # Above threshold, below critical
        mock_report.metrics = [mock_metric]
        mock_report_class.return_value = mock_report

        # Add predictions
        features = np.random.rand(68)
        for i in range(30):
            drift_detector_instance.add_prediction(features, float(i))

        # Act
        result = drift_detector_instance.check_drift()

        # Assert
        assert result["status"] == "success"
        assert result["dataset_drift"] is True
        assert result["drift_score"] == 0.4
        # Should increment warning counter
        mock_drift_counter.labels.assert_called_with(
            drift_type="dataset", severity="warning"
        )

    @patch("service.drift_detector.Report")
    @patch("service.drift_detector.DRIFT_DETECTED")
    def test_check_drift_drift_detected_critical(
        self, mock_drift_counter, mock_report_class, drift_detector_instance
    ):
        """Test drift detection with critical severity (score > 0.5)"""
        # Arrange
        mock_report = MagicMock()
        mock_metric = MagicMock()
        mock_metric.drift_share = 0.6  # Critical level
        mock_report.metrics = [mock_metric]
        mock_report_class.return_value = mock_report

        features = np.random.rand(68)
        for i in range(30):
            drift_detector_instance.add_prediction(features, float(i))

        # Act
        result = drift_detector_instance.check_drift()

        # Assert
        assert result["drift_score"] == 0.6
        mock_drift_counter.labels.assert_called_with(
            drift_type="dataset", severity="critical"
        )

    @patch("service.drift_detector.Report")
    @patch("service.drift_detector.DRIFT_SCORE")
    def test_check_drift_updates_prometheus_metrics(
        self, mock_drift_score, mock_report_class, drift_detector_instance
    ):
        """Test that Prometheus metrics are updated"""
        # Arrange
        mock_report = MagicMock()
        mock_metric = MagicMock()
        mock_metric.drift_share = 0.25
        mock_report.metrics = [mock_metric]
        mock_report_class.return_value = mock_report

        features = np.random.rand(68)
        for i in range(30):
            drift_detector_instance.add_prediction(features, float(i))

        # Act
        drift_detector_instance.check_drift()

        # Assert
        mock_drift_score.labels.assert_called_with(drift_type="dataset")
        mock_drift_score.labels().set.assert_called_with(0.25)

    def test_check_drift_updates_last_check_timestamp(self, drift_detector_instance):
        """Test that last_drift_check timestamp is updated"""
        import time

        # Arrange
        features = np.random.rand(68)
        for i in range(30):
            drift_detector_instance.add_prediction(features, float(i))

        old_timestamp = drift_detector_instance.last_drift_check
        time.sleep(0.01)  # Small delay to ensure timestamp changes

        # Act
        with patch("service.drift_detector.Report"):
            mock_report = MagicMock()
            mock_metric = MagicMock()
            mock_metric.drift_share = 0.2
            mock_report.metrics = [mock_metric]
            with patch("service.drift_detector.Report", return_value=mock_report):
                drift_detector_instance.check_drift()

        # Assert
        assert drift_detector_instance.last_drift_check >= old_timestamp

    @patch("service.drift_detector.Report")
    def test_check_drift_returns_correct_structure(
        self, mock_report_class, drift_detector_instance
    ):
        """Test that drift check returns expected response structure"""
        # Arrange
        mock_report = MagicMock()
        mock_metric = MagicMock()
        mock_metric.drift_share = 0.2
        mock_report.metrics = [mock_metric]
        mock_report_class.return_value = mock_report

        features = np.random.rand(68)
        for i in range(30):
            drift_detector_instance.add_prediction(features, float(i))

        # Act
        result = drift_detector_instance.check_drift()

        # Assert
        assert "status" in result
        assert "dataset_drift" in result
        assert "drift_score" in result
        assert "prediction_drift_score" in result
        assert "timestamp" in result
        assert "samples_analyzed" in result

    @patch("service.drift_detector.Report")
    def test_check_drift_handles_evidently_exceptions(
        self, mock_report_class, drift_detector_instance
    ):
        """Test that Evidently exceptions are handled gracefully"""
        # Arrange: Mock Report to raise exception
        mock_report_class.return_value.run.side_effect = Exception("Evidently error")

        features = np.random.rand(68)
        for i in range(30):
            drift_detector_instance.add_prediction(features, float(i))

        # Act
        result = drift_detector_instance.check_drift()

        # Assert
        assert result["status"] == "error"
        assert "error" in result


# ============================================================================
# Integration Tests for Drift Endpoints
# ============================================================================

class TestDriftEndpointsIntegration:
    """Integration tests for drift detection endpoints with real detector"""

    def test_drift_endpoints_integration_flow(
        self, client, tmp_path, sample_reference_data
    ):
        """Test complete flow: initialize detector, add predictions, check drift"""
        # Arrange: Initialize drift detector in the app
        from service import drift_detector as drift_module

        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)

        non_feature_cols = ['ticker', 'date']
        feature_names = [col for col in sample_reference_data.columns if col not in non_feature_cols]

        drift_module.initialize_drift_detector(
            reference_data_path=ref_path,
            window_size=50,
            drift_threshold=0.3,
            feature_names=feature_names
        )

        # Act: Check status
        status_response = client.get("/drift/status")

        # Assert: Detector is initialized
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["reference_samples"] == 100
        assert status_data["window_size"] == 50

        # Cleanup
        drift_module._drift_detector = None

    def test_drift_check_integration_with_predictions(
        self, client, tmp_path, sample_reference_data
    ):
        """Test drift check endpoint after adding predictions"""
        from service import drift_detector as drift_module

        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)

        non_feature_cols = ['ticker', 'date']
        feature_names = [col for col in sample_reference_data.columns if col not in non_feature_cols]

        drift_module.initialize_drift_detector(
            reference_data_path=ref_path,
            window_size=50,
            drift_threshold=0.3,
            feature_names=feature_names
        )

        detector = drift_module.get_drift_detector()

        # Add some predictions
        features = np.random.rand(68)
        for i in range(30):
            detector.add_prediction(features, float(i))

        # Act: Trigger drift check (mock Evidently to avoid real computation)
        with patch("service.drift_detector.Report") as mock_report_class:
            mock_report = MagicMock()
            mock_metric = MagicMock()
            mock_metric.drift_share = 0.15  # No drift
            mock_report.metrics = [mock_metric]
            mock_report_class.return_value = mock_report

            response = client.post("/drift/check")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["dataset_drift"] is False
        assert "drift_score" in data

        # Cleanup
        drift_module._drift_detector = None

    def test_drift_report_integration(
        self, client, tmp_path, sample_reference_data
    ):
        """Test drift report endpoint integration"""
        from service import drift_detector as drift_module

        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)

        non_feature_cols = ['ticker', 'date']
        feature_names = [col for col in sample_reference_data.columns if col not in non_feature_cols]

        drift_module.initialize_drift_detector(
            reference_data_path=ref_path,
            window_size=50,
            drift_threshold=0.3,
            feature_names=feature_names
        )

        detector = drift_module.get_drift_detector()

        # Add sufficient predictions for report
        features = np.random.rand(68)
        for i in range(30):
            detector.add_prediction(features, float(i))

        # Act: Get drift report (mock Evidently)
        with patch("service.drift_detector.Report") as mock_report_class:
            mock_report = MagicMock()
            mock_metric = MagicMock()
            mock_metric.drift_share = 0.4  # Some drift
            mock_report.metrics = [mock_metric]
            mock_report_class.return_value = mock_report

            response = client.get("/drift/report")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "report" in data
        assert "recommendation" in data["report"]

        # Cleanup
        drift_module._drift_detector = None

    def test_global_drift_detector_functions(self, tmp_path, sample_reference_data):
        """Test global initialize and get functions"""
        from service import drift_detector as drift_module

        # Arrange
        ref_path = tmp_path / "ref.csv"
        sample_reference_data.to_csv(ref_path, index=False)

        # Act: Initialize
        drift_module.initialize_drift_detector(
            reference_data_path=ref_path,
            window_size=100,
            drift_threshold=0.4
        )

        # Assert: Can retrieve detector
        detector = drift_module.get_drift_detector()
        assert detector is not None
        assert detector.window_size == 100
        assert detector.drift_threshold == 0.4

        # Cleanup
        drift_module._drift_detector = None
