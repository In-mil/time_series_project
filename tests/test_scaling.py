"""
Tests for data scaling and transformation logic
"""
import pytest
import numpy as np
import joblib
from pathlib import Path


class TestScalerLoading:
    """Tests for scaler loading"""

    def test_scaler_X_exists(self):
        """Test that scaler_X file exists"""
        scaler_path = Path("artifacts/ensemble/scaler_X.pkl")
        assert scaler_path.exists(), "scaler_X.pkl not found"

    def test_scaler_y_exists(self):
        """Test that scaler_y file exists"""
        scaler_path = Path("artifacts/ensemble/scaler_y.pkl")
        assert scaler_path.exists(), "scaler_y.pkl not found"

    def test_scaler_X_loads_successfully(self):
        """Test that scaler_X loads without errors"""
        scaler_path = Path("artifacts/ensemble/scaler_X.pkl")
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            assert scaler is not None
            assert hasattr(scaler, "transform")

    def test_scaler_y_loads_successfully(self):
        """Test that scaler_y loads without errors"""
        scaler_path = Path("artifacts/ensemble/scaler_y.pkl")
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            assert scaler is not None
            assert hasattr(scaler, "transform")
            assert hasattr(scaler, "inverse_transform")


class TestFeatureScaling:
    """Tests for feature scaling behavior"""

    def test_scaling_transforms_data(self, mock_scaler_X, valid_sequence):
        """Test that scaling transforms data"""
        arr = np.array(valid_sequence, dtype=float)
        scaled = mock_scaler_X.transform(arr)
        assert scaled.shape == arr.shape
        assert not np.array_equal(scaled, arr)  # Should be different

    def test_scaling_preserves_shape(self, mock_scaler_X, valid_sequence):
        """Test that scaling preserves array shape"""
        arr = np.array(valid_sequence, dtype=float)
        scaled = mock_scaler_X.transform(arr)
        assert scaled.shape == (20, 68)

    def test_scaled_values_are_numeric(self, mock_scaler_X, valid_sequence):
        """Test that scaled values are numeric"""
        arr = np.array(valid_sequence, dtype=float)
        scaled = mock_scaler_X.transform(arr)
        assert not np.any(np.isnan(scaled))
        assert not np.any(np.isinf(scaled))

    def test_scaling_is_reversible(self, mock_scaler_X, valid_sequence):
        """Test that scaling can be reversed"""
        arr = np.array(valid_sequence, dtype=float)
        scaled = mock_scaler_X.transform(arr)
        unscaled = mock_scaler_X.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(arr, unscaled, decimal=5)


class TestTargetScaling:
    """Tests for target variable scaling"""

    def test_target_scaling_works(self, mock_scaler_y):
        """Test that target scaling works"""
        target = np.array([[100.0]])
        scaled = mock_scaler_y.transform(target)
        assert scaled.shape == target.shape
        assert not np.array_equal(scaled, target)

    def test_target_inverse_transform_works(self, mock_scaler_y):
        """Test that target inverse transform works"""
        target = np.array([[100.0]])
        scaled = mock_scaler_y.transform(target)
        unscaled = mock_scaler_y.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(target, unscaled, decimal=5)

    def test_multiple_predictions_scale_correctly(self, mock_scaler_y):
        """Test that multiple predictions scale correctly"""
        predictions = np.array([[10.0], [20.0], [30.0], [40.0]])
        scaled = mock_scaler_y.transform(predictions)
        unscaled = mock_scaler_y.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(predictions, unscaled, decimal=5)


class TestScalingEdgeCases:
    """Tests for edge cases in scaling"""

    def test_scaling_handles_zeros(self, mock_scaler_X):
        """Test that scaling handles zero values"""
        zeros = np.zeros((20, 16))
        scaled = mock_scaler_X.transform(zeros)
        assert not np.any(np.isnan(scaled))
        assert not np.any(np.isinf(scaled))

    def test_scaling_handles_negative_values(self, mock_scaler_X):
        """Test that scaling handles negative values"""
        negative = -np.ones((20, 16)) * 100
        scaled = mock_scaler_X.transform(negative)
        assert not np.any(np.isnan(scaled))
        assert not np.any(np.isinf(scaled))

    def test_scaling_handles_large_values(self, mock_scaler_X):
        """Test that scaling handles large values"""
        large = np.ones((20, 16)) * 1000
        scaled = mock_scaler_X.transform(large)
        assert not np.any(np.isnan(scaled))
        assert not np.any(np.isinf(scaled))

    def test_scaling_handles_small_values(self, mock_scaler_X):
        """Test that scaling handles small values"""
        small = np.ones((20, 16)) * 0.001
        scaled = mock_scaler_X.transform(small)
        assert not np.any(np.isnan(scaled))
        assert not np.any(np.isinf(scaled))


class TestScalerConsistency:
    """Tests for scaler consistency"""

    def test_scaler_produces_consistent_output(self, mock_scaler_X, valid_sequence):
        """Test that scaler produces consistent output for same input"""
        arr = np.array(valid_sequence, dtype=float)
        scaled1 = mock_scaler_X.transform(arr)
        scaled2 = mock_scaler_X.transform(arr)
        np.testing.assert_array_equal(scaled1, scaled2)

    def test_scaler_handles_different_inputs(self, mock_scaler_X):
        """Test that scaler handles different inputs differently"""
        arr1 = np.ones((20, 16))
        arr2 = np.ones((20, 16)) * 2
        scaled1 = mock_scaler_X.transform(arr1)
        scaled2 = mock_scaler_X.transform(arr2)
        assert not np.array_equal(scaled1, scaled2)
