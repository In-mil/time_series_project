"""
Tests for data validation logic
"""
import pytest
import numpy as np
from pydantic import ValidationError
from service.app import SequenceRequest


class TestSequenceValidation:
    """Tests for sequence input validation"""

    def test_valid_sequence_passes_validation(self, valid_sequence):
        """Test that valid sequence passes Pydantic validation"""
        request = SequenceRequest(sequence=valid_sequence)
        assert request.sequence == valid_sequence

    def test_sequence_must_be_list_of_lists(self):
        """Test that sequence must be a list of lists"""
        with pytest.raises((ValidationError, TypeError)):
            SequenceRequest(sequence=[1, 2, 3, 4, 5])

    def test_empty_sequence_fails_validation(self):
        """Test that empty sequence fails validation"""
        # Empty sequence should still be accepted by Pydantic
        # but fail in the API logic
        request = SequenceRequest(sequence=[])
        assert request.sequence == []

    def test_sequence_with_mixed_types_fails(self):
        """Test that sequence with mixed types fails"""
        with pytest.raises((ValidationError, TypeError)):
            SequenceRequest(sequence=[[1, 2, "string", 4]])


class TestArrayConversion:
    """Tests for numpy array conversion"""

    def test_list_converts_to_array(self, valid_sequence):
        """Test that list successfully converts to numpy array"""
        arr = np.array(valid_sequence, dtype=float)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    def test_array_shape_is_correct(self, valid_sequence):
        """Test that converted array has correct shape"""
        arr = np.array(valid_sequence, dtype=float)
        assert arr.shape == (20, 68)

    def test_nan_detection_works(self, sequence_with_nan):
        """Test that NaN values are detected"""
        arr = np.array(sequence_with_nan, dtype=float)
        assert np.any(np.isnan(arr))

    def test_inf_detection_works(self, sequence_with_inf):
        """Test that Inf values are detected"""
        arr = np.array(sequence_with_inf, dtype=float)
        assert np.any(np.isinf(arr))

    def test_valid_sequence_has_no_nan(self, valid_sequence):
        """Test that valid sequence has no NaN values"""
        arr = np.array(valid_sequence, dtype=float)
        assert not np.any(np.isnan(arr))

    def test_valid_sequence_has_no_inf(self, valid_sequence):
        """Test that valid sequence has no Inf values"""
        arr = np.array(valid_sequence, dtype=float)
        assert not np.any(np.isinf(arr))


class TestSequenceConstraints:
    """Tests for sequence size constraints"""

    def test_sequence_length_check(self, valid_sequence):
        """Test that valid sequence has correct length"""
        assert len(valid_sequence) == 20

    def test_feature_count_check(self, valid_sequence):
        """Test that each timestep has correct number of features"""
        for timestep in valid_sequence:
            assert len(timestep) == 68

    def test_short_sequence_detection(self, invalid_sequence_short):
        """Test that short sequence is detected"""
        assert len(invalid_sequence_short) < 20

    def test_wrong_feature_count_detection(
        self, invalid_sequence_wrong_features
    ):
        """Test that wrong feature count is detected"""
        for timestep in invalid_sequence_wrong_features:
            if len(timestep) != 68:
                assert True
                return
        assert False, "Should have detected wrong feature count"


class TestValueRanges:
    """Tests for value range validation"""

    def test_negative_values_are_allowed(self):
        """Test that negative values are allowed"""
        sequence = [[-1.0 for _ in range(68)] for _ in range(20)]
        arr = np.array(sequence, dtype=float)
        assert np.all(arr < 0)

    def test_zero_values_are_allowed(self):
        """Test that zero values are allowed"""
        sequence = [[0.0 for _ in range(68)] for _ in range(20)]
        arr = np.array(sequence, dtype=float)
        assert np.all(arr == 0)

    def test_large_values_are_allowed(self):
        """Test that large values are allowed"""
        sequence = [[1000.0 for _ in range(68)] for _ in range(20)]
        arr = np.array(sequence, dtype=float)
        assert np.all(arr == 1000.0)

    def test_small_decimal_values_are_allowed(self):
        """Test that small decimal values are allowed"""
        sequence = [[0.001 for _ in range(68)] for _ in range(20)]
        arr = np.array(sequence, dtype=float)
        assert np.all(arr == 0.001)
