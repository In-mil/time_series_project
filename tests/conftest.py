"""
Pytest fixtures and configuration for tests
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import service module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if models exist locally - use mocks only as fallback
REPO_ROOT = Path(__file__).parent.parent
MODELS_EXIST = (REPO_ROOT / "models" / "model_ann.keras").exists()
SCALERS_EXIST = (REPO_ROOT / "artifacts" / "ensemble" / "scaler_X.pkl").exists()

# Only use mocks if models don't exist (e.g., fresh clone without DVC pull)
USE_MOCKS = not (MODELS_EXIST and SCALERS_EXIST)

if USE_MOCKS:
    print("⚠️  WARNING: Models not found locally, using mocks for testing")
    print("   Run 'dvc pull' to use real models in tests")

    # Create persistent mocks for models and scalers
    import unittest.mock as mock_module

    # Mock Keras model that returns slightly different values based on input
    mock_keras_model = MagicMock()
    def mock_predict(x, verbose=0):
        # Return different values based on input sum for variety
        input_sum = float(np.sum(x))
        result = 0.5 + (input_sum % 1.0) * 0.1  # Varies between 0.5 and 0.6
        return np.array([[result]])
    mock_keras_model.predict.side_effect = mock_predict

    # Smart mock scaler that validates shape
    class MockScaler:
        def transform(self, X):
            X = np.array(X)
            # Validate shape like real scaler
            if X.shape[1] != 68:
                raise ValueError(f"X has {X.shape[1]} features, but StandardScaler is expecting 68 features as input.")
            # Return slightly modified data
            return X / 100.0

        def inverse_transform(self, X):
            X = np.array(X)
            # Return scaled back
            return X * 100.0

    mock_scaler_transform = MockScaler()

    # Apply patches globally
    _keras_patcher = patch("tensorflow.keras.models.load_model", return_value=mock_keras_model)
    _joblib_patcher = patch("joblib.load", return_value=mock_scaler_transform)

    _keras_patcher.start()
    _joblib_patcher.start()
else:
    print("✅ Using real models from DVC for testing")

# Now import the app (with real models or mocks)
from service.app import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def valid_sequence():
    """Generate a valid input sequence (20 timesteps x 68 features)"""
    return [[float(i + j * 0.1) for j in range(68)] for i in range(20)]


@pytest.fixture
def valid_request_payload(valid_sequence):
    """Generate a valid request payload"""
    return {"sequence": valid_sequence}


@pytest.fixture
def invalid_sequence_short():
    """Generate an invalid sequence (too short)"""
    return [[float(i + j * 0.1) for j in range(68)] for i in range(10)]


@pytest.fixture
def invalid_sequence_wrong_features():
    """Generate an invalid sequence (wrong number of features)"""
    return [[float(i + j * 0.1) for j in range(10)] for i in range(20)]


@pytest.fixture
def sequence_with_nan():
    """Generate a sequence containing NaN values"""
    seq = [[float(i + j * 0.1) for j in range(68)] for i in range(20)]
    seq[10][5] = float('nan')
    return seq


@pytest.fixture
def sequence_with_inf():
    """Generate a sequence containing Inf values"""
    seq = [[float(i + j * 0.1) for j in range(68)] for i in range(20)]
    seq[15][10] = float('inf')
    return seq


@pytest.fixture
def mock_scaler_X():
    """Mock scaler for features"""
    class MockScaler:
        def transform(self, X):
            return np.array(X) / 100.0

        def inverse_transform(self, X):
            return np.array(X) * 100.0

    return MockScaler()


@pytest.fixture
def mock_scaler_y():
    """Mock scaler for target"""
    class MockScaler:
        def transform(self, y):
            return np.array(y) / 10.0

        def inverse_transform(self, y):
            return np.array(y) * 10.0

    return MockScaler()
