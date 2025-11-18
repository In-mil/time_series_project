"""
Pytest fixtures and configuration for tests
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path so we can import service module
sys.path.insert(0, str(Path(__file__).parent.parent))

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
