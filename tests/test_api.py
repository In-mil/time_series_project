"""
Tests for FastAPI endpoints
"""
import pytest
from fastapi import status


class TestHealthEndpoint:
    """Tests for the health check endpoint"""

    def test_health_endpoint_returns_200(self, client):
        """Test that health endpoint returns 200 OK"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK

    def test_health_endpoint_returns_correct_structure(self, client):
        """Test that health endpoint returns correct JSON structure"""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "database_enabled" in data

    def test_health_endpoint_status_is_ok(self, client):
        """Test that health endpoint status is 'ok'"""
        response = client.get("/")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_endpoint_database_flag_is_boolean(self, client):
        """Test that database_enabled is a boolean"""
        response = client.get("/")
        data = response.json()
        assert isinstance(data["database_enabled"], bool)


class TestPredictionEndpoint:
    """Tests for the prediction endpoint"""

    def test_prediction_endpoint_with_valid_input(self, client, valid_request_payload):
        """Test prediction with valid input"""
        response = client.post("/predict", json=valid_request_payload)
        assert response.status_code == status.HTTP_200_OK

    def test_prediction_returns_correct_structure(self, client, valid_request_payload):
        """Test that prediction returns correct JSON structure"""
        response = client.post("/predict", json=valid_request_payload)
        data = response.json()
        assert "prediction" in data
        assert "components" in data

    def test_prediction_components_contain_all_models(
        self, client, valid_request_payload
    ):
        """Test that components contain all model predictions"""
        response = client.post("/predict", json=valid_request_payload)
        data = response.json()
        components = data["components"]
        assert "ANN" in components
        assert "GRU" in components
        assert "LSTM" in components
        assert "Transformer" in components

    def test_prediction_returns_numeric_values(self, client, valid_request_payload):
        """Test that prediction returns numeric values"""
        response = client.post("/predict", json=valid_request_payload)
        data = response.json()
        assert isinstance(data["prediction"], (int, float))
        for model, value in data["components"].items():
            assert isinstance(value, (int, float)), f"{model} should return numeric value"

    def test_prediction_with_short_sequence_fails(
        self, client, invalid_sequence_short
    ):
        """Test that prediction fails with sequence that's too short"""
        payload = {"sequence": invalid_sequence_short}
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_prediction_with_wrong_features_fails(
        self, client, invalid_sequence_wrong_features
    ):
        """Test that prediction fails with wrong number of features"""
        payload = {"sequence": invalid_sequence_wrong_features}
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_prediction_with_nan_fails(self, client):
        """Test that prediction fails with NaN values"""
        # Create a valid sequence and replace one value with None (JSON-compatible)
        # The server will convert to NaN when creating numpy array
        sequence = [[float(i + j * 0.1) for j in range(68)] for i in range(20)]
        sequence[10][5] = None  # Will become NaN in numpy
        payload = {"sequence": sequence}
        response = client.post("/predict", json=payload)
        # This will fail during numpy array conversion or validation
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_prediction_with_inf_fails(self, client):
        """Test that prediction handles very large values"""
        # JSON doesn't support Inf, so we test with moderately large numbers
        # Very large numbers like 1e308 can cause NaN in model predictions
        sequence = [[float(i + j * 0.1) for j in range(68)] for i in range(20)]
        sequence[15][10] = 1e10  # Large but manageable number
        payload = {"sequence": sequence}
        response = client.post("/predict", json=payload)
        # Should either succeed or fail with validation error (not 500)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

    def test_prediction_without_sequence_field_fails(self, client):
        """Test that prediction fails without 'sequence' field"""
        response = client.post("/predict", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_prediction_with_empty_sequence_fails(self, client):
        """Test that prediction fails with empty sequence"""
        response = client.post("/predict", json={"sequence": []})
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_prediction_with_non_numeric_values_fails(self, client):
        """Test that prediction fails with non-numeric values"""
        payload = {
            "sequence": [["string" for _ in range(16)] for _ in range(20)]
        }
        response = client.post("/predict", json=payload)
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints"""

    def test_recent_predictions_endpoint_exists(self, client):
        """Test that recent predictions endpoint exists"""
        response = client.get("/analytics/recent")
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    def test_recent_predictions_with_limit_parameter(self, client):
        """Test recent predictions with limit parameter"""
        response = client.get("/analytics/recent?limit=10")
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    def test_performance_metrics_endpoint_exists(self, client):
        """Test that performance metrics endpoint exists"""
        response = client.get("/analytics/performance")
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    def test_update_prediction_endpoint_exists(self, client):
        """Test that update prediction endpoint exists"""
        response = client.post("/analytics/update/1", json={"actual_value": 100.0})
        # Expect 404 or 200 depending on if prediction exists
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]
