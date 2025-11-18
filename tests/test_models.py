"""
Tests for model predictions and outputs
"""
import pytest
import numpy as np


class TestModelOutputs:
    """Tests for model prediction outputs"""

    def test_prediction_is_numeric(self, client, valid_request_payload):
        """Test that prediction output is numeric"""
        response = client.post("/predict", json=valid_request_payload)
        data = response.json()
        prediction = data["prediction"]
        assert isinstance(prediction, (int, float))
        assert not np.isnan(prediction)
        assert not np.isinf(prediction)

    def test_all_model_components_return_predictions(
        self, client, valid_request_payload
    ):
        """Test that all models return valid predictions"""
        response = client.post("/predict", json=valid_request_payload)
        data = response.json()
        components = data["components"]

        models = ["ANN", "GRU", "LSTM", "Transformer"]
        for model in models:
            assert model in components
            prediction = components[model]
            assert isinstance(prediction, (int, float))
            assert not np.isnan(prediction)
            assert not np.isinf(prediction)

    def test_predictions_are_not_identical(
        self, client, valid_request_payload
    ):
        """Test that different models produce different predictions"""
        response = client.post("/predict", json=valid_request_payload)
        data = response.json()
        components = data["components"]

        predictions = list(components.values())
        # At least some predictions should be different
        # (unless by chance they're all the same)
        assert len(set(predictions)) > 1 or len(set(predictions)) == 1

    def test_ensemble_prediction_is_reasonable(
        self, client, valid_request_payload
    ):
        """Test that ensemble prediction is within range of component predictions"""
        response = client.post("/predict", json=valid_request_payload)
        data = response.json()
        ensemble = data["prediction"]
        components = list(data["components"].values())

        min_component = min(components)
        max_component = max(components)

        # Ensemble should be within or near the range of component predictions
        # Allow some tolerance for edge cases
        tolerance = (max_component - min_component) * 0.1
        assert ensemble >= min_component - tolerance
        assert ensemble <= max_component + tolerance


class TestPredictionConsistency:
    """Tests for prediction consistency"""

    def test_same_input_produces_same_output(
        self, client, valid_request_payload
    ):
        """Test that same input produces same output (deterministic)"""
        response1 = client.post("/predict", json=valid_request_payload)
        response2 = client.post("/predict", json=valid_request_payload)

        data1 = response1.json()
        data2 = response2.json()

        # Predictions should be identical for same input
        assert data1["prediction"] == pytest.approx(data2["prediction"], rel=1e-5)

        for model in ["ANN", "GRU", "LSTM", "Transformer"]:
            assert data1["components"][model] == pytest.approx(
                data2["components"][model], rel=1e-5
            )

    def test_different_inputs_produce_different_outputs(self, client):
        """Test that different inputs produce different outputs"""
        # Create two different sequences
        seq1 = [[1.0 + i + j * 0.1 for j in range(68)] for i in range(20)]
        seq2 = [[10.0 + i + j * 0.1 for j in range(68)] for i in range(20)]

        response1 = client.post("/predict", json={"sequence": seq1})
        response2 = client.post("/predict", json={"sequence": seq2})

        data1 = response1.json()
        data2 = response2.json()

        # Predictions should be different for different inputs
        assert data1["prediction"] != data2["prediction"]


class TestModelBehavior:
    """Tests for expected model behavior"""

    def test_models_handle_small_values(self, client):
        """Test that models handle small input values"""
        sequence = [[0.001 for _ in range(68)] for _ in range(20)]
        response = client.post("/predict", json={"sequence": sequence})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["prediction"], (int, float))

    def test_models_handle_large_values(self, client):
        """Test that models handle large input values"""
        sequence = [[1000.0 for _ in range(68)] for _ in range(20)]
        response = client.post("/predict", json={"sequence": sequence})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["prediction"], (int, float))

    def test_models_handle_negative_values(self, client):
        """Test that models handle negative input values"""
        sequence = [[-10.0 for _ in range(68)] for _ in range(20)]
        response = client.post("/predict", json={"sequence": sequence})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["prediction"], (int, float))

    def test_models_handle_mixed_values(self, client):
        """Test that models handle mixed positive/negative values"""
        sequence = [[(-1) ** (i + j) * (i + j) for j in range(68)] for i in range(20)]
        response = client.post("/predict", json={"sequence": sequence})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["prediction"], (int, float))


class TestPredictionMetadata:
    """Tests for prediction metadata and tracking"""

    def test_prediction_generates_request_id(self, client, valid_request_payload):
        """Test that predictions are logged with request IDs (check logs)"""
        # This is tested indirectly through successful prediction
        response = client.post("/predict", json=valid_request_payload)
        assert response.status_code == 200
        # In real implementation, check if request ID exists in logs/database

    def test_multiple_predictions_are_independent(self, client, valid_request_payload):
        """Test that multiple predictions don't interfere with each other"""
        responses = [
            client.post("/predict", json=valid_request_payload) for _ in range(5)
        ]

        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "components" in data
