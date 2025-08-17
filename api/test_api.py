import pytest
import json
import main as api
from fastapi.testclient import TestClient


# ------------ Fixtures ------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Default client with model and dataset loaded via lifespan using
    monkeypatch loaders. This will also redirect logs to a temporary
    directory to avoid file system issues.
    """
    # Redirects logs to a temporary directory BEFORE TestClient triggers
    # lifespan
    monkeypatch.setattr(api, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(
        api, "LOG_FILE", str(tmp_path / "predictions_logs.json")
    )

    # Return a test client with lifespan triggered
    with TestClient(api.app) as client:
        yield client


@pytest.fixture
def client_no_model(tmp_path, monkeypatch):
    """
    Client with model not loaded, simulating a failure scenario.
    This will also redirect logs to a temporary directory to avoid file system
    issues.
    """
    # Redirects logs to a temporary directory BEFORE TestClient triggers
    # lifespan
    monkeypatch.setattr(api, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(
        api, "LOG_FILE", str(tmp_path / "predictions_logs.json")
    )

    # The forces 'lifespan' to not load the model by raising an error whenever
    # joblib.load is called
    def _raise_missing(_):
        raise FileNotFoundError

    monkeypatch.setattr(api.joblib, "load", _raise_missing, raising=False)

    with TestClient(api.app) as client:
        yield client


# ------------ Healthy State Test Cases ------------
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}


# ------------ Predict Test Cases ------------
# Add test cases using the /predict endpoint
@pytest.mark.parametrize(
    "input_data, expected_sentiment",
    [
        ({"text": "I love this!", "true_label": "positive"}, "positive"),
        ({"text": "This is terrible.", "true_label": "negative"}, "negative"),
    ],
    ids=["positive_case", "negative_case"],  # IDs for better readability
)
def test_predict_valid_input(client, input_data, expected_sentiment):
    response = client.post("/predict", json=input_data)
    # Assert the api didn't return an error
    assert response.status_code == 200
    assert "sentiment" in response.json()
    # Assert the sentiment is correctly predicted
    result = response.json()
    assert result["sentiment"] == expected_sentiment


# ------------ Model Not Loaded Test Cases ------------
# Add test cases for model not loaded
def test_predict_model_not_loaded(client_no_model):
    # Simulate model not being loaded
    response = client_no_model.post(
        "/predict", json={"text": "I love this!", "true_label": "positive"}
    )
    assert response.status_code == 503  # Service Unavailable
    assert (
        response.json()["detail"]
        == "Model is not loaded. Cannot make predictions."
    )


# ------------ Invalid Input Test Cases ------------
# Add test cases for invalid input (malformed, empty, etc.)
@pytest.mark.parametrize(
    "input_data",
    [
        {"text": "", "true_label": "positive"},  # Empty text
        {"text": "This is a awesome!", "true_label": ""},  # Empty true label
        {"text": 123, "true_label": "negative"},  # text is not a string
        {"text": "ok", "true_label": 1},  # true_label is not a string
        {"data": "invalid"},  # Completely invalid structure
        {"text": None, "true_label": "negative"},  # None text
        {"text": "This has no true label"},  # Missing true label
        {},  # Missing both fields
    ],
)
def test_predict_invalid_input(client, input_data):
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Unprocessable Entity


# ------------ Predict Probability Test Cases ------------
def test_predict_proba_valid_input(client):
    # Positive case
    response = client.post(
        "/predict_proba",
        json={"text": "I love this!", "true_label": "positive"}
    )
    assert response.status_code == 200
    # Assert that the api returned the expected keys
    result_pos = response.json()
    assert set(result_pos.keys()) == {"sentiment", "probability"}
    # Assert that the sentiment is positive
    assert result_pos["sentiment"] == "positive"
    # Assert that the probability is a float between 0.5 and 1
    assert 0.0 <= result_pos["probability"] <= 1.0

    # Negative case
    response = client.post(
        "/predict_proba",
        json={"text": "This is terrible.", "true_label": "negative"}
    )
    assert response.status_code == 200
    result_neg = response.json()
    assert result_neg["sentiment"] == "negative"
    assert 0.0 <= result_neg["probability"] <= 1.0


# ------------ Predict Probability Test Cases ------------
def test_predict_appends_log_file(client, tmp_path, monkeypatch):
    # Ensure a clean file and point there
    log_path = tmp_path / "predictions_logs.json"
    monkeypatch.setattr(api, "LOG_FILE", str(log_path))

    # Make a prediction
    response = client.post(
        "/predict", json={"text": "I love this!", "true_label": "positive"}
    )
    assert response.status_code == 200

    # Verify one JSON line exists and has required fields
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    log_entry = json.loads(lines[0])
    assert set(log_entry) == {
        "timestamp",
        "request_text",
        "predicted_sentiment",
        "true_sentiment",
    }
    assert log_entry["request_text"] == "I love this!"
    assert log_entry["predicted_sentiment"] == "positive"
    assert log_entry["true_sentiment"] == "positive"
