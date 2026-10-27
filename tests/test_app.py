"""
PhishGuard AI - Automated Test Suite
------------------------------------
Tests the Flask backend, REST API endpoints, validation logic,
input sanitation, model inference accuracy, and error handling.
"""

import os
import sys
import json
import pytest

# Add parent directory to path so app and features can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, predict_email, MODEL, VECTORIZER, SCALER


@pytest.fixture
def client():
    """Configures Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page_loads(client):
    """Verifies that GET / renders the main HTML dashboard successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"PhishGuard" in response.data
    assert b"Email Security Inspection" in response.data


def test_health_check_endpoint(client):
    """Verifies that GET /health returns 200 with healthy state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_empty_input_rejected(client):
    """Verifies that empty string email_text returns a 400 Bad Request."""
    response = client.post(
        "/predict",
        data=json.dumps({"email_text": ""}),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_whitespace_only_input_rejected(client):
    """Verifies that whitespace-only string returns a 400 Bad Request."""
    response = client.post(
        "/predict",
        data=json.dumps({"email_text": "   \n\t  "}),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_malformed_json_handled(client):
    """Verifies that malformed JSON payload returns a 400 Bad Request."""
    response = client.post(
        "/predict",
        data="this is not a valid json string {missing_brace",
        content_type="application/json"
    )
    assert response.status_code == 400


def test_missing_email_text_field(client):
    """Verifies that JSON without email_text key returns 400."""
    response = client.post(
        "/predict",
        data=json.dumps({"other_key": "some value"}),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_oversized_input_rejected(client):
    """Verifies that input exceeding 50,000 characters is safely rejected."""
    oversized = "a" * 50001
    response = client.post(
        "/predict",
        data=json.dumps({"email_text": oversized}),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "exceeds maximum" in data.get("error", "").lower()


def test_prediction_returns_required_fields(client):
    """Verifies that a valid email returns all required output fields."""
    sample = "Hello team, please see the notes from today's meeting."
    response = client.post(
        "/predict",
        data=json.dumps({"email_text": sample}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert "label" in data
    assert data["label"] in ["Phishing", "Safe"]
    assert "confidence" in data
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 100
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Moderate", "High", "Critical"]
    assert "red_flags" in data
    assert isinstance(data["red_flags"], list)
    assert "explanation" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


def test_phishing_demonstration_email(client):
    """Verifies that a classic phishing email is analyzed and flagged."""
    phishing_text = (
        "URGENT: Your bank account has been suspended due to suspicious activity. "
        "Click here immediately to verify your password and confirm credentials: "
        "http://192.168.1.1/update-login or bit.ly/secure-auth"
    )
    response = client.post(
        "/predict",
        data=json.dumps({"email_text": phishing_text}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["label"] == "Phishing"
    assert data["risk_level"] in ["High", "Critical"]
    assert len(data["red_flags"]) > 0


def test_safe_demonstration_email(client):
    """Verifies that a standard workplace communication email is analyzed."""
    safe_text = (
        "Hi John,\n\n"
        "Here is the monthly report for our department. Let us discuss the "
        "quarterly budget during tomorrow's scheduled team sync. Let me know "
        "if you need any revisions.\n\nBest regards,\nSarah"
    )
    response = client.post(
        "/predict",
        data=json.dumps({"email_text": safe_text}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["label"] == "Safe"
    assert data["risk_level"] in ["Low", "Moderate"]


def test_xss_injection_handled_safely(client):
    """Verifies that malicious HTML/script tags are processed safely."""
    xss_text = "<script>alert('xss')</script><img src=x onerror=alert(1)> Urgent click here!"
    response = client.post(
        "/predict",
        data=json.dumps({"email_text": xss_text}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    # The output should be valid JSON and not crash
    assert "label" in data


def test_metrics_api_endpoint(client):
    """Verifies that GET /api/metrics returns the model evaluation data."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert data["accuracy"] > 90.0


def test_confusion_matrix_image_served(client):
    """Verifies that the confusion matrix image is accessible."""
    response = client.get("/confusion_matrix.png")
    assert response.status_code == 200
    assert response.mimetype in ["image/png", "application/octet-stream"]
