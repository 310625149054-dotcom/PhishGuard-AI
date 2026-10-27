"""
PhishGuard AI - Flask Web Application Backend
---------------------------------------------
Serves the cybersecurity web dashboard and exposes the /predict REST API
to classify email text as 'Phishing' or 'Safe' with confidence estimation,
extracted threat indicators (red flags), and actionable safety recommendations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
from scipy.sparse import hstack, csr_matrix

from features import (
    clean_email_text,
    extract_tabular_features,
    analyze_risk_signals
)

# Initialize Flask application
app = Flask(__name__, template_folder="templates", static_folder="static")

# Restrict maximum payload size to prevent denial of service (1MB)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

# Production-safe logging: log operational events only, NEVER user email contents
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("phishguard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")

# Global model state loaded once at startup
MODEL = None
VECTORIZER = None
SCALER = None
METRICS = None


def load_artifacts():
    """Loads trained model, vectorizer, scaler, and metrics once at startup."""
    global MODEL, VECTORIZER, SCALER, METRICS
    try:
        model_path = os.path.join(MODEL_DIR, "phishing_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "model.joblib")
            
        vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(scaler_path):
            MODEL = joblib.load(model_path)
            VECTORIZER = joblib.load(vectorizer_path)
            SCALER = joblib.load(scaler_path)
            logger.info("Successfully loaded machine learning model, vectorizer, and feature scaler.")
        else:
            logger.warning("Model artifacts not found in %s. Run 'python phishing_detector.py' to train.", MODEL_DIR)

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                METRICS = json.load(f)
            logger.info("Loaded model evaluation metrics.")
        else:
            logger.warning("Metrics file not found at %s.", METRICS_PATH)
            
    except Exception as e:
        logger.error("Error loading model artifacts: %s", str(e))


# Load artifacts immediately
load_artifacts()


def predict_email(raw_text: str) -> Dict[str, Any]:
    """
    Executes identical preprocessing, vectorization, and inference
    as the training pipeline.
    
    Returns:
        Dictionary with label, confidence, risk_level, red_flags, explanation, recommendations
    """
    if MODEL is None or VECTORIZER is None or SCALER is None:
        raise RuntimeError("Machine learning model is not loaded. Please train the model using 'python phishing_detector.py'.")

    # 1. Preprocess and clean text
    cleaned = clean_email_text(raw_text)
    if not cleaned or len(cleaned.strip()) < 3:
        cleaned = "empty email body"

    # 2. Extract textual TF-IDF features
    tfidf_vec = VECTORIZER.transform([cleaned])

    # 3. Extract tabular URL and keyword features
    tab_vec = extract_tabular_features(raw_text)
    tab_vec_scaled = SCALER.transform([tab_vec])

    # 4. Combine into unified feature matrix
    combined_features = hstack([tfidf_vec, csr_matrix(tab_vec_scaled)]).tocsr()

    # 5. Predict class and estimated probability
    pred = int(MODEL.predict(combined_features)[0])
    
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(combined_features)[0]
        # probabilities[1] = phishing probability, probabilities[0] = safe probability
        phishing_prob = float(probabilities[1])
        safe_prob = float(probabilities[0])
        confidence = round((phishing_prob if pred == 1 else safe_prob) * 100, 1)
    else:
        confidence = 90.0  # Fallback if probability estimation unavailable

    label = "Phishing" if pred == 1 else "Safe"

    # 6. Analyze rule-based threat signals and indicators
    red_flags, risk_level, explanation, recommendations = analyze_risk_signals(raw_text)

    # Defense-in-depth: if multiple critical threat vectors are verified (e.g. masked shortener/IP + credential theft),
    # elevate classification to prevent dangerous False Negatives
    if label == "Safe" and risk_level == "Critical":
        label = "Phishing"
        confidence = max(confidence, 88.5)
        explanation = "Elevated to Phishing: Critical deceptive signals detected (masked URL/IP with credential harvesting cues)."
    elif label == "Phishing" and risk_level == "Low":
        risk_level = "High"
        explanation = "The statistical language and vocabulary pattern closely matches verified phishing emails in our training corpus."
    elif label == "Safe" and risk_level == "High":
        explanation += " Caution: While general wording resembles legitimate mail, detected URLs or keywords carry suspicious risk signals."

    return {
        "label": label,
        "confidence": confidence,
        "risk_level": risk_level,
        "red_flags": red_flags,
        "explanation": explanation,
        "recommendations": recommendations,
        "metadata": {
            "confidence_disclaimer": "Confidence values are statistical estimates based on corpus features and should not be treated as absolute proof.",
            "char_count": len(raw_text)
        }
    }


@app.route("/", methods=["GET"])
def index():
    """Renders the main PhishGuard AI cybersecurity dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint.
    Accepts JSON: {"email_text": "..."}
    Returns JSON prediction payload.
    """
    # Verify Content-Type
    if not request.is_json and request.content_type != "application/json":
        # Handle standard form submission or missing JSON header
        data = request.get_json(silent=True) or request.form.to_dict()
    else:
        data = request.get_json(silent=True)

    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Invalid request. Expected JSON object with 'email_text' property.",
            "status": 400
        }), 400

    email_text = data.get("email_text")
    if email_text is None or not isinstance(email_text, str) or not email_text.strip():
        return jsonify({
            "error": "Email text cannot be empty. Please provide email content to analyze.",
            "status": 400
        }), 400

    # Input length validation (protect against DOS or memory exhaustion)
    MAX_CHAR_LENGTH = 50000
    if len(email_text) > MAX_CHAR_LENGTH:
        return jsonify({
            "error": f"Email text exceeds maximum allowed length of {MAX_CHAR_LENGTH:,} characters.",
            "status": 400
        }), 400

    try:
        result = predict_email(email_text)
        return jsonify(result), 200
    except RuntimeError as re_err:
        logger.error("Runtime prediction error: %s", str(re_err))
        return jsonify({
            "error": str(re_err),
            "status": 503
        }), 503
    except Exception as e:
        logger.error("Internal server error during analysis: %s", str(e))
        return jsonify({
            "error": "An internal error occurred while processing the email. Please try again.",
            "status": 500
        }), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns evaluation metrics generated by phishing_detector.py."""
    if METRICS is not None:
        return jsonify(METRICS), 200
    
    # Try reading from disk if not cached
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data), 200
        except Exception as e:
            logger.error("Error reading metrics file: %s", str(e))

    return jsonify({
        "status": "not_trained",
        "message": "Evaluation metrics not found. Train the model using 'python phishing_detector.py'."
    }), 404


@app.route("/confusion_matrix.png", methods=["GET"])
def get_confusion_matrix_root():
    """Serves the generated confusion matrix image from the root directory."""
    if os.path.exists(CONFUSION_MATRIX_PATH):
        return send_from_directory(BASE_DIR, "confusion_matrix.png")
    
    static_cm = os.path.join(STATIC_DIR, "confusion_matrix.png")
    if os.path.exists(static_cm):
        return send_from_directory(STATIC_DIR, "confusion_matrix.png")
        
    return jsonify({"error": "Confusion matrix image not found. Please train the model."}), 404


@app.route("/health", methods=["GET"])
def health():
    """Healthcheck endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "metrics_available": METRICS is not None
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[*] Starting PhishGuard AI on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
