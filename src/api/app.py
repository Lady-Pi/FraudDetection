# Flask API

import os
import joblib
import json
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
import logging
from pathlib import Path

from .auth import require_auth
from .schemas import PredictionRequest, PredictionResponse, validate_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define project paths (same as in notebooks)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Navigate from src/api/app.py to project root
MODELS_DIR = PROJECT_ROOT / "models"
CURRENT_DIR = MODELS_DIR / "current"

app = Flask(__name__)

# Global variables for model artifacts
model = None
scaler = None
encoders = None
metadata = None
feature_columns = None


def load_model_artifacts():
    """Load model artifacts from models/current/ directory"""
    global model, scaler, encoders, metadata, feature_columns

    try:
        models_dir = CURRENT_DIR

        # Load model
        model = joblib.load(models_dir / "model.pkl")

        # Load scaler
        scaler = joblib.load(models_dir / "scaler.pkl")

        # Load encoders
        encoders = joblib.load(models_dir / "encoders.pkl")

        # Load metadata
        with open(models_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        feature_columns = metadata.get('features', [])

        logger.info(f"Model artifacts loaded successfully")
        logger.info(f"Model version: {metadata.get('version', 'unknown')}")
        logger.info(f"Features: {len(feature_columns)} columns")

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        raise


def preprocess_features(data):
    """Preprocess input data to match training format with correlation-based features"""
    try:
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # Create the 5 correlation-based model features from API input
        model_features = pd.DataFrame()

        # debt_to_income_ratio (highest correlation: 0.005644)
        if 'debt_to_income_ratio' in df.columns:
            model_features['debt_to_income_ratio'] = df['debt_to_income_ratio']
        else:
            # Calculate from existing_emis_monthly / monthly_income
            existing_emis = df.get('existing_emis_monthly', 0)
            monthly_income = df['monthly_income']
            model_features['debt_to_income_ratio'] = existing_emis / monthly_income

        # applicant_age (0.004846)
        model_features['applicant_age'] = df['applicant_age']

        #  cibil_score (0.001095)
        model_features['cibil_score'] = df['cibil_score']

        # loan_tenure_months (-0.000484)
        if 'loan_tenure_months' in df.columns:
            model_features['loan_tenure_months'] = df['loan_tenure_months']
        else:
            # Use reasonable default based on loan amount
            loan_amount = df.get('loan_amount_requested', 50000)
            # Larger loans typically have longer tenure
            default_tenure = 12 + (loan_amount / 10000).clip(0, 48)  # 12-60 months
            model_features['loan_tenure_months'] = default_tenure.astype(int)

        # existing_emis_monthly (-0.000631)
        if 'existing_emis_monthly' in df.columns:
            model_features['existing_emis_monthly'] = df['existing_emis_monthly']
        else:
            # Calculate based on income (assume 10-30% of income for existing EMIs)
            monthly_income = df['monthly_income']
            # Conservative estimate: 15% of income
            model_features['existing_emis_monthly'] = monthly_income * 0.15

        # Ensure feature order matches training
        feature_columns_ordered = [
            'debt_to_income_ratio',
            'applicant_age',
            'cibil_score',
            'loan_tenure_months',
            'existing_emis_monthly'
        ]

        # Select features in correct order
        df_final = model_features[feature_columns_ordered]

        # Handle missing values
        df_final = df_final.fillna(0)

        # Scale features using loaded scaler
        df_scaled = scaler.transform(df_final)

        logger.info(f"Preprocessed {len(df_final)} records with correlation-based features")
        logger.debug(f"Feature means: {df_final.mean().to_dict()}")

        return df_scaled

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ValueError(f"Data preprocessing error: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 503

        return jsonify({
            "status": "ok",
            "model_version": metadata.get('version', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    """Fraud prediction endpoint"""
    start_time = datetime.now()

    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate input format
        validation_error = validate_request(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        # Apply defaults for missing optional fields
        from .schemas import apply_defaults_to_request
        data_with_defaults = apply_defaults_to_request(data)

        # Handle both single prediction and batch
        is_batch = isinstance(data_with_defaults, list)
        input_data = data_with_defaults if is_batch else [data_with_defaults]

        # Preprocess features
        X = preprocess_features(input_data)

        # Make predictions
        probabilities = model.predict_proba(X)[:, 1]  # Fraud probability
        predictions = (probabilities >= 0.5).astype(int)  # Binary predictions

        # Format response
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            results.append({
                "fraud_probability": float(prob),
                "prediction": int(pred),
                "confidence": "high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.1 else "low"
            })

        # Log request
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Prediction request: {len(input_data)} records, {processing_time:.3f}s")

        # Return single result or batch
        response = results if is_batch else results[0]
        return jsonify(response), 200

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": "Unauthorized - valid bearer token required"}), 401


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Load model artifacts on startup
    try:
        load_model_artifacts()
        logger.info("Flask app initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize app: {str(e)}")
        exit(1)

    # Run the app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )