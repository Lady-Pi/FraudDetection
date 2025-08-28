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
    """Preprocess input data to match 14-feature notebook 3 approach"""
    try:
        # Convert to DataFrame if it's a dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # Create the 14-feature set from notebook 3
        model_features = pd.DataFrame()

        # Loan features (9 features)
        model_features['debt_to_income_ratio'] = df.get('debt_to_income_ratio',
                                                        df.get('existing_emis_monthly', 0) / df['monthly_income'])
        model_features['applicant_age'] = df['applicant_age']
        model_features['cibil_score'] = df['cibil_score']
        model_features['loan_amount_requested'] = df['loan_amount_requested']
        model_features['monthly_income'] = df['monthly_income']
        model_features['existing_emis_monthly'] = df.get('existing_emis_monthly', df['monthly_income'] * 0.15)
        model_features['interest_rate_offered'] = df.get('interest_rate_offered', 10.0)
        model_features['loan_tenure_months'] = df.get('loan_tenure_months', 24)
        model_features['number_of_dependents'] = df['number_of_dependents']

        # Transaction aggregation features (5 features)
        # Use defaults if transaction data not provided via API
        model_features['trans_transaction_amount_count'] = df.get('trans_transaction_amount_count', 30)
        model_features['trans_transaction_amount_sum'] = df.get('trans_transaction_amount_sum',
                                                                df['monthly_income'] * 0.8)
        model_features['trans_transaction_amount_mean'] = df.get('trans_transaction_amount_mean',
                                                                 model_features['trans_transaction_amount_sum'] /
                                                                 model_features['trans_transaction_amount_count'])
        model_features['trans_transaction_amount_std'] = df.get('trans_transaction_amount_std',
                                                                model_features['trans_transaction_amount_mean'] * 0.5)
        model_features['trans_fraud_flag_sum'] = df.get('trans_fraud_flag_sum', 0)

        # Ensure feature order matches training (14 features from notebook 3)
        feature_columns_ordered = [
            'debt_to_income_ratio',
            'applicant_age',
            'cibil_score',
            'loan_amount_requested',
            'monthly_income',
            'existing_emis_monthly',
            'interest_rate_offered',
            'loan_tenure_months',
            'number_of_dependents',
            'trans_transaction_amount_count',
            'trans_transaction_amount_sum',
            'trans_transaction_amount_mean',
            'trans_transaction_amount_std',
            'trans_fraud_flag_sum'
        ]

        # Select features in correct order
        df_final = model_features[feature_columns_ordered]

        # Handle any remaining missing values
        df_final = df_final.fillna(0)

        # Scale features using loaded scaler
        df_scaled = scaler.transform(df_final)

        logger.info(f"Preprocessed {len(df_final)} records with 14-feature notebook 3 approach")

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