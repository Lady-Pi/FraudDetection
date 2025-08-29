#!/usr/bin/env python3
"""
Model training script for fraud detection MLOps pipeline.
Trains logistic regression model on monthly data and saves artifacts.

Usage:
    python -m src.models.train --through 03
"""

import argparse
import pandas as pd
import joblib
import json
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Import from utils
from .utils import (
    get_standard_features,
    load_monthly_data_range,
    get_project_paths,
    archive_current_model
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project paths
paths = get_project_paths()
CURRENT_DIR = paths['CURRENT_DIR']
METRICS_DIR = paths['METRICS_DIR']

# Ensure directories exist
CURRENT_DIR.mkdir(parents=True, exist_ok=True)
(METRICS_DIR / "artifacts").mkdir(parents=True, exist_ok=True)


def engineer_features(loans_df, transactions_df):
    """Create features for fraud detection model (matching notebook approach)"""

    # Start with loan application features
    features_df = loans_df.copy()

    # Transaction aggregations per customer
    trans_agg = transactions_df.groupby('customer_id').agg({
        'transaction_amount': ['count', 'sum', 'mean', 'std'],
        'fraud_flag': 'sum'
    }).round(3)

    # Flatten column names to match expected feature names
    trans_agg.columns = [f'trans_{col[0]}_{col[1]}' for col in trans_agg.columns]
    trans_agg = trans_agg.fillna(0)

    # Merge transaction features with loan data
    features_df = features_df.merge(trans_agg, on='customer_id', how='left')

    # Fill missing transaction data (customers with no transactions)
    trans_cols = [col for col in features_df.columns if col.startswith('trans_')]
    features_df[trans_cols] = features_df[trans_cols].fillna(0)

    return features_df


def preprocess_training_data(loans_df, transactions_df, feature_columns):
    """Preprocess training data to create feature matrix"""

    # Engineer features
    features_df = engineer_features(loans_df, transactions_df)

    # Ensure all required features exist
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0
            logger.warning(f"Missing feature {col}, filled with 0")

    # Handle missing values
    features_df[feature_columns] = features_df[feature_columns].fillna(0)

    # Create feature matrix and target
    X = features_df[feature_columns].copy()
    y = features_df['fraud_flag'].copy()

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution - Fraud: {y.sum()}, Non-fraud: {len(y) - y.sum()}")

    return X, y


def train_model(X, y):
    """Train logistic regression model with class balancing"""

    # Initialize and train model
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model.fit(X_scaled, y)

    # Calculate training metrics
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    train_accuracy = accuracy_score(y, y_pred)
    train_auroc = roc_auc_score(y, y_pred_proba)

    logger.info(f"Training metrics - Accuracy: {train_accuracy:.4f}, AUROC: {train_auroc:.4f}")

    return model, scaler, train_accuracy, train_auroc


def save_model_artifacts(model, scaler, feature_columns, metrics, month_range):
    """Save all model artifacts to current directory"""

    # Archive existing model first
    archive_current_model()

    # Save model and scaler
    joblib.dump(model, CURRENT_DIR / "model.pkl")
    joblib.dump(scaler, CURRENT_DIR / "scaler.pkl")

    # Create encoders placeholder (for API compatibility)
    encoders = {}
    joblib.dump(encoders, CURRENT_DIR / "encoders.pkl")

    # Create metadata
    metadata = {
        'model_type': 'LogisticRegression',
        'training_months': f"01-{month_range:02d}",
        'feature_count': len(feature_columns),
        'features': feature_columns,
        'training_date': datetime.now().isoformat(),
        'version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'class_weight': 'balanced',
        'random_state': 42
    }

    with open(CURRENT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save baseline metrics for drift detection
    baseline_metrics = {
        'accuracy': metrics['accuracy'],
        'auroc': metrics['auroc'],
        'training_months': f"01-{month_range:02d}",
        'created_date': datetime.now().isoformat()
    }

    with open(CURRENT_DIR / "baseline_metrics.json", 'w') as f:
        json.dump(baseline_metrics, f, indent=2)

    logger.info("Model artifacts saved successfully")
    return metadata


def log_to_mlflow(model, metrics, metadata):
    """Log training run to MLFlow"""

    try:
        mlflow.set_tracking_uri("file:./mlruns")

        # Create experiment if it doesn't exist
        experiment_name = "fraud-detection"
        try:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except:
            experiment_id = mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("training_months", metadata['training_months'])
            mlflow.log_param("feature_count", metadata['feature_count'])

            # Log metrics
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("auroc", metrics['auroc'])

            # Log model
            mlflow.sklearn.log_model(model, "model")

            logger.info("Training run logged to MLflow")

    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        # Continue without MLflow - training still succeeded


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--through', type=int, required=True,
                        help='Train through month (e.g., --through 03)')

    args = parser.parse_args()

    try:
        logger.info(f"Starting training through month {args.through:02d}")

        # Load training data using utils function
        loans_df, trans_df = load_monthly_data_range(1, args.through)

        if loans_df.empty:
            raise ValueError("No loan data loaded")

        # Get feature columns from utils
        feature_columns = get_standard_features()

        # Preprocess data
        X, y = preprocess_training_data(loans_df, trans_df, feature_columns)

        # Train model
        model, scaler, train_accuracy, train_auroc = train_model(X, y)

        # Prepare metrics
        metrics = {
            'accuracy': train_accuracy,
            'auroc': train_auroc
        }

        # Save artifacts
        metadata = save_model_artifacts(model, scaler, feature_columns, metrics, args.through)

        # Log to MLflow
        log_to_mlflow(model, metrics, metadata)

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {CURRENT_DIR}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()