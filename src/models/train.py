#!/usr/bin/env python3
"""
Model training script for fraud detection MLOps pipeline.
Usage: python -m src.models.train --through 04
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "monthly"
MODELS_DIR = PROJECT_ROOT / "models"
CURRENT_DIR = MODELS_DIR / "current"
ARCHIVE_DIR = MODELS_DIR / "archive"
METRICS_DIR = PROJECT_ROOT / "metrics"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def setup_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("fraud_detection_training")


def load_monthly_data(through_month):
    """Load and combine data from months 01 through specified month"""
    all_loans = []
    all_transactions = []

    for month in range(1, through_month + 1):
        month_str = f"{month:02d}"
        month_dir = DATA_DIR / month_str

        # Load monthly data
        loans_file = month_dir / "loan_applications.csv"
        trans_file = month_dir / "transactions.csv"

        if loans_file.exists() and trans_file.exists():
            loans = pd.read_csv(loans_file)
            transactions = pd.read_csv(trans_file)

            # Add month identifier
            loans['data_month'] = month
            transactions['data_month'] = month

            all_loans.append(loans)
            all_transactions.append(transactions)
            logger.info(f"Loaded month {month_str}: {len(loans)} loans, {len(transactions)} transactions")
        else:
            logger.warning(f"Data files not found for month {month_str}")

    if not all_loans:
        raise ValueError(f"No data found for months 1-{through_month}")

    # Combine all months
    loans_df = pd.concat(all_loans, ignore_index=True)
    transactions_df = pd.concat(all_transactions, ignore_index=True)

    logger.info(f"Total training data: {len(loans_df)} loans, {len(transactions_df)} transactions")
    return loans_df, transactions_df


def preprocess_data(loans_df, transactions_df):
    """Preprocess data for training"""

    # Focus on loan applications for fraud detection
    df = loans_df.copy()

    # Select top 5 features based on correlation analysis with fraud_flag
    feature_columns = [
        'debt_to_income_ratio',  # 0.005644 - highest correlation
        'applicant_age',  # 0.004846
        'cibil_score',  # 0.001095
        'loan_tenure_months',  # -0.000484
        'existing_emis_monthly'  # -0.000631
    ]

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"Missing column {col}, setting to 0")
            df[col] = 0

    # Handle missing values
    df[feature_columns] = df[feature_columns].fillna(0)

    # Create feature matrix
    X = df[feature_columns].copy()
    y = df['fraud_flag'].copy()

    logger.info(f"Features: {feature_columns}")
    logger.info(f"Data shape: {X.shape}, Fraud rate: {y.mean():.3f}")

    return X, y, feature_columns


def train_model(X, y, feature_columns):
    """Train logistic regression model"""

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)

    # Validate model
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"Training accuracy: {accuracy:.4f}")
    logger.info(f"Training AUROC: {auroc:.4f}")

    return model, scaler, accuracy, auroc


def save_model_artifacts(model, scaler, feature_columns, accuracy, auroc, through_month):
    """Save model artifacts to models/current/"""

    # Create directories
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Save model artifacts
    joblib.dump(model, CURRENT_DIR / "model.pkl")
    joblib.dump(scaler, CURRENT_DIR / "scaler.pkl")

    # Create dummy encoders dict (for API compatibility)
    encoders = {}
    joblib.dump(encoders, CURRENT_DIR / "encoders.pkl")

    # Save metadata
    metadata = {
        "version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "trained_through_month": through_month,
        "training_date": datetime.now().isoformat(),
        "features": feature_columns,
        "model_type": "LogisticRegression",
        "accuracy": accuracy,
        "auroc": auroc,
        "feature_count": len(feature_columns)
    }

    with open(CURRENT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model artifacts saved to {CURRENT_DIR}")
    return metadata


def log_to_mlflow(metadata, model, accuracy, auroc):
    """Log training run to MLflow"""

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", metadata["model_type"])
        mlflow.log_param("trained_through_month", metadata["trained_through_month"])
        mlflow.log_param("feature_count", metadata["feature_count"])

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auroc", auroc)

        # Log model
        mlflow.sklearn.log_model(model, "fraud_detection_model")

        # Log metadata as artifact
        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact("model_metadata.json")

        logger.info("Training run logged to MLflow")


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--through', type=int, required=True,
                        help='Train through this month (1-12)')

    args = parser.parse_args()

    if args.through < 1 or args.through > 12:
        raise ValueError("Month must be between 1 and 12")

    logger.info(f"Starting training through month {args.through:02d}")

    # Setup
    setup_mlflow()

    # Load data
    loans_df, transactions_df = load_monthly_data(args.through)

    # Preprocess
    X, y, feature_columns = preprocess_data(loans_df, transactions_df)

    # Train model
    model, scaler, accuracy, auroc = train_model(X, y, feature_columns)

    # Save artifacts
    metadata = save_model_artifacts(model, scaler, feature_columns, accuracy, auroc, args.through)

    # Log to MLflow
    log_to_mlflow(metadata, model, accuracy, auroc)

    logger.info("Training completed successfully!")
    print(f"Model trained through month {args.through:02d}")
    print(f"Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}")


if __name__ == "__main__":
    main()