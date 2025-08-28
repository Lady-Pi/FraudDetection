#!/usr/bin/env python3
"""
Model evaluation script for fraud detection MLOps pipeline.
Usage: python -m src.models.evaluate --month 05 --output-json
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import mlflow
from pathlib import Path
from datetime import datetime
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
METRICS_DIR = PROJECT_ROOT / "metrics"
ARTIFACTS_DIR = METRICS_DIR / "artifacts"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def setup_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("fraud_detection_evaluation")


def load_model_artifacts():
    """Load trained model artifacts"""
    try:
        model = joblib.load(CURRENT_DIR / "model.pkl")
        scaler = joblib.load(CURRENT_DIR / "scaler.pkl")
        encoders = joblib.load(CURRENT_DIR / "encoders.pkl")

        with open(CURRENT_DIR / "metadata.json", "r") as f:
            metadata = json.load(f)

        feature_columns = metadata.get('features', [])

        logger.info(f"Model artifacts loaded: {metadata.get('version', 'unknown')}")
        return model, scaler, encoders, metadata, feature_columns

    except FileNotFoundError as e:
        logger.error(f"Model artifacts not found: {e}")
        raise ValueError("No trained model found. Please run training first.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_evaluation_data(month):
    """Load data for specified month including transactions"""
    month_str = f"{month:02d}"
    month_dir = DATA_DIR / month_str

    loans_file = month_dir / "loan_applications.csv"
    trans_file = month_dir / "transactions.csv"

    if not loans_file.exists():
        raise FileNotFoundError(f"Loan data not found for month {month_str}")

    if not trans_file.exists():
        raise FileNotFoundError(f"Transaction data not found for month {month_str}")

    loans_df = pd.read_csv(loans_file)
    transactions_df = pd.read_csv(trans_file)

    logger.info(
        f"Loaded evaluation data for month {month_str}: {len(loans_df)} loans, {len(transactions_df)} transactions")

    return loans_df, transactions_df


def engineer_features(loans_df, transactions_df):
    """Create features for fraud detection model (matching notebook 3)"""

    # Start with loan application features
    features_df = loans_df.copy()

    # Transaction aggregations per customer
    trans_agg = transactions_df.groupby('customer_id').agg({
        'transaction_amount': ['count', 'sum', 'mean', 'std', 'max'],
        'fraud_flag': 'sum'  # Number of fraudulent transactions
    }).round(3)

    # Flatten column names
    trans_agg.columns = [f'trans_{col[0]}_{col[1]}' if col[1] else f'trans_{col[0]}'
                         for col in trans_agg.columns]
    trans_agg = trans_agg.fillna(0)

    # Merge with loan data
    features_df = features_df.merge(trans_agg, on='customer_id', how='left')

    # Fill missing transaction data (customers with no transactions)
    trans_cols = [col for col in features_df.columns if col.startswith('trans_')]
    features_df[trans_cols] = features_df[trans_cols].fillna(0)

    return features_df


def preprocess_evaluation_data(loans_df, transactions_df, feature_columns):
    """Preprocess evaluation data to match training format (notebook 3 approach)"""

    # Engineer features using same logic as training
    features_df = engineer_features(loans_df, transactions_df)

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in features_df.columns:
            logger.warning(f"Missing feature {col}, setting to 0")
            features_df[col] = 0

    # Handle missing values
    features_df[feature_columns] = features_df[feature_columns].fillna(0)

    # Create feature matrix
    X = features_df[feature_columns].copy()
    y = features_df['fraud_flag'].copy()

    return X, y


def evaluate_model(model, scaler, X, y):
    """Evaluate model performance"""

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    auroc = roc_auc_score(y, y_pred_proba)

    # Generate classification report
    class_report = classification_report(y, y_pred, output_dict=True)

    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred)

    metrics = {
        'accuracy': accuracy,
        'auroc': auroc,
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'confusion_matrix': cm.tolist(),
        'fraud_rate': float(y.mean()),
        'sample_count': len(y)
    }

    logger.info(f"Evaluation metrics - Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}")

    return metrics


def get_baseline_performance():
    """Get baseline accuracy from stored metrics"""
    baseline_file = CURRENT_DIR / "baseline_metrics.json"

    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
        return float(baseline_data['accuracy'])
    else:
        # Fallback for backward compatibility
        logger.warning("No baseline metrics found, using default 0.8")
        return 0.8


def save_artifacts(metrics, month, metadata):
    """Save evaluation artifacts"""

    # Create artifacts directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix as simple text
    cm_file = ARTIFACTS_DIR / f"confusion_matrix_month_{month:02d}.txt"
    with open(cm_file, "w") as f:
        f.write(f"Confusion Matrix - Month {month:02d}\n")
        f.write("=" * 30 + "\n")
        f.write(f"True Negative:  {metrics['confusion_matrix'][0][0]}\n")
        f.write(f"False Positive: {metrics['confusion_matrix'][0][1]}\n")
        f.write(f"False Negative: {metrics['confusion_matrix'][1][0]}\n")
        f.write(f"True Positive:  {metrics['confusion_matrix'][1][1]}\n")
        f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"AUROC: {metrics['auroc']:.4f}\n")

    logger.info(f"Evaluation artifacts saved to {ARTIFACTS_DIR}")


def log_to_mlflow(metrics, month, metadata):
    """Log evaluation to MLflow"""

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("evaluation_month", month)
        mlflow.log_param("model_version", metadata.get('version', 'unknown'))
        mlflow.log_param("sample_count", metrics['sample_count'])

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("auroc", metrics['auroc'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("fraud_rate", metrics['fraud_rate'])

        logger.info("Evaluation logged to MLflow")


def main():
    parser = argparse.ArgumentParser(description='Evaluate fraud detection model')
    parser.add_argument('--month', type=int, required=True,
                        help='Month to evaluate (1-12)')
    parser.add_argument('--output-json', action='store_true',
                        help='Output results as JSON file')

    args = parser.parse_args()

    if args.month < 1 or args.month > 12:
        raise ValueError("Month must be between 1 and 12")

    logger.info(f"Starting evaluation for month {args.month:02d}")

    try:
        # Setup
        setup_mlflow()

        # Load model
        model, scaler, encoders, metadata, feature_columns = load_model_artifacts()

        # Load evaluation data
        loans_df, transactions_df = load_evaluation_data(args.month)

        # Preprocess data
        X, y = preprocess_evaluation_data(loans_df, transactions_df, feature_columns)

        # Evaluate model
        metrics = evaluate_model(model, scaler, X, y)

        # Get baseline for comparison
        baseline_accuracy = get_baseline_performance()

        # Add comparison metrics
        performance_drop = baseline_accuracy - metrics['accuracy']
        metrics['baseline_accuracy'] = baseline_accuracy
        metrics['performance_drop'] = performance_drop

        # Save artifacts
        save_artifacts(metrics, args.month, metadata)

        # Log to MLflow
        log_to_mlflow(metrics, args.month, metadata)

        # Output results
        if args.output_json:
            with open("evaluation_results.json", "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Results saved to evaluation_results.json")

        # Print summary
        print(f"Evaluation completed for month {args.month:02d}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"Performance drop: {performance_drop:.4f}")

        if performance_drop >= 0.05:
            print("WARNING: Significant performance drop detected - retraining recommended")
        else:
            print("Model performance acceptable")


    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # Output error results for GitHub Actions
        if args.output_json:
            # Try to get the baseline even if evaluation failed
            try:
                baseline_accuracy = get_baseline_performance()
            except:
                baseline_accuracy = 0.8

            error_result = {
                "accuracy": "ERROR",
                "auroc": "ERROR",
                "baseline_accuracy": baseline_accuracy,
                "performance_drop": "ERROR",
                "error": str(e)
            }
            with open("evaluation_results.json", "w") as f:
                json.dump(error_result, f, indent=2)
        # Don't re-raise the exception - let GitHub Actions continue
        print(f"Evaluation failed: {e}")
        return


if __name__ == "__main__":
    main()