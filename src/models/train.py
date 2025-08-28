#!/usr/bin/env python3
"""
Utility functions for fraud detection model pipeline.
Shared functions for data loading, preprocessing, and model operations.
"""

import pandas as pd
import numpy as np
import joblib
import json
import mlflow
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "monthly"
MODELS_DIR = PROJECT_ROOT / "models"
CURRENT_DIR = MODELS_DIR / "current"
ARCHIVE_DIR = MODELS_DIR / "archive"
METRICS_DIR = PROJECT_ROOT / "metrics"


def get_project_paths():
    """Return standardized project paths"""
    return {
        'PROJECT_ROOT': PROJECT_ROOT,
        'DATA_DIR': DATA_DIR,
        'MODELS_DIR': MODELS_DIR,
        'CURRENT_DIR': CURRENT_DIR,
        'METRICS_DIR': METRICS_DIR
    }


def load_monthly_data_range(start_month, end_month):
    """Load data for a range of months"""
    all_loans = []
    all_transactions = []

    for month in range(start_month, end_month + 1):
        month_str = f"{month:02d}"
        month_dir = DATA_DIR / month_str

        loans_file = month_dir / "loan_applications.csv"
        trans_file = month_dir / "transactions.csv"

        if loans_file.exists():
            loans = pd.read_csv(loans_file)
            loans['data_month'] = month
            all_loans.append(loans)

        if trans_file.exists():
            transactions = pd.read_csv(trans_file)
            transactions['data_month'] = month
            all_transactions.append(transactions)

    loans_df = pd.concat(all_loans, ignore_index=True) if all_loans else pd.DataFrame()
    trans_df = pd.concat(all_transactions, ignore_index=True) if all_transactions else pd.DataFrame()

    return loans_df, trans_df


def get_standard_features():
    """Return the 14-feature set from notebook 3 analysis"""
    return [
        # Top correlated loan features
        'debt_to_income_ratio',  # Highest correlation (0.0056)
        'applicant_age',  # Second highest (0.0048)
        'cibil_score',  # Credit score
        'loan_amount_requested',  # Loan amount
        'monthly_income',  # Income
        'existing_emis_monthly',  # Existing EMIs
        'interest_rate_offered',  # Interest rate
        'loan_tenure_months',  # Loan tenure
        'number_of_dependents',  # Dependents

        # Transaction aggregations (add predictive power)
        'trans_transaction_amount_count',  # Number of transactions
        'trans_transaction_amount_sum',  # Total transaction volume
        'trans_transaction_amount_mean',  # Average transaction size
        'trans_transaction_amount_std',  # Transaction variability
        'trans_fraud_flag_sum'  # Number of fraudulent transactions
    ]


def preprocess_features(df, feature_columns, fill_missing=True):
    """Standard preprocessing for features"""

    # Ensure all features exist
    for col in feature_columns:
        if col not in df.columns:
            if fill_missing:
                df[col] = 0
                logger.warning(f"Missing feature {col}, filled with 0")
            else:
                raise ValueError(f"Required feature {col} not found in data")

    # Handle missing values
    if fill_missing:
        df[feature_columns] = df[feature_columns].fillna(0)

    # Return features and target
    X = df[feature_columns].copy()
    y = df['fraud_flag'].copy() if 'fraud_flag' in df.columns else None

    return X, y


def calculate_performance_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive performance metrics"""

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    if y_pred_proba is not None:
        metrics['auroc'] = roc_auc_score(y_true, y_pred_proba)

    # Precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1

    # Additional info
    metrics['fraud_rate'] = float(y_true.mean())
    metrics['sample_count'] = len(y_true)

    return metrics


def save_performance_history(month, metrics, notes="normal"):
    """Update performance history CSV"""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history_file = METRICS_DIR / "performance_history.csv"

    # Create new record
    new_record = {
        'timestamp': datetime.now().isoformat(),
        'month': f"{month:02d}",
        'accuracy': metrics['accuracy'],
        'auroc': metrics.get('auroc', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'f1_score': metrics.get('f1_score', 0.0),
        'fraud_rate': metrics['fraud_rate'],
        'sample_count': metrics['sample_count'],
        'notes': notes
    }

    # Create or append to history
    if history_file.exists():
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        history_df = pd.DataFrame([new_record])

    # Save updated history
    history_df.to_csv(history_file, index=False)
    logger.info(f"Performance history updated for month {month:02d}")


def archive_current_model():
    """Archive current model before replacement"""

    if not (CURRENT_DIR / "model.pkl").exists():
        logger.info("No current model to archive")
        return None

    # Create timestamped archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = ARCHIVE_DIR / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Copy current model files to archive
    import shutil
    for file_path in CURRENT_DIR.glob("*"):
        if file_path.is_file():
            shutil.copy2(file_path, archive_dir / file_path.name)

    logger.info(f"Model archived to {archive_dir}")
    return archive_dir


def validate_model_artifacts():
    """Check if all required model artifacts exist"""

    required_files = [
        "model.pkl",
        "scaler.pkl",
        "encoders.pkl",
        "metadata.json"
    ]

    missing_files = []
    for filename in required_files:
        if not (CURRENT_DIR / filename).exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(f"Missing model artifacts: {missing_files}")

    logger.info("All model artifacts validated successfully")
    return True


def get_drift_months():
    """Return months where drift is expected (for testing)"""
    return [6, 9, 11]


def is_drift_month(month):
    """Check if current month is a known drift month"""
    return month in get_drift_months()


def determine_drift_type(month):
    """Determine what type of drift to expect for a given month"""
    drift_map = {
        6: "feature_drift_debt_to_income",
        9: "fraud_rate_increase",
        11: "transaction_behavior_and_demographics"
    }
    return drift_map.get(month, "none")


if __name__ == "__main__":
    # Test utility functions
    print("Fraud Detection MLOps Utilities")
    print("=" * 40)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Standard features: {get_standard_features()}")
    print(f"Expected drift months: {get_drift_months()}")