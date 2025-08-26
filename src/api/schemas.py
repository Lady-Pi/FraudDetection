# Request/Response schemas and validation for Flask API

from typing import Dict, List, Union, Any

# Required fields matching trained model's 14 features
REQUIRED_FIELDS = [
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
    'trans_transaction_amount_max'
]

# Optional fields
OPTIONAL_FIELDS = [
    'employment_status',
    'property_ownership_status',
    'gender',
    'residential_address'
]

ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS


class PredictionRequest:
    """Schema for prediction requests"""

    def __init__(self, data: Union[Dict, List[Dict]]):
        self.data = data
        self.is_batch = isinstance(data, list)

    def validate(self) -> Union[str, None]:
        """Validate request data format"""
        try:
            if self.is_batch:
                if not isinstance(self.data, list) or len(self.data) == 0:
                    return "Batch request must be a non-empty list"

                for i, record in enumerate(self.data):
                    error = self._validate_single_record(record, f"Record {i}")
                    if error:
                        return error
            else:
                error = self._validate_single_record(self.data, "Single record")
                if error:
                    return error

            return None  # No errors

        except Exception as e:
            return f"Validation error: {str(e)}"

    def _validate_single_record(self, record: Dict, context: str) -> Union[str, None]:
        """Validate a single record"""
        if not isinstance(record, dict):
            return f"{context}: must be a JSON object"

        # Check for required fields
        missing_fields = []
        for field in REQUIRED_FIELDS:
            if field not in record or record[field] is None:
                missing_fields.append(field)

        if missing_fields:
            return f"{context}: missing required fields: {missing_fields}"

        # Validate field types and ranges
        try:
            # Basic numeric validations
            if not isinstance(record.get('loan_amount_requested'), (int, float)) or record[
                'loan_amount_requested'] <= 0:
                return f"{context}: loan_amount_requested must be positive number"

            if not isinstance(record.get('monthly_income'), (int, float)) or record['monthly_income'] <= 0:
                return f"{context}: monthly_income must be positive number"

            if not isinstance(record.get('debt_to_income_ratio'), (int, float)) or record['debt_to_income_ratio'] < 0:
                return f"{context}: debt_to_income_ratio must be non-negative number"

            if not isinstance(record.get('applicant_age'), (int, float)) or not (18 <= record['applicant_age'] <= 100):
                return f"{context}: applicant_age must be between 18 and 100"

            if not isinstance(record.get('number_of_dependents'), int) or record['number_of_dependents'] < 0:
                return f"{context}: number_of_dependents must be non-negative integer"

            if not isinstance(record.get('cibil_score'), (int, float)) or not (300 <= record['cibil_score'] <= 850):
                return f"{context}: cibil_score must be between 300 and 850"

            if not isinstance(record.get('existing_emis_monthly'), (int, float)) or record['existing_emis_monthly'] < 0:
                return f"{context}: existing_emis_monthly must be non-negative number"

            if not isinstance(record.get('interest_rate_offered'), (int, float)) or record[
                'interest_rate_offered'] <= 0:
                return f"{context}: interest_rate_offered must be positive number"

            if not isinstance(record.get('loan_tenure_months'), int) or record['loan_tenure_months'] <= 0:
                return f"{context}: loan_tenure_months must be positive integer"

            # Transaction aggregation features
            transaction_fields = ['trans_transaction_amount_count', 'trans_transaction_amount_sum',
                                  'trans_transaction_amount_mean', 'trans_transaction_amount_std',
                                  'trans_transaction_amount_max']

            for field in transaction_fields:
                if not isinstance(record.get(field), (int, float)) or record[field] < 0:
                    return f"{context}: {field} must be non-negative number"

            return None  # No errors

        except (KeyError, TypeError, ValueError) as e:
            return f"{context}: data type validation error: {str(e)}"


class PredictionResponse:
    """Schema for prediction responses"""

    def __init__(self, fraud_probability: float, prediction: int, confidence: str = "medium"):
        self.fraud_probability = fraud_probability
        self.prediction = prediction
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraud_probability": self.fraud_probability,
            "prediction": self.prediction,
            "confidence": self.confidence
        }


def validate_request(data: Union[Dict, List[Dict]]) -> Union[str, None]:
    """Validate incoming request data"""
    request_obj = PredictionRequest(data)
    return request_obj.validate()


# Example valid request format for testing
EXAMPLE_SINGLE_REQUEST = {
    "debt_to_income_ratio": 0.3,
    "applicant_age": 35,
    "cibil_score": 750,
    "loan_amount_requested": 50000,
    "monthly_income": 8000,
    "existing_emis_monthly": 1200,
    "interest_rate_offered": 8.5,
    "loan_tenure_months": 24,
    "number_of_dependents": 2,
    "trans_transaction_amount_count": 45,
    "trans_transaction_amount_sum": 25000,
    "trans_transaction_amount_mean": 555.56,
    "trans_transaction_amount_std": 200.0,
    "trans_transaction_amount_max": 2000
}

EXAMPLE_BATCH_REQUEST = [
    EXAMPLE_SINGLE_REQUEST,
    {
        "debt_to_income_ratio": 0.4,
        "applicant_age": 28,
        "cibil_score": 680,
        "loan_amount_requested": 25000,
        "monthly_income": 5000,
        "existing_emis_monthly": 800,
        "interest_rate_offered": 9.2,
        "loan_tenure_months": 36,
        "number_of_dependents": 1,
        "trans_transaction_amount_count": 32,
        "trans_transaction_amount_sum": 18000,
        "trans_transaction_amount_mean": 562.5,
        "trans_transaction_amount_std": 150.0,
        "trans_transaction_amount_max": 1500
    }
]