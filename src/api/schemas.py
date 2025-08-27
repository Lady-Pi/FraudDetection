# Request/Response schemas and validation for Flask API

# Request/Response schemas and validation for Flask API

from typing import Dict, List, Union, Any

# Core required fields that users would typically provide
REQUIRED_FIELDS = [
    'loan_amount_requested',
    'monthly_income',
    'applicant_age',
    'cibil_score',
    'number_of_dependents'
]

# Optional fields with defaults
OPTIONAL_FIELDS = [
    'debt_to_income_ratio',
    'existing_emis_monthly',
    'interest_rate_offered',
    'loan_tenure_months',
    'trans_transaction_amount_count',
    'trans_transaction_amount_sum',
    'trans_transaction_amount_mean',
    'trans_transaction_amount_std',
    'trans_transaction_amount_max'
]

ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

# Default values for optional fields
FIELD_DEFAULTS = {
    'debt_to_income_ratio': 0.3,  # Reasonable default
    'existing_emis_monthly': 0.0,  # No existing EMIs
    'interest_rate_offered': 10.0,  # Market average
    'loan_tenure_months': 24,  # 2 years
    'trans_transaction_amount_count': 30,  # Monthly average
    'trans_transaction_amount_sum': 50000,  # Reasonable sum
    'trans_transaction_amount_mean': 1666.67,  # Derived from sum/count
    'trans_transaction_amount_std': 500.0,  # Moderate variability
    'trans_transaction_amount_max': 5000  # Reasonable max transaction
}


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

        # Check for required fields only
        missing_fields = []
        for field in REQUIRED_FIELDS:
            if field not in record or record[field] is None:
                missing_fields.append(field)

        if missing_fields:
            return f"{context}: missing required fields: {missing_fields}"

        # Validate field types and ranges for provided fields
        try:
            # Required field validations
            if not isinstance(record.get('loan_amount_requested'), (int, float)) or record[
                'loan_amount_requested'] <= 0:
                return f"{context}: loan_amount_requested must be positive number"

            if not isinstance(record.get('monthly_income'), (int, float)) or record['monthly_income'] <= 0:
                return f"{context}: monthly_income must be positive number"

            if not isinstance(record.get('applicant_age'), (int, float)) or not (18 <= record['applicant_age'] <= 100):
                return f"{context}: applicant_age must be between 18 and 100"

            if not isinstance(record.get('cibil_score'), (int, float)) or not (300 <= record['cibil_score'] <= 850):
                return f"{context}: cibil_score must be between 300 and 850"

            if not isinstance(record.get('number_of_dependents'), int) or record['number_of_dependents'] < 0:
                return f"{context}: number_of_dependents must be non-negative integer"

            # Optional field validations
            if 'debt_to_income_ratio' in record:
                if not isinstance(record['debt_to_income_ratio'], (int, float)) or record['debt_to_income_ratio'] < 0:
                    return f"{context}: debt_to_income_ratio must be non-negative number"

            if 'existing_emis_monthly' in record:
                if not isinstance(record['existing_emis_monthly'], (int, float)) or record['existing_emis_monthly'] < 0:
                    return f"{context}: existing_emis_monthly must be non-negative number"

            if 'interest_rate_offered' in record:
                if not isinstance(record['interest_rate_offered'], (int, float)) or record[
                    'interest_rate_offered'] <= 0:
                    return f"{context}: interest_rate_offered must be positive number"

            if 'loan_tenure_months' in record:
                if not isinstance(record['loan_tenure_months'], int) or record['loan_tenure_months'] <= 0:
                    return f"{context}: loan_tenure_months must be positive integer"

            # Transaction field validations (only if provided)
            transaction_fields = ['trans_transaction_amount_count', 'trans_transaction_amount_sum',
                                  'trans_transaction_amount_mean', 'trans_transaction_amount_std',
                                  'trans_transaction_amount_max']

            for field in transaction_fields:
                if field in record:
                    if not isinstance(record[field], (int, float)) or record[field] < 0:
                        return f"{context}: {field} must be non-negative number"

            return None  # No errors

        except (KeyError, TypeError, ValueError) as e:
            return f"{context}: data type validation error: {str(e)}"

    def apply_defaults(self, data: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
        """Apply default values for missing optional fields"""
        if isinstance(data, list):
            return [self._apply_defaults_single(record) for record in data]
        else:
            return self._apply_defaults_single(data)

    def _apply_defaults_single(self, record: Dict) -> Dict:
        """Apply defaults to a single record"""
        result = record.copy()

        for field, default_value in FIELD_DEFAULTS.items():
            if field not in result:
                result[field] = default_value

        # Calculate debt_to_income_ratio if not provided
        if 'debt_to_income_ratio' not in result and 'existing_emis_monthly' in result:
            result['debt_to_income_ratio'] = result['existing_emis_monthly'] / result['monthly_income']

        return result


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


def apply_defaults_to_request(data: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """Apply default values to request data"""
    request_obj = PredictionRequest(data)
    return request_obj.apply_defaults(data)


# Example valid request format for testing
EXAMPLE_MINIMAL_REQUEST = {
    "loan_amount_requested": 50000,
    "monthly_income": 8000,
    "applicant_age": 35,
    "cibil_score": 750,
    "number_of_dependents": 2
}

# Example with optional fields
EXAMPLE_FULL_REQUEST = {
    "loan_amount_requested": 50000,
    "monthly_income": 8000,
    "applicant_age": 35,
    "cibil_score": 750,
    "number_of_dependents": 2,
    "debt_to_income_ratio": 0.3,
    "existing_emis_monthly": 1200,
    "interest_rate_offered": 8.5,
    "loan_tenure_months": 24,
    "trans_transaction_amount_count": 45,
    "trans_transaction_amount_sum": 25000,
    "trans_transaction_amount_mean": 555.56,
    "trans_transaction_amount_std": 200.0,
    "trans_transaction_amount_max": 2000
}

EXAMPLE_BATCH_REQUEST = [EXAMPLE_MINIMAL_REQUEST, EXAMPLE_FULL_REQUEST]