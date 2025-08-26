# Authentication module for Flask API

import os
from functools import wraps
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)


def get_bearer_token():
    """Get bearer token from environment variable"""
    return os.environ.get('API_TOKEN', 'your-secret-token-here')


def require_auth(f):
    """Decorator to require bearer token authentication"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get authorization header
        auth_header = request.headers.get('Authorization')

        if not auth_header:
            logger.warning("Missing Authorization header")
            return jsonify({"error": "Missing Authorization header"}), 401

        # Check bearer token format
        if not auth_header.startswith('Bearer '):
            logger.warning("Invalid Authorization header format")
            return jsonify({"error": "Invalid Authorization header format"}), 401

        # Extract token
        token = auth_header.split(' ')[1]
        expected_token = get_bearer_token()

        # Validate token
        if token != expected_token:
            logger.warning("Invalid bearer token")
            return jsonify({"error": "Invalid bearer token"}), 401

        # Token is valid, continue to endpoint
        return f(*args, **kwargs)

    return decorated_function