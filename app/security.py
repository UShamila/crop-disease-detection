import os
import secrets
from functools import wraps
from flask import jsonify, request


def admin_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        expected = os.environ.get('ADMIN_API_KEY', '')
        supplied = request.headers.get('X-Admin-Key', '')
        if not expected or len(expected) < 24:
            return jsonify({'error': 'Admin access is not configured.'}), 503
        if not secrets.compare_digest(supplied, expected):
            return jsonify({'error': 'Administrator key required or incorrect.'}), 401
        return function(*args, **kwargs)
    return wrapper
