"""
Data Collection Module

This module provides tools for collecting data from various sources:
- Oura Ring API
- Apple Health export
- Training logs
"""

# Import Apple Health parser (no external dependencies)
from .apple_health import AppleHealthParser

# Import Training Logger (no external dependencies)
from .training_log import TrainingLogger

# Import Oura API (optional - requires python-oura)
try:
    from .oura_api import OuraDataCollector
    __all__ = [
        'OuraDataCollector',
        'AppleHealthParser',
        'TrainingLogger',
    ]
except ImportError:
    # Oura API not available (python-oura not installed)
    # This is OK if you already have Oura CSV data
    __all__ = [
        'AppleHealthParser',
        'TrainingLogger',
    ]

