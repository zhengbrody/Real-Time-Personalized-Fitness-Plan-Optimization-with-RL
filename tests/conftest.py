"""Shared test fixtures for the RL project."""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.serving.api_server import app
from src.recommendation.action_space import ActionSpace
from src.recommendation.contextual_bandits import ContextualBandit


@pytest.fixture
def api_client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def tmp_memory_path(tmp_path):
    """Return a temporary file path for coach memory JSON storage."""
    return str(tmp_path / "test_memory.json")


@pytest.fixture
def sample_state():
    """Return a valid state dict usable across multiple test modules."""
    return {
        "readiness_score": 75,
        "hrv": 50,
        "resting_hr": 60,
        "sleep_duration_hours": 8,
        "fatigue": 3,
        "soreness": 2,
        "activity_score": 70,
    }


@pytest.fixture
def sample_bandit():
    """Return an initialized ContextualBandit with the default action space."""
    action_space = ActionSpace()
    return ContextualBandit(action_space=action_space)
