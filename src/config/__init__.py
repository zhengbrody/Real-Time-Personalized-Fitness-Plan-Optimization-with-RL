"""
Configuration management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Agent Configuration
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/fitness.db")

# Feature Store
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "./feature_repo")

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

