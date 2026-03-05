# Multi-stage Dockerfile for ProFit AI
# Supports: API server, Web interface, Kafka consumer

# Base stage with common dependencies
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features /app/models

# Set Python path
ENV PYTHONPATH=/app

# ============================================
# API Server Stage
# ============================================
FROM base as api

WORKDIR /app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "src/serving/api_server.py"]

# ============================================
# Web Interface Stage
# ============================================
FROM base as web

WORKDIR /app

# Copy web application
COPY web_app_pro.py .
COPY .env.example .env

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "web_app_pro.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]

# ============================================
# Kafka Consumer Stage
# ============================================
FROM base as consumer

WORKDIR /app

# No exposed ports (consumer only)

# Run Kafka consumer
CMD ["python", "src/online_learning/kafka_consumer.py"]

# ============================================
# Development Stage (with dev dependencies)
# ============================================
FROM base as development

WORKDIR /app

# Copy dev requirements
COPY requirements-dev.txt* ./

# Install dev dependencies if file exists
RUN if [ -f requirements-dev.txt ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    fi

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Copy test files
COPY tests/ ./tests/

# Copy all application code
COPY . .

# Default to interactive shell
CMD ["/bin/bash"]

# ============================================
# Production Stage (optimized, minimal)
# ============================================
FROM python:3.11-alpine as production

WORKDIR /app

# Install minimal system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    curl

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN adduser -D -u 1000 profit && \
    chown -R profit:profit /app

# Switch to non-root user
USER profit

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "src/serving/api_server.py"]
