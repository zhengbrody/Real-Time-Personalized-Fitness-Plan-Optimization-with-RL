# Multi-stage Dockerfile for ProFit AI
# Supports: API server, Web interface, Kafka consumer

# Base stage with common dependencies
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create data/model directories (data & models are gitignored; populated at runtime)
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features /app/models

# Set Python path
ENV PYTHONPATH=/app

# ============================================
# API Server Stage
# ============================================
FROM base AS api

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
FROM base AS web

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
FROM base AS consumer

WORKDIR /app

# No exposed ports (consumer only)

# Run Kafka consumer
CMD ["python", "src/online_learning/kafka_consumer.py"]

# ============================================
# Development Stage (with dev dependencies)
# ============================================
FROM base AS development

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
FROM python:3.11-slim AS production

WORKDIR /app

# Install system dependencies (slim has glibc for compiled packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/

# Create models directory (models/ is gitignored; populated at runtime or via volume)
RUN mkdir -p /app/models

# Create non-root user
RUN useradd --no-create-home --uid 1000 profit && \
    chown -R profit:profit /app

# Switch to non-root user
USER profit

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "src/serving/api_server.py"]
