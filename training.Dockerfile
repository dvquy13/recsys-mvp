# Start from Python 3.11.9 base image
FROM python:3.11.9-slim
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    # libpq-dev is needed for psycopg2
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy Poetry files
COPY uv.lock pyproject.toml ./

# Install Python dependencies using Poetry
RUN uv sync --group ml --group pipeline

RUN mkdir data
COPY notebooks/*.ipynb ./notebooks/
COPY notebooks/*.py ./notebooks/
COPY src/ ./src/
COPY feature_store_offline_server.yaml ./

WORKDIR /app/notebooks
