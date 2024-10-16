# Start from Python 3.11.9 base image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    # libpq-dev is needed for psycopg2
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Create and set the working directory
WORKDIR /app

# Copy Poetry files
COPY poetry.lock pyproject.toml ./

# Install Python dependencies using Poetry
RUN poetry install --no-root --with pipeline,training --without dev

COPY feature_pipeline/notebooks/*.ipynb ./feature_pipeline/notebooks/
COPY feature_pipeline/notebooks/*.py ./feature_pipeline/notebooks/
COPY feature_pipeline/src/ ./feature_pipeline/src/
COPY src/ ./src/

WORKDIR /app/feature_pipeline/notebooks

RUN mkdir -p ./papermill-output
