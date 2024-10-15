# Start from Python 3.11.9 base image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    # libpq-dev and python3-dev is needed for psycopg2
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
RUN poetry install --no-root --with pipeline --without dev

COPY dbt/ ./dbt/
COPY notebooks/*.ipynb ./notebooks/
COPY notebooks/*.py ./notebooks/
COPY src/ ./src/

RUN mkdir -p ./notebooks/papermill-output

WORKDIR /app/notebooks
