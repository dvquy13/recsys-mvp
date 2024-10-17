#!/bin/bash
set -e

# Connect to the 'postgres' database and create the 'airflow' user and 'airflow' database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE USER airflow WITH PASSWORD 'airflow';
    CREATE DATABASE airflow OWNER airflow;
EOSQL

# Connect to the database and create schemas using the environment variables
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS "$POSTGRES_OLTP_SCHEMA";
    CREATE SCHEMA IF NOT EXISTS "$POSTGRES_FEATURE_STORE_OFFLINE_SCHEMA";
    CREATE SCHEMA IF NOT EXISTS "$POSTGRES_FEATURE_STORE_ONLINE_SCHEMA";
EOSQL
