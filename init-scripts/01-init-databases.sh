#!/bin/bash
set -e

echo '=== Installing pgvector extension ==='

# Устанавливаем build tools если их нет
apt-get update && apt-get install -y postgresql-17-pgvector || true

echo '=== Creating databases for Terra and Sport projects ==='

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
    -- Создаём базу данных для проекта Terra
    CREATE DATABASE mem0_terra;
    
    -- Создаём базу данных для проекта Sport
    CREATE DATABASE mem0_sport;
    
    -- Включаем pgvector для каждой базы
    \c mem0_terra
    CREATE EXTENSION IF NOT EXISTS vector;
    
    \c mem0_sport
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL

echo '=== PostgreSQL databases created: mem0_terra, mem0_sport ==='
