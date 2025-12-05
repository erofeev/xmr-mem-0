-- Create databases for all projects
CREATE DATABASE mem0_terra;
CREATE DATABASE mem0_sport;
CREATE DATABASE mem0_datashowcase;
CREATE DATABASE mem0_trialprj;

-- Enable pgvector extension for each database
\c mem0_terra
CREATE EXTENSION IF NOT EXISTS vector;

\c mem0_sport
CREATE EXTENSION IF NOT EXISTS vector;

\c mem0_datashowcase
CREATE EXTENSION IF NOT EXISTS vector;

\c mem0_trialprj
CREATE EXTENSION IF NOT EXISTS vector;
