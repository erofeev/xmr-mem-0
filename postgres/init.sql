-- Create databases for all projects
CREATE DATABASE mem0_terra;
CREATE DATABASE mem0_sport;
CREATE DATABASE mem0_datashowcase;
CREATE DATABASE mem0_trialprj;

-- Function to setup SSOT schema in a database
CREATE OR REPLACE FUNCTION setup_ssot_schema() RETURNS void AS $$
BEGIN
    -- Enable extensions
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- Deduplication Registry (SSOT for all systems)
    CREATE TABLE IF NOT EXISTS dedup_registry (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        content_hash VARCHAR(64) NOT NULL,
        embedding vector(768),
        source_system VARCHAR(50) NOT NULL,
        source_id VARCHAR(255) NOT NULL,
        canonical_id UUID REFERENCES dedup_registry(id),
        operation VARCHAR(20) NOT NULL,
        similarity_score FLOAT,
        metadata JSONB DEFAULT ''{}\',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        CONSTRAINT unique_content_source UNIQUE(content_hash, source_system)
    );

    -- Memories (Cipher-compatible)
    CREATE TABLE IF NOT EXISTS memories (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id VARCHAR(255) NOT NULL,
        project_id VARCHAR(255),
        content TEXT NOT NULL,
        embedding vector(768),
        memory_type VARCHAR(50) NOT NULL DEFAULT 'system1',
        domain VARCHAR(100),
        tags TEXT[] DEFAULT '{}'',
        target_memory_id UUID REFERENCES memories(id),
        dedup_id UUID REFERENCES dedup_registry(id),
        code_pattern TEXT,
        confidence FLOAT DEFAULT 1.0,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ,
        is_deleted BOOLEAN DEFAULT FALSE
    );

    -- Reasoning Traces (System 2)
    CREATE TABLE IF NOT EXISTS reasoning_traces (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        memory_id UUID REFERENCES memories(id),
        user_id VARCHAR(255) NOT NULL,
        project_id VARCHAR(255),
        steps JSONB NOT NULL,
        evaluation JSONB,
        task_context JSONB,
        searchable_content TEXT,
        embedding vector(768),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Workspace Memory (Team Context)
    CREATE TABLE IF NOT EXISTS workspace_memories (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        project_id VARCHAR(255) NOT NULL,
        team_member VARCHAR(255),
        activity_type VARCHAR(100),
        content TEXT NOT NULL,
        embedding vector(768),
        extracted_data JSONB,
        domain VARCHAR(100),
        status VARCHAR(50),
        progress_percent INT,
        dedup_id UUID REFERENCES dedup_registry(id),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Graphiti Episodes
    CREATE TABLE IF NOT EXISTS graphiti_episodes (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        graphiti_uuid VARCHAR(255) UNIQUE,
        user_id VARCHAR(255) NOT NULL,
        project_id VARCHAR(255),
        name VARCHAR(500),
        content TEXT NOT NULL,
        source VARCHAR(100) DEFAULT 'user',
        embedding vector(768),
        dedup_id UUID REFERENCES dedup_registry(id),
        valid_from TIMESTAMPTZ DEFAULT NOW(),
        valid_to TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Graphiti Entities
    CREATE TABLE IF NOT EXISTS graphiti_entities (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        graphiti_uuid VARCHAR(255) UNIQUE,
        name VARCHAR(500) NOT NULL,
        entity_type VARCHAR(100),
        summary TEXT,
        embedding vector(768),
        dedup_id UUID REFERENCES dedup_registry(id),
        valid_from TIMESTAMPTZ DEFAULT NOW(),
        valid_to TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Graphiti Facts
    CREATE TABLE IF NOT EXISTS graphiti_facts (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        graphiti_uuid VARCHAR(255) UNIQUE,
        source_entity_id UUID REFERENCES graphiti_entities(id),
        target_entity_id UUID REFERENCES graphiti_entities(id),
        relation_type VARCHAR(255),
        fact_text TEXT,
        embedding vector(768),
        episode_id UUID REFERENCES graphiti_episodes(id),
        valid_from TIMESTAMPTZ DEFAULT NOW(),
        valid_to TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Knowledge Base Documents
    CREATE TABLE IF NOT EXISTS kb_documents (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        project_id VARCHAR(255) NOT NULL,
        title VARCHAR(500),
        source_type VARCHAR(100),
        source_path TEXT,
        content TEXT NOT NULL,
        metadata JSONB DEFAULT ''{}\',
        dedup_id UUID REFERENCES dedup_registry(id),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Knowledge Base Chunks
    CREATE TABLE IF NOT EXISTS kb_chunks (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        document_id UUID REFERENCES kb_documents(id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        chunk_index INT NOT NULL,
        embedding vector(768),
        dedup_id UUID REFERENCES dedup_registry(id),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_dedup_embedding ON dedup_registry USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, project_id);
    CREATE INDEX IF NOT EXISTS idx_workspace_project ON workspace_memories(project_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON graphiti_episodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_entities_embedding ON graphiti_entities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON kb_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
END;
$$ LANGUAGE plpgsql;

-- Setup each database
\c mem0_terra
SELECT setup_ssot_schema();

-- Unified search view
CREATE OR REPLACE VIEW unified_search_view AS
SELECT id, 'memory' as source_type, content, embedding, user_id, project_id, created_at,
       jsonb_build_object('memory_type', memory_type, 'domain', domain) as metadata
FROM memories WHERE NOT is_deleted
UNION ALL
SELECT id, 'episode' as source_type, content, embedding, user_id, project_id, created_at,
       jsonb_build_object('name', name) as metadata
FROM graphiti_episodes
UNION ALL
SELECT id, 'entity' as source_type, COALESCE(summary, name) as content, embedding, 
       NULL as user_id, NULL as project_id, created_at,
       jsonb_build_object('name', name, 'entity_type', entity_type) as metadata
FROM graphiti_entities
UNION ALL
SELECT c.id, 'kb_chunk' as source_type, c.content, c.embedding,
       NULL as user_id, d.project_id, c.created_at,
       jsonb_build_object('document_title', d.title) as metadata
FROM kb_chunks c JOIN kb_documents d ON c.document_id = d.id;

-- Unified search function
CREATE OR REPLACE FUNCTION unified_semantic_search(
    query_embedding vector(768),
    similarity_threshold FLOAT DEFAULT 0.5,
    max_results INT DEFAULT 10,
    filter_project VARCHAR DEFAULT NULL,
    filter_user VARCHAR DEFAULT NULL
)
RETURNS TABLE (id UUID, source_type VARCHAR, content TEXT, similarity FLOAT, metadata JSONB, created_at TIMESTAMPTZ) AS $$
BEGIN
    RETURN QUERY
    SELECT v.id, v.source_type::VARCHAR, v.content,
           (1 - (v.embedding <=> query_embedding))::FLOAT as similarity,
           v.metadata, v.created_at
    FROM unified_search_view v
    WHERE (1 - (v.embedding <=> query_embedding)) >= similarity_threshold
      AND (filter_project IS NULL OR v.project_id = filter_project)
      AND (filter_user IS NULL OR v.user_id = filter_user)
    ORDER BY v.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

\c mem0_sport
SELECT setup_ssot_schema();

\c mem0_datashowcase
SELECT setup_ssot_schema();

\c mem0_trialprj
SELECT setup_ssot_schema();
