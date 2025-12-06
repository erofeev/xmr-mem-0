-- MCP Team Memory v3.0 - PostgreSQL Init
-- Cipher + Graphiti architecture

-- Create extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Database for Cipher
CREATE DATABASE cipher;

\c cipher

CREATE EXTENSION IF NOT EXISTS vector;

-- Cipher chat history table (for session management)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_chat_sessions_project ON chat_sessions(project_id);
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);

-- Graphiti sync table (for temporal queries via SQL)
CREATE TABLE IF NOT EXISTS graphiti_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100),
    decision TEXT NOT NULL,
    reasoning TEXT,
    module VARCHAR(200),
    tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    graphiti_episode_id VARCHAR(100)
);

CREATE INDEX idx_decisions_project ON graphiti_decisions(project_id);
CREATE INDEX idx_decisions_module ON graphiti_decisions(module);
CREATE INDEX idx_decisions_created ON graphiti_decisions(created_at);

CREATE TABLE IF NOT EXISTS graphiti_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    activity_type VARCHAR(100) NOT NULL,
    module VARCHAR(200),
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    graphiti_episode_id VARCHAR(100)
);

CREATE INDEX idx_activities_project ON graphiti_activities(project_id);
CREATE INDEX idx_activities_user ON graphiti_activities(user_id);
CREATE INDEX idx_activities_module ON graphiti_activities(module);
CREATE INDEX idx_activities_created ON graphiti_activities(created_at);

-- View for "who worked on what in last N days"
CREATE OR REPLACE VIEW recent_activities AS
SELECT 
    project_id,
    user_id,
    module,
    activity_type,
    description,
    created_at,
    created_at::date as activity_date
FROM graphiti_activities
ORDER BY created_at DESC;

-- View for decisions by module
CREATE OR REPLACE VIEW module_decisions AS
SELECT 
    project_id,
    module,
    decision,
    reasoning,
    user_id,
    created_at,
    tags
FROM graphiti_decisions
ORDER BY created_at DESC;

-- Function: get activities for last N days
CREATE OR REPLACE FUNCTION get_recent_activities(
    p_project_id VARCHAR,
    p_days INTEGER DEFAULT 14
)
RETURNS TABLE (
    user_id VARCHAR,
    module VARCHAR,
    activity_type VARCHAR,
    description TEXT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.user_id,
        a.module,
        a.activity_type,
        a.description,
        a.created_at
    FROM graphiti_activities a
    WHERE a.project_id = p_project_id
      AND a.created_at >= NOW() - (p_days || ' days')::INTERVAL
    ORDER BY a.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: get decisions by module
CREATE OR REPLACE FUNCTION get_module_decisions(
    p_project_id VARCHAR,
    p_module VARCHAR
)
RETURNS TABLE (
    decision TEXT,
    reasoning TEXT,
    user_id VARCHAR,
    created_at TIMESTAMPTZ,
    tags TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.decision,
        d.reasoning,
        d.user_id,
        d.created_at,
        d.tags
    FROM graphiti_decisions d
    WHERE d.project_id = p_project_id
      AND (p_module IS NULL OR d.module ILIKE '%' || p_module || '%')
    ORDER BY d.created_at DESC;
END;
$$ LANGUAGE plpgsql;

