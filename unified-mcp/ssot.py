"""
SSOT Layer: PostgreSQL + pgvector + Deduplication Orchestrator
"""
import hashlib
import logging
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

import httpx
import asyncpg

logger = logging.getLogger(__name__)


class PostgresSSOT:
    """PostgreSQL as Single Source of Truth"""
    
    def __init__(self, database_url: str, project_id: str):
        self.database_url = database_url
        self.project_id = project_id
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Create connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info(f"PostgreSQL SSOT initialized for project {self.project_id}")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    async def close(self):
        if self.pool:
            await self.pool.close()
    
    async def check_duplicate(self, content_hash: str, embedding: List[float],
                              similarity_threshold: float = 0.9) -> Optional[Dict]:
        """Check for duplicates in registry"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, source_system, source_id FROM dedup_registry WHERE content_hash = $1",
                content_hash
            )
            if row:
                return {
                    "is_duplicate": True,
                    "id": str(row["id"]),
                    "source_system": row["source_system"],
                    "similarity": 1.0,
                    "reason": "exact_hash"
                }
            
            if embedding is not None and len(embedding) > 0:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                row = await conn.fetchrow(
                    """
                    SELECT id, source_system, source_id,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM dedup_registry
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT 1
                    """,
                    embedding_str
                )
                if row and row["similarity"] and row["similarity"] >= similarity_threshold:
                    return {
                        "is_duplicate": True,
                        "id": str(row["id"]),
                        "source_system": row["source_system"],
                        "similarity": float(row["similarity"]),
                        "reason": "embedding_similar"
                    }
        return None
    
    async def register_content(self, content_hash: str, embedding: List[float],
                               source_system: str, source_id: str,
                               operation: str, similarity_score: float = 0.0) -> str:
        """Register content in dedup registry"""
        async with self.pool.acquire() as conn:
            embedding_str = None
            if embedding is not None and len(embedding) > 0:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            row = await conn.fetchrow(
                """
                INSERT INTO dedup_registry 
                (content_hash, embedding, source_system, source_id, operation, similarity_score)
                VALUES ($1, $2::vector, $3, $4, $5, $6)
                ON CONFLICT (content_hash, source_system) DO UPDATE SET
                    updated_at = NOW(), operation = EXCLUDED.operation
                RETURNING id
                """,
                content_hash, embedding_str, source_system, source_id,
                operation, similarity_score
            )
            return str(row["id"])
    
    async def add_memory(self, user_id: str, content: str, embedding: List[float],
                         memory_type: str = "system1", project_id: str = None,
                         domain: str = None, tags: List[str] = None) -> str:
        """Add memory to PostgreSQL SSOT"""
        async with self.pool.acquire() as conn:
            embedding_str = None
            if embedding is not None and len(embedding) > 0:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            row = await conn.fetchrow(
                """
                INSERT INTO memories 
                (user_id, project_id, content, embedding, memory_type, domain, tags)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
                RETURNING id
                """,
                user_id, project_id or self.project_id, content, embedding_str,
                memory_type, domain, tags or []
            )
            return str(row["id"])
    
    async def search_memories(self, embedding: List[float], limit: int = 10,
                              user_id: str = None, project_id: str = None,
                              similarity_threshold: float = 0.5) -> List[Dict]:
        """Search memories by embedding similarity"""
        async with self.pool.acquire() as conn:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            rows = await conn.fetch(
                """
                SELECT id, user_id, project_id, content, memory_type, domain,
                       created_at, 1 - (embedding <=> $1::vector) as similarity
                FROM memories
                WHERE NOT is_deleted
                  AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str, similarity_threshold, limit
            )
            return [
                {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "domain": row["domain"],
                    "similarity": float(row["similarity"]) if row["similarity"] else 0
                }
                for row in rows
            ]
    
    async def unified_search(self, embedding: List[float], limit: int = 10,
                            similarity_threshold: float = 0.5) -> List[Dict]:
        """Search across all sources"""
        async with self.pool.acquire() as conn:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            rows = await conn.fetch(
                """
                SELECT * FROM unified_semantic_search($1::vector, $2, $3, NULL, NULL)
                """,
                embedding_str, similarity_threshold, limit
            )
            return [
                {
                    "id": str(row["id"]),
                    "source_type": row["source_type"],
                    "content": row["content"],
                    "similarity": float(row["similarity"]) if row["similarity"] else 0
                }
                for row in rows
            ]
    
    async def add_workspace_memory(self, project_id: str, content: str,
                                   embedding: List[float], team_member: str = None,
                                   activity_type: str = None, status: str = None) -> str:
        """Add workspace memory"""
        async with self.pool.acquire() as conn:
            embedding_str = None
            if embedding is not None and len(embedding) > 0:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            row = await conn.fetchrow(
                """
                INSERT INTO workspace_memories 
                (project_id, content, embedding, team_member, activity_type, status)
                VALUES ($1, $2, $3::vector, $4, $5, $6)
                RETURNING id
                """,
                project_id, content, embedding_str, team_member, activity_type, status
            )
            return str(row["id"])
    
    async def sync_episode(self, graphiti_uuid: str, user_id: str, content: str,
                          embedding: List[float], name: str = None) -> str:
        """Sync Graphiti episode to PostgreSQL"""
        async with self.pool.acquire() as conn:
            embedding_str = None
            if embedding is not None and len(embedding) > 0:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            row = await conn.fetchrow(
                """
                INSERT INTO graphiti_episodes 
                (graphiti_uuid, user_id, project_id, name, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (graphiti_uuid) DO UPDATE SET
                    content = EXCLUDED.content, embedding = EXCLUDED.embedding
                RETURNING id
                """,
                graphiti_uuid, user_id, self.project_id, name, content, embedding_str
            )
            return str(row["id"])


class DeduplicationOrchestrator:
    """5-layer deduplication based on Cipher approach"""
    
    def __init__(self, ssot: PostgresSSOT, embedder_func):
        self.ssot = ssot
        self.get_embedding = embedder_func
        self.similarity_threshold = 0.7
        self.high_similarity_threshold = 0.9
    
    async def process(self, content: str, source_system: str,
                      user_id: str, project_id: str = None) -> Dict:
        """5-layer deduplication"""
        result = {
            "operation": "ADD",
            "reason": None,
            "similar_id": None,
            "similarity_score": 0.0
        }
        
        # Layer 1: Significance filtering
        if len(content.strip()) < 10:
            result["operation"] = "NONE"
            result["reason"] = "too_short"
            return result
        
        trivial = ["hello", "hi", "thanks", "ok", "yes", "no", "да", "нет", "привет"]
        if content.lower().strip() in trivial:
            result["operation"] = "NONE"
            result["reason"] = "trivial_content"
            return result
        
        # Layer 2+3: Hash + Embedding check
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        embedding = await self.get_embedding(content)
        
        if embedding is None or len(embedding) == 0:
            return result
        
        dup_check = await self.ssot.check_duplicate(
            content_hash, embedding, self.high_similarity_threshold
        )
        
        if dup_check:
            # Layer 4: High similarity gate
            if dup_check["similarity"] >= self.high_similarity_threshold:
                result["operation"] = "NONE"
                result["reason"] = dup_check["reason"]
                result["similar_id"] = dup_check["id"]
                result["similarity_score"] = dup_check["similarity"]
                return result
            
            # Layer 5: Cross-system check
            if dup_check["source_system"] != source_system:
                result["operation"] = "MERGE"
                result["reason"] = "cross_system"
            else:
                result["operation"] = "UPDATE"
                result["reason"] = "same_system"
            
            result["similar_id"] = dup_check["id"]
            result["similarity_score"] = dup_check["similarity"]
        
        return result
    
    async def register(self, content: str, source_system: str, source_id: str,
                       operation: str, embedding: List[float] = None) -> str:
        """Register content after operation"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if embedding is None:
            embedding = await self.get_embedding(content)
        
        return await self.ssot.register_content(
            content_hash, embedding, source_system, source_id, operation
        )
