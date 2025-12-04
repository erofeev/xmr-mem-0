"""
Unified MCP Server for Mem0 + Knowledge Base + Cognee

Combines:
- Mem0 memory tools (via HTTP API to mem0-server) - PER USER isolation
- Knowledge Base tools (local FAISS indexes) - SHARED per project
- Cognee knowledge graph tools - SHARED per project

Isolation model:
- PROJECT_ID: determines which project (terra, sport, etc.)
- user_id (parameter): determines memory isolation within project
- Knowledge Base and Cognee are shared across all users in the project
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
PROJECT_ID = os.getenv("CURRENT_PROJECT_ID", "default")
MEM0_API_URL = os.getenv("MEM0_API_URL", "http://localhost:8000")
KNOWLEDGE_BASES_ROOT = Path(os.getenv("KNOWLEDGE_BASES_ROOT", "/data/knowledge_bases"))
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "/data/faiss_indexes"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:1b")
KB_CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "1000"))
KB_CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "200"))
COGNEE_ENABLED = os.getenv("COGNEE_ENABLED", "false").lower() == "true"
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")

# Neo4j for Cognee
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Global instances
http_client: Optional[httpx.AsyncClient] = None
kb_manager: Optional["KnowledgeBaseManager"] = None
cognee_manager: Optional["CogneeManager"] = None

# Initialize FastMCP server
mcp = FastMCP(f"unified-{PROJECT_ID}", host="0.0.0.0")


def get_full_user_id(user_id: Optional[str] = None) -> str:
    """Generate full user_id with project prefix for Mem0 isolation."""
    uid = user_id or DEFAULT_USER_ID
    return f"{PROJECT_ID}_{uid}"


# ============================================
# Cognee Manager (Knowledge Graph)
# ============================================
class CogneeManager:
    """Manages Cognee knowledge graph operations.

    Cognee provides:
    - Entity extraction from text
    - Relationship detection between entities
    - Knowledge graph storage (Neo4j)
    - Graph-based search and reasoning
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.initialized = False
        self._cognee = None

    async def initialize(self):
        """Initialize Cognee with configuration.

        Cognee 0.4.x uses environment variables for configuration:
        - LLM_PROVIDER, LLM_MODEL, LLM_ENDPOINT, LLM_API_KEY
        - EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_ENDPOINT
        - GRAPH_DATABASE_PROVIDER, GRAPH_DATABASE_URL, GRAPH_DATABASE_USERNAME
        """
        if self.initialized:
            return True

        try:
            import cognee
            self._cognee = cognee

            # Cognee 0.4.x reads configuration from environment variables
            # No need to call config.set_* methods - they are deprecated

            # Just verify cognee is importable and ready
            logger.info(f"📊 Cognee version: {cognee.__version__ if hasattr(cognee, '__version__') else 'unknown'}")
            logger.info(f"   LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'not set')}")
            logger.info(f"   LLM_MODEL: {os.getenv('LLM_MODEL', 'not set')}")
            logger.info(f"   GRAPH_DATABASE_PROVIDER: {os.getenv('GRAPH_DATABASE_PROVIDER', 'not set')}")

            self.initialized = True
            logger.info(f"✅ Cognee initialized for project '{self.project_id}'")
            return True

        except ImportError:
            logger.warning("⚠️ Cognee not installed. Install with: pip install cognee")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cognee: {e}")
            return False

    async def add_knowledge(self, content: str, source: str = "unknown") -> dict:
        """Add content to Cognee knowledge graph.

        This will:
        1. Extract entities from content
        2. Detect relationships
        3. Store in knowledge graph
        """
        if not self.initialized:
            await self.initialize()

        if not self._cognee:
            return {"error": "Cognee not available"}

        try:
            # Add content to Cognee
            await self._cognee.add(content, dataset_name=self.project_id)

            # Process and build knowledge graph
            await self._cognee.cognify()

            return {
                "status": "success",
                "source": source,
                "project": self.project_id,
                "message": "Content processed and added to knowledge graph"
            }

        except Exception as e:
            logger.error(f"❌ Cognee add failed: {e}")
            return {"error": str(e)}

    async def search(self, query: str, search_type: str = "insights") -> list[dict]:
        """Search Cognee knowledge graph.

        Args:
            query: Search query
            search_type: Type of search - 'insights', 'chunks', or 'graph_completion'
        """
        if not self.initialized:
            await self.initialize()

        if not self._cognee:
            return []

        try:
            results = await self._cognee.search(
                query_type=search_type,
                query_text=query
            )

            # Format results
            formatted = []
            for result in results:
                formatted.append({
                    "content": str(result),
                    "type": search_type,
                    "project": self.project_id
                })

            return formatted

        except Exception as e:
            logger.error(f"❌ Cognee search failed: {e}")
            return []

    async def get_graph_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        if not self.initialized:
            await self.initialize()

        try:
            # Query Neo4j for stats
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=("neo4j", NEO4J_PASSWORD)
            )

            with driver.session(database=f"cognee_{self.project_id}") as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()["count"]

                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]

                # Get node labels
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]

                # Get relationship types
                types_result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in types_result]

            driver.close()

            return {
                "project": self.project_id,
                "nodes": node_count,
                "relationships": rel_count,
                "node_labels": labels,
                "relationship_types": rel_types
            }

        except Exception as e:
            logger.error(f"❌ Failed to get graph stats: {e}")
            return {
                "project": self.project_id,
                "error": str(e)
            }

    async def get_entities(self, entity_type: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Get entities from knowledge graph."""
        if not self.initialized:
            await self.initialize()

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=("neo4j", NEO4J_PASSWORD)
            )

            with driver.session(database=f"cognee_{self.project_id}") as session:
                if entity_type:
                    query = f"MATCH (n:{entity_type}) RETURN n LIMIT $limit"
                else:
                    query = "MATCH (n) RETURN n LIMIT $limit"

                result = session.run(query, limit=limit)

                entities = []
                for record in result:
                    node = record["n"]
                    entities.append({
                        "id": node.element_id,
                        "labels": list(node.labels),
                        "properties": dict(node)
                    })

            driver.close()
            return entities

        except Exception as e:
            logger.error(f"❌ Failed to get entities: {e}")
            return []

    async def get_relationships(self, entity_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Get relationships from knowledge graph."""
        if not self.initialized:
            await self.initialize()

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=("neo4j", NEO4J_PASSWORD)
            )

            with driver.session(database=f"cognee_{self.project_id}") as session:
                if entity_id:
                    query = """
                    MATCH (a)-[r]->(b)
                    WHERE elementId(a) = $entity_id OR elementId(b) = $entity_id
                    RETURN a, r, b LIMIT $limit
                    """
                    result = session.run(query, entity_id=entity_id, limit=limit)
                else:
                    query = "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT $limit"
                    result = session.run(query, limit=limit)

                relationships = []
                for record in result:
                    relationships.append({
                        "from": {
                            "id": record["a"].element_id,
                            "labels": list(record["a"].labels),
                            "name": dict(record["a"]).get("name", "unknown")
                        },
                        "relationship": {
                            "type": record["r"].type,
                            "properties": dict(record["r"])
                        },
                        "to": {
                            "id": record["b"].element_id,
                            "labels": list(record["b"].labels),
                            "name": dict(record["b"]).get("name", "unknown")
                        }
                    })

            driver.close()
            return relationships

        except Exception as e:
            logger.error(f"❌ Failed to get relationships: {e}")
            return []

    async def clear_graph(self) -> dict:
        """Clear the entire knowledge graph for this project."""
        if not self.initialized:
            await self.initialize()

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=("neo4j", NEO4J_PASSWORD)
            )

            db_name = f"cognee_{self.project_id}"

            with driver.session(database=db_name) as session:
                result = session.run("MATCH (n) DETACH DELETE n")
                summary = result.consume()

            driver.close()

            logger.info(f"🧹 Cleared Cognee graph for project '{self.project_id}'")

            return {
                "status": "success",
                "message": f"Cleared knowledge graph for project '{self.project_id}'",
                "project": self.project_id
            }

        except Exception as e:
            logger.error(f"❌ Failed to clear graph: {e}")
            return {"status": "error", "message": str(e)}


# ============================================
# Sync Manager: Mem0 → Cognee
# ============================================
class SyncManager:
    """Manages synchronization between Mem0 and Cognee.

    This allows keeping Cognee knowledge graph in sync with Mem0 memories.
    Workflow:
    1. Get all memories from Mem0 (optionally filtered by user)
    2. Clear Cognee graph
    3. Process each memory through Cognee for entity extraction
    """

    def __init__(self, project_id: str):
        self.project_id = project_id

    async def sync_mem0_to_cognee(
        self,
        user_id: Optional[str] = None,
        clear_before_sync: bool = True
    ) -> dict:
        """Sync all Mem0 memories to Cognee knowledge graph.

        Args:
            user_id: Optional - sync only specific user's memories
            clear_before_sync: If True (default), clears Cognee before syncing

        Returns:
            Sync status with statistics
        """
        global http_client, cognee_manager

        if not COGNEE_ENABLED:
            return {"error": "Cognee is disabled"}

        if cognee_manager is None:
            return {"error": "Cognee manager not initialized"}

        if http_client is None:
            return {"error": "HTTP client not initialized"}

        try:
            stats = {
                "memories_fetched": 0,
                "memories_processed": 0,
                "memories_failed": 0,
                "entities_before": 0,
                "entities_after": 0,
                "user_filter": user_id
            }

            # Get current graph stats
            if clear_before_sync:
                graph_stats = await cognee_manager.get_graph_stats()
                stats["entities_before"] = graph_stats.get("nodes", 0)

            # Fetch memories from Mem0
            full_user_id = get_full_user_id(user_id) if user_id else None

            if full_user_id:
                # Fetch specific user's memories
                response = await http_client.get(
                    "/memories",
                    params={"user_id": full_user_id}
                )
            else:
                # Fetch all memories (will need to iterate users)
                # For now, fetch default user
                full_user_id = get_full_user_id(None)
                response = await http_client.get(
                    "/memories",
                    params={"user_id": full_user_id}
                )

            response.raise_for_status()
            memories_data = response.json()
            memories = memories_data.get("results", [])
            stats["memories_fetched"] = len(memories)

            if not memories:
                return {
                    "status": "success",
                    "message": "No memories to sync",
                    "stats": stats
                }

            # Clear Cognee if requested
            if clear_before_sync:
                clear_result = await cognee_manager.clear_graph()
                if clear_result.get("status") != "success":
                    logger.warning(f"⚠️ Failed to clear Cognee: {clear_result}")

            # Process each memory through Cognee
            for memory in memories:
                memory_content = memory.get("memory", "")
                memory_id = memory.get("id", "unknown")

                if not memory_content:
                    continue

                try:
                    # Add memory content to Cognee
                    result = await cognee_manager.add_knowledge(
                        content=f"Memory [{memory_id}]: {memory_content}",
                        source=f"mem0_sync_{user_id or 'all'}"
                    )

                    if "error" not in result:
                        stats["memories_processed"] += 1
                    else:
                        stats["memories_failed"] += 1
                        logger.warning(f"⚠️ Failed to process memory {memory_id}: {result['error']}")

                except Exception as e:
                    stats["memories_failed"] += 1
                    logger.error(f"❌ Error processing memory {memory_id}: {e}")

            # Get final graph stats
            final_stats = await cognee_manager.get_graph_stats()
            stats["entities_after"] = final_stats.get("nodes", 0)

            logger.info(f"🔄 Sync complete: {stats['memories_processed']}/{stats['memories_fetched']} memories processed")

            return {
                "status": "success",
                "message": f"Synced {stats['memories_processed']} memories to Cognee",
                "stats": stats
            }

        except Exception as e:
            logger.error(f"❌ Sync failed: {e}")
            return {"status": "error", "message": str(e)}

# Global sync manager instance
sync_manager: Optional[SyncManager] = None


# ============================================
# Knowledge Base Manager (FAISS + Ollama)
# ============================================
class KnowledgeBaseManager:
    """Manages knowledge bases with FAISS indexes and Ollama embeddings.

    Knowledge bases are SHARED across all users in a project.
    """

    def __init__(self, project_id: str, kb_root: Path, index_dir: Path):
        self.project_id = project_id
        self.kb_root = kb_root / project_id
        self.index_dir = index_dir / project_id
        self.indexes: dict[str, Any] = {}
        self.documents: dict[str, list[dict]] = {}

        # Ensure directories exist
        self.kb_root.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📚 KnowledgeBaseManager initialized for project '{project_id}'")
        logger.info(f"   KB root: {self.kb_root}")
        logger.info(f"   Index dir: {self.index_dir}")

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": OLLAMA_EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embedding"], dtype=np.float32)

    async def create_index(self, kb_name: str) -> bool:
        """Create or rebuild FAISS index for a knowledge base."""
        try:
            import faiss

            kb_path = self.kb_root / kb_name
            if not kb_path.exists():
                kb_path.mkdir(parents=True)
                logger.info(f"📁 Created knowledge base directory: {kb_path}")

            # Find all markdown files
            md_files = list(kb_path.glob("**/*.md"))
            if not md_files:
                logger.warning(f"⚠️ No markdown files found in {kb_path}")
                self.indexes[kb_name] = None
                self.documents[kb_name] = []
                return True

            # Process documents
            docs = []
            embeddings = []

            for md_file in md_files:
                content = md_file.read_text(encoding="utf-8")
                chunks = self._chunk_text(content, md_file.name)

                for i, chunk in enumerate(chunks):
                    doc = {
                        "id": f"{md_file.stem}_{i}",
                        "file": str(md_file.relative_to(kb_path)),
                        "content": chunk,
                        "chunk_index": i,
                    }
                    docs.append(doc)

                    # Get embedding
                    embedding = await self.get_embedding(chunk)
                    embeddings.append(embedding)

            if not embeddings:
                logger.warning(f"⚠️ No content to index in {kb_name}")
                self.indexes[kb_name] = None
                self.documents[kb_name] = []
                return True

            # Build FAISS index
            embeddings_matrix = np.vstack(embeddings)
            dimension = embeddings_matrix.shape[1]

            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(embeddings_matrix)  # Normalize for cosine
            index.add(embeddings_matrix)

            # Save index
            index_path = self.index_dir / f"{kb_name}.index"
            faiss.write_index(index, str(index_path))

            # Save documents metadata
            docs_path = self.index_dir / f"{kb_name}_docs.json"
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)

            self.indexes[kb_name] = index
            self.documents[kb_name] = docs

            logger.info(f"✅ Created index for '{kb_name}': {len(docs)} chunks from {len(md_files)} files")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create index for '{kb_name}': {e}")
            return False

    def _chunk_text(self, text: str, filename: str) -> list[str]:
        """Split text into chunks with overlap."""
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            if current_size + para_size > KB_CHUNK_SIZE and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text] if len(overlap_text) < KB_CHUNK_OVERLAP else []
                current_size = len(overlap_text) if current_chunk else 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text[:KB_CHUNK_SIZE]]

    async def load_index(self, kb_name: str) -> bool:
        """Load existing FAISS index from disk."""
        try:
            import faiss

            index_path = self.index_dir / f"{kb_name}.index"
            docs_path = self.index_dir / f"{kb_name}_docs.json"

            if not index_path.exists() or not docs_path.exists():
                logger.info(f"📁 Index for '{kb_name}' not found, will create on first use")
                return False

            self.indexes[kb_name] = faiss.read_index(str(index_path))
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents[kb_name] = json.load(f)

            logger.info(f"✅ Loaded index for '{kb_name}': {len(self.documents[kb_name])} chunks")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load index for '{kb_name}': {e}")
            return False

    async def search(self, kb_name: str, query: str, top_k: int = 5) -> list[dict]:
        """Search knowledge base using semantic similarity."""
        try:
            import faiss

            if kb_name not in self.indexes:
                if not await self.load_index(kb_name):
                    if not await self.create_index(kb_name):
                        return []

            index = self.indexes.get(kb_name)
            docs = self.documents.get(kb_name, [])

            if index is None or not docs:
                return []

            query_embedding = await self.get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            k = min(top_k, len(docs))
            scores, indices = index.search(query_embedding, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(docs):
                    doc = docs[idx].copy()
                    doc["score"] = float(score)
                    results.append(doc)

            return results

        except Exception as e:
            logger.error(f"❌ Search failed in '{kb_name}': {e}")
            return []

    def list_knowledge_bases(self) -> list[dict]:
        """List all available knowledge bases."""
        kbs = []
        if self.kb_root.exists():
            for kb_dir in self.kb_root.iterdir():
                if kb_dir.is_dir():
                    md_files = list(kb_dir.glob("**/*.md"))
                    index_exists = (self.index_dir / f"{kb_dir.name}.index").exists()
                    kbs.append({
                        "name": kb_dir.name,
                        "files": len(md_files),
                        "indexed": index_exists,
                        "path": str(kb_dir)
                    })
        return kbs

    async def add_document(
        self, 
        kb_name: str, 
        filename: str, 
        content: str,
        replace: bool = False
    ) -> dict:
        """Add a document to knowledge base with optional replacement.
        
        Args:
            kb_name: Name of the knowledge base
            filename: Name of the file (will add .md extension if missing)
            content: Document content (markdown)
            replace: If True, replace existing file with same name
        
        Returns:
            Status dict with document metadata
        """
        try:
            # Ensure .md extension
            if not filename.endswith('.md'):
                filename = f"{filename}.md"
            
            kb_path = self.kb_root / kb_name
            kb_path.mkdir(parents=True, exist_ok=True)
            
            file_path = kb_path / filename
            file_exists = file_path.exists()
            
            # Check if file exists and replace not requested
            if file_exists and not replace:
                return {
                    "status": "error",
                    "message": f"File '{filename}' already exists. Use replace=True to overwrite.",
                    "existing_file": filename,
                    "hint": "Set replace=True to update the document"
                }
            
            # Add metadata header with timestamp
            timestamp = datetime.now().isoformat()
            metadata_header = f"""---
filename: {filename}
knowledge_base: {kb_name}
project: {self.project_id}
created_at: {timestamp if not file_exists else 'preserved'}
updated_at: {timestamp}
version: {'updated' if file_exists and replace else 'initial'}
---

"""
            
            # Check if content already has frontmatter - preserve it but update timestamp
            if content.strip().startswith('---'):
                # Find end of frontmatter and update updated_at
                lines = content.split('\n')
                in_frontmatter = False
                new_lines = []
                updated_at_added = False
                
                for line in lines:
                    if line.strip() == '---':
                        if not in_frontmatter:
                            in_frontmatter = True
                            new_lines.append(line)
                        else:
                            if not updated_at_added:
                                new_lines.append(f"updated_at: {timestamp}")
                            new_lines.append(line)
                            in_frontmatter = False
                    elif in_frontmatter and line.startswith('updated_at:'):
                        new_lines.append(f"updated_at: {timestamp}")
                        updated_at_added = True
                    else:
                        new_lines.append(line)
                
                full_content = '\n'.join(new_lines)
            else:
                full_content = metadata_header + content
            
            # Write file
            file_path.write_text(full_content, encoding="utf-8")
            
            action = "replaced" if file_exists and replace else "created"
            logger.info(f"📝 Document {action}: {filename} in {kb_name}")
            
            # Rebuild index
            await self.create_index(kb_name)
            
            return {
                "status": "success",
                "message": f"Document {action} successfully",
                "knowledge_base": kb_name,
                "filename": filename,
                "action": action,
                "updated_at": timestamp,
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            return {"status": "error", "message": str(e)}

    async def delete_document(self, kb_name: str, filename: str) -> dict:
        """Delete a specific document from knowledge base and reindex."""
        try:
            if not filename.endswith('.md'):
                filename = f"{filename}.md"
            
            kb_path = self.kb_root / kb_name
            file_path = kb_path / filename
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File '{filename}' not found in '{kb_name}'"
                }
            
            file_path.unlink()
            logger.info(f"🗑️ Deleted document: {filename} from {kb_name}")
            
            await self.create_index(kb_name)
            
            return {
                "status": "success",
                "message": f"Deleted '{filename}' and reindexed '{kb_name}'",
                "knowledge_base": kb_name,
                "deleted_file": filename
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return {"status": "error", "message": str(e)}

    async def clear_knowledge_base(self, kb_name: str) -> dict:
        """Clear all documents from a knowledge base."""
        try:
            kb_path = self.kb_root / kb_name
            
            if not kb_path.exists():
                return {
                    "status": "error",
                    "message": f"Knowledge base '{kb_name}' not found"
                }
            
            md_files = list(kb_path.glob("**/*.md"))
            file_count = len(md_files)
            
            for md_file in md_files:
                md_file.unlink()
            
            index_path = self.index_dir / f"{kb_name}.index"
            docs_path = self.index_dir / f"{kb_name}_docs.json"
            
            if index_path.exists():
                index_path.unlink()
            if docs_path.exists():
                docs_path.unlink()
            
            if kb_name in self.indexes:
                del self.indexes[kb_name]
            if kb_name in self.documents:
                del self.documents[kb_name]
            
            logger.info(f"🧹 Cleared knowledge base '{kb_name}': {file_count} files deleted")
            
            return {
                "status": "success",
                "message": f"Cleared '{kb_name}'",
                "knowledge_base": kb_name,
                "deleted_files_count": file_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to clear knowledge base: {e}")
            return {"status": "error", "message": str(e)}

    def list_documents(self, kb_name: str) -> list:
        """List all documents in a knowledge base with metadata."""
        try:
            kb_path = self.kb_root / kb_name
            
            if not kb_path.exists():
                return []
            
            documents = []
            for md_file in kb_path.glob("**/*.md"):
                content = md_file.read_text(encoding="utf-8")
                
                metadata = {}
                if content.strip().startswith('---'):
                    lines = content.split('\n')
                    in_frontmatter = False
                    for line in lines:
                        if line.strip() == '---':
                            if not in_frontmatter:
                                in_frontmatter = True
                            else:
                                break
                        elif in_frontmatter and ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
                
                stat = md_file.stat()
                documents.append({
                    "filename": md_file.name,
                    "path": str(md_file.relative_to(kb_path)),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "metadata": metadata
                })
            
            documents.sort(key=lambda x: x["modified_at"], reverse=True)
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []

    def get_stats(self, kb_name: str) -> dict:
        """Get statistics for a knowledge base."""
        kb_path = self.kb_root / kb_name
        index_path = self.index_dir / f"{kb_name}.index"

        stats = {
            "name": kb_name,
            "project": self.project_id,
            "exists": kb_path.exists(),
            "indexed": index_path.exists(),
            "files": 0,
            "chunks": 0,
            "size_bytes": 0,
        }

        if kb_path.exists():
            md_files = list(kb_path.glob("**/*.md"))
            stats["files"] = len(md_files)
            stats["size_bytes"] = sum(f.stat().st_size for f in md_files)

        if kb_name in self.documents:
            stats["chunks"] = len(self.documents[kb_name])

        return stats


# ============================================
# Mem0 API Client
# ============================================
async def mem0_request(method: str, endpoint: str, **kwargs) -> dict:
    """Make request to Mem0 API."""
    global http_client

    if http_client is None:
        raise RuntimeError("HTTP client not initialized")

    url = f"{endpoint}"
    try:
        if method == "GET":
            response = await http_client.get(url, params=kwargs.get("params"))
        elif method == "POST":
            response = await http_client.post(url, json=kwargs.get("json"))
        elif method == "DELETE":
            response = await http_client.delete(url, params=kwargs.get("params"))
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Mem0 API error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Mem0 request failed: {e}")
        raise


# ============================================
# MCP Tools: Memory (via Mem0) - PER USER
# ============================================
@mcp.tool()
async def memory_add(content: str, user_id: Optional[str] = None, metadata: Optional[str] = None) -> str:
    """
    Add a memory to persistent storage for a specific user.

    Args:
        content: The memory content to store
        user_id: User identifier for isolation (default: 'default'). Each user has their own memory space.
        metadata: Optional JSON metadata string

    Returns:
        Confirmation message with memory ID
    """
    try:
        full_user_id = get_full_user_id(user_id)
        # Mem0 API requires messages format
        data = {
            "messages": [{"role": "user", "content": content}],
            "user_id": full_user_id
        }
        if metadata:
            data["metadata"] = json.loads(metadata)

        result = await mem0_request("POST", "/memories", json=data)
        return f"✅ Memory added for user '{user_id or DEFAULT_USER_ID}': {result.get('id', 'unknown')}"
    except Exception as e:
        return f"❌ Failed to add memory: {e}"


@mcp.tool()
async def memory_search(query: str, user_id: Optional[str] = None, limit: int = 10) -> str:
    """
    Search memories using semantic similarity for a specific user.

    Args:
        query: Search query
        user_id: User identifier (default: 'default'). Searches only this user's memories.
        limit: Maximum results (default: 10)

    Returns:
        JSON string with matching memories
    """
    try:
        full_user_id = get_full_user_id(user_id)
        # Mem0 API uses /search endpoint
        result = await mem0_request(
            "POST",
            "/search",
            json={"query": query, "user_id": full_user_id}
        )
        memories = result.get("results", [])
        return json.dumps(memories, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Search failed: {e}"


@mcp.tool()
async def memory_get_all(user_id: Optional[str] = None, limit: int = 100) -> str:
    """
    Get all memories for a specific user.

    Args:
        user_id: User identifier (default: 'default'). Gets only this user's memories.
        limit: Maximum memories to return

    Returns:
        JSON string with all memories
    """
    try:
        full_user_id = get_full_user_id(user_id)
        result = await mem0_request(
            "GET",
            "/memories",
            params={"user_id": full_user_id}
        )
        memories = result.get("results", [])
        return json.dumps(memories, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Failed to get memories: {e}"


@mcp.tool()
async def memory_delete(memory_id: str) -> str:
    """
    Delete a specific memory.

    Args:
        memory_id: ID of memory to delete

    Returns:
        Confirmation message
    """
    try:
        await mem0_request("DELETE", f"/memories/{memory_id}")
        return f"✅ Memory {memory_id} deleted"
    except Exception as e:
        return f"❌ Failed to delete: {e}"


@mcp.tool()
async def memory_get_users() -> str:
    """
    List all users who have memories in this project.

    Returns:
        JSON string with list of user IDs (without project prefix)
    """
    try:
        # This would need to be implemented in Mem0 API
        # For now, return a placeholder
        return json.dumps({
            "project": PROJECT_ID,
            "note": "User listing requires Mem0 API extension",
            "default_user": DEFAULT_USER_ID
        }, indent=2)
    except Exception as e:
        return f"❌ Failed to get users: {e}"


# ============================================
# MCP Tools: Knowledge Base - SHARED PER PROJECT
# ============================================
@mcp.tool()
async def kb_search(knowledge_base: str, query: str, limit: int = 5) -> str:
    """
    Search knowledge base using semantic similarity.
    Knowledge bases are SHARED across all users in the project.

    Args:
        knowledge_base: Name of the knowledge base to search
        query: Search query
        limit: Maximum results (default: 5)

    Returns:
        JSON string with matching document chunks
    """
    global kb_manager

    if kb_manager is None:
        return "❌ Knowledge base manager not initialized"

    try:
        results = await kb_manager.search(knowledge_base, query, limit)
        if not results:
            return f"No results found in knowledge base '{knowledge_base}'"

        return json.dumps(results, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Search failed: {e}"


@mcp.tool()
async def kb_list() -> str:
    """
    List all available knowledge bases in this project.
    Knowledge bases are SHARED across all users in the project.

    Returns:
        JSON string with knowledge base information
    """
    global kb_manager

    if kb_manager is None:
        return "❌ Knowledge base manager not initialized"

    try:
        kbs = kb_manager.list_knowledge_bases()
        if not kbs:
            return f"No knowledge bases found for project '{PROJECT_ID}'"

        return json.dumps({
            "project": PROJECT_ID,
            "knowledge_bases": kbs,
            "note": "Knowledge bases are shared across all users in this project"
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Failed to list: {e}"


@mcp.tool()
async def kb_add(
    knowledge_base: str, 
    filename: str, 
    content: str,
    replace: bool = False
) -> str:
    """
    Add a document to a knowledge base (shared across all project users).
    
    Automatically adds metadata (timestamps, version) as YAML frontmatter.
    If a file with the same name exists, use replace=True to update it.

    Args:
        knowledge_base: Name of the knowledge base (created if doesn't exist)
        filename: Name of the file (will add .md extension if missing)
        content: Document content (markdown)
        replace: If True, replace existing file with same name (default: False)

    Returns:
        JSON with document metadata including timestamps
    
    Examples:
        # Add new document
        kb_add("arch", "api-design.md", "# API Design\n...")
        
        # Update existing document  
        kb_add("arch", "api-design.md", "# Updated API\n...", replace=True)
    """
    global kb_manager

    if kb_manager is None:
        return json.dumps({"error": "Knowledge base manager not initialized"})

    result = await kb_manager.add_document(knowledge_base, filename, content, replace)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def kb_delete(knowledge_base: str, filename: str) -> str:
    """
    Delete a specific document from knowledge base.
    
    Removes the document and automatically reindexes the knowledge base.
    Use this when a document is outdated or no longer needed.
    
    Args:
        knowledge_base: Name of the knowledge base
        filename: Name of the file to delete (e.g., "architecture-v1.md")
    
    Returns:
        JSON with deletion status
    """
    global kb_manager
    
    if kb_manager is None:
        return json.dumps({"error": "Knowledge base manager not initialized"})
    
    result = await kb_manager.delete_document(knowledge_base, filename)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def kb_clear(knowledge_base: str) -> str:
    """
    Clear ALL documents from a knowledge base.
    
    ⚠️ WARNING: This permanently deletes all documents!
    Use with caution. Useful when you need to completely refresh a KB.
    
    Args:
        knowledge_base: Name of the knowledge base to clear
    
    Returns:
        JSON with status and count of deleted files
    """
    global kb_manager
    
    if kb_manager is None:
        return json.dumps({"error": "Knowledge base manager not initialized"})
    
    result = await kb_manager.clear_knowledge_base(knowledge_base)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def kb_list_documents(knowledge_base: str) -> str:
    """
    List all documents in a knowledge base with metadata.
    
    Returns information about each document including:
    - Filename and path
    - File size and modification time
    - Frontmatter metadata (if present)
    
    Documents sorted by modification time (newest first).
    
    Args:
        knowledge_base: Name of the knowledge base
    
    Returns:
        JSON array of document metadata
    """
    global kb_manager
    
    if kb_manager is None:
        return json.dumps({"error": "Knowledge base manager not initialized"})
    
    documents = kb_manager.list_documents(knowledge_base)
    return json.dumps({
        "knowledge_base": knowledge_base,
        "project": PROJECT_ID,
        "document_count": len(documents),
        "documents": documents
    }, ensure_ascii=False, indent=2)


@mcp.tool()
async def kb_reindex(knowledge_base: str) -> str:
    """
    Rebuild the FAISS index for a knowledge base.

    Args:
        knowledge_base: Name of the knowledge base to reindex

    Returns:
        Confirmation message with statistics
    """
    global kb_manager

    if kb_manager is None:
        return "❌ Knowledge base manager not initialized"

    try:
        success = await kb_manager.create_index(knowledge_base)
        if success:
            stats = kb_manager.get_stats(knowledge_base)
            return f"✅ Reindexed '{knowledge_base}': {stats['chunks']} chunks from {stats['files']} files"
        return f"❌ Reindex failed"
    except Exception as e:
        return f"❌ Reindex failed: {e}"


@mcp.tool()
async def kb_stats(knowledge_base: str) -> str:
    """
    Get statistics for a knowledge base.

    Args:
        knowledge_base: Name of the knowledge base

    Returns:
        JSON string with statistics
    """
    global kb_manager

    if kb_manager is None:
        return "❌ Knowledge base manager not initialized"

    try:
        stats = kb_manager.get_stats(knowledge_base)
        return json.dumps(stats, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Failed to get stats: {e}"


# ============================================
# MCP Tools: Cognee (Knowledge Graph) - SHARED PER PROJECT
# ============================================
@mcp.tool()
async def cognee_add(content: str, source: str = "manual") -> str:
    """
    Add content to Cognee knowledge graph for entity extraction and relationship detection.
    Knowledge graph is SHARED across all users in the project.

    This will:
    1. Extract entities (people, organizations, concepts, etc.)
    2. Detect relationships between entities
    3. Store in Neo4j knowledge graph

    Args:
        content: Text content to process
        source: Source identifier for tracking (default: 'manual')

    Returns:
        Confirmation message with processing status
    """
    global cognee_manager

    if not COGNEE_ENABLED:
        return "❌ Cognee is disabled. Set COGNEE_ENABLED=true to enable."

    if cognee_manager is None:
        return "❌ Cognee manager not initialized"

    try:
        result = await cognee_manager.add_knowledge(content, source)
        if "error" in result:
            return f"❌ Cognee add failed: {result['error']}"
        return f"✅ Content processed by Cognee: {result['message']}"
    except Exception as e:
        return f"❌ Cognee add failed: {e}"


@mcp.tool()
async def cognee_search(query: str, search_type: str = "insights") -> str:
    """
    Search Cognee knowledge graph using various search strategies.

    Args:
        query: Search query
        search_type: Type of search:
            - 'insights': Get high-level insights (default)
            - 'chunks': Get relevant text chunks
            - 'graph_completion': Complete graph patterns

    Returns:
        JSON string with search results
    """
    global cognee_manager

    if not COGNEE_ENABLED:
        return "❌ Cognee is disabled. Set COGNEE_ENABLED=true to enable."

    if cognee_manager is None:
        return "❌ Cognee manager not initialized"

    try:
        results = await cognee_manager.search(query, search_type)
        if not results:
            return f"No results found for query: {query}"
        return json.dumps(results, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Cognee search failed: {e}"


@mcp.tool()
async def cognee_graph_stats() -> str:
    """
    Get statistics about the Cognee knowledge graph.

    Returns:
        JSON string with graph statistics (nodes, relationships, types)
    """
    global cognee_manager

    if not COGNEE_ENABLED:
        return "❌ Cognee is disabled. Set COGNEE_ENABLED=true to enable."

    if cognee_manager is None:
        return "❌ Cognee manager not initialized"

    try:
        stats = await cognee_manager.get_graph_stats()
        return json.dumps(stats, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Failed to get graph stats: {e}"


@mcp.tool()
async def cognee_get_entities(entity_type: Optional[str] = None, limit: int = 50) -> str:
    """
    Get entities from the knowledge graph.

    Args:
        entity_type: Filter by entity type (e.g., 'Person', 'Organization', 'Concept')
        limit: Maximum entities to return (default: 50)

    Returns:
        JSON string with entities
    """
    global cognee_manager

    if not COGNEE_ENABLED:
        return "❌ Cognee is disabled. Set COGNEE_ENABLED=true to enable."

    if cognee_manager is None:
        return "❌ Cognee manager not initialized"

    try:
        entities = await cognee_manager.get_entities(entity_type, limit)
        if not entities:
            return "No entities found in knowledge graph"
        return json.dumps(entities, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Failed to get entities: {e}"


@mcp.tool()
async def cognee_get_relationships(entity_id: Optional[str] = None, limit: int = 50) -> str:
    """
    Get relationships from the knowledge graph.

    Args:
        entity_id: Filter by specific entity (get relationships for this entity)
        limit: Maximum relationships to return (default: 50)

    Returns:
        JSON string with relationships (from -> type -> to)
    """
    global cognee_manager

    if not COGNEE_ENABLED:
        return "❌ Cognee is disabled. Set COGNEE_ENABLED=true to enable."

    if cognee_manager is None:
        return "❌ Cognee manager not initialized"

    try:
        relationships = await cognee_manager.get_relationships(entity_id, limit)
        if not relationships:
            return "No relationships found in knowledge graph"
        return json.dumps(relationships, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Failed to get relationships: {e}"


@mcp.tool()
async def cognee_clear() -> str:
    """
    Clear the entire Cognee knowledge graph for this project.

    ⚠️ WARNING: This permanently deletes ALL entities and relationships!

    Use this when:
    - Architecture changed significantly
    - Need to rebuild graph from scratch
    - Before running sync_mem0_to_cognee

    Returns:
        JSON with status and confirmation
    """
    global cognee_manager

    if not COGNEE_ENABLED:
        return json.dumps({"error": "Cognee is disabled. Set COGNEE_ENABLED=true to enable."})

    if cognee_manager is None:
        return json.dumps({"error": "Cognee manager not initialized"})

    result = await cognee_manager.clear_graph()
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def sync_mem0_to_cognee(user_id: Optional[str] = None, clear_before_sync: bool = True) -> str:
    """
    Synchronize Mem0 memories to Cognee knowledge graph.

    This tool:
    1. Fetches all memories from Mem0 (optionally filtered by user)
    2. Clears Cognee graph (if clear_before_sync=True)
    3. Processes each memory through Cognee for entity extraction
    4. Builds relationships between extracted entities

    Use cases:
    - Keep Cognee graph in sync with Mem0 facts
    - Regular maintenance to ensure data consistency
    - After significant changes to memories

    Args:
        user_id: Optional - sync only specific user's memories (default: all users)
        clear_before_sync: If True (default), clears Cognee before syncing

    Returns:
        JSON with sync statistics:
        - memories_fetched: total memories found
        - memories_processed: successfully processed
        - memories_failed: failed to process
        - entities_before/after: graph size comparison

    Example workflow:
        # Weekly sync to keep Cognee fresh
        sync_mem0_to_cognee(clear_before_sync=True)

        # Incremental add without clearing
        sync_mem0_to_cognee(user_id="dev1", clear_before_sync=False)
    """
    global sync_manager

    if not COGNEE_ENABLED:
        return json.dumps({"error": "Cognee is disabled. Set COGNEE_ENABLED=true to enable."})

    if sync_manager is None:
        return json.dumps({"error": "Sync manager not initialized"})

    result = await sync_manager.sync_mem0_to_cognee(user_id, clear_before_sync)
    return json.dumps(result, indent=2, ensure_ascii=False)


# ============================================
# MCP Tools: Info
# ============================================
@mcp.tool()
async def get_project_info() -> str:
    """
    Get information about the current project and isolation model.

    Returns:
        JSON string with project configuration
    """
    return json.dumps({
        "project_id": PROJECT_ID,
        "isolation_model": {
            "memory": "Per user - each user has private memories",
            "knowledge_base": "Shared - all users in project share the same KB",
            "cognee": "Shared - all users in project share the same knowledge graph"
        },
        "services": {
            "mem0": MEM0_API_URL,
            "ollama": OLLAMA_BASE_URL,
            "cognee_enabled": COGNEE_ENABLED,
            "neo4j": NEO4J_URI if COGNEE_ENABLED else "disabled"
        },
        "default_user_id": DEFAULT_USER_ID,
        "usage": {
            "memory_tools": "Pass user_id parameter to isolate memories per user",
            "kb_tools": "Shared across all users - no user_id needed",
            "cognee_tools": "Shared across all users - knowledge graph for entity relationships"
        }
    }, indent=2, ensure_ascii=False)


# ============================================
# Starlette Application
# ============================================
def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create Starlette app with SSE and HTTP Stream transports."""

    sse = SseServerTransport("/messages/")

    class SSEApp:
        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                logger.info("📡 New SSE connection")
                async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                    await mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp_server.create_initialization_options(),
                    )

    sse_app = SSEApp()

    class HTTPStreamApp:
        def __init__(self, streamable_app):
            self.streamable_app = streamable_app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                logger.info(f"🌐 HTTP Stream: {scope['method']} {scope['path']}")
            await self.streamable_app(scope, receive, send)

    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "healthy",
            "service": f"unified-{PROJECT_ID}",
            "project_id": PROJECT_ID,
            "mem0_url": MEM0_API_URL,
            "cognee_enabled": COGNEE_ENABLED,
            "isolation": {
                "memory": "per_user",
                "knowledge_base": "shared_per_project",
                "cognee": "shared_per_project" if COGNEE_ENABLED else "disabled"
            }
        })

    @asynccontextmanager
    async def lifespan(app):
        global http_client, kb_manager, cognee_manager, sync_manager

        logger.info(f"🚀 Starting Unified MCP for project '{PROJECT_ID}'")

        http_client = httpx.AsyncClient(base_url=MEM0_API_URL, timeout=60.0)
        logger.info(f"✅ HTTP client initialized: {MEM0_API_URL}")

        kb_manager = KnowledgeBaseManager(PROJECT_ID, KNOWLEDGE_BASES_ROOT, FAISS_INDEX_DIR)
        logger.info("✅ Knowledge Base manager initialized")

        for kb in kb_manager.list_knowledge_bases():
            await kb_manager.load_index(kb["name"])

        # Initialize Cognee if enabled
        if COGNEE_ENABLED:
            cognee_manager = CogneeManager(PROJECT_ID)
            await cognee_manager.initialize()
            logger.info("✅ Cognee manager initialized")

            # Initialize Sync Manager
            sync_manager = SyncManager(PROJECT_ID)
            logger.info("✅ Sync manager initialized")
        else:
            logger.info("ℹ️ Cognee is disabled")

        logger.info("Starting FastMCP session manager...")
        async with mcp.session_manager.run():
            logger.info("✅ FastMCP session manager started")
            yield
            logger.info("Shutting down...")

        await http_client.aclose()
        logger.info("✅ Cleanup complete")

    mcp.settings.streamable_http_path = "/"
    http_stream_base_app = mcp.streamable_http_app()
    http_stream_app = HTTPStreamApp(http_stream_base_app)

    return Starlette(
        debug=debug,
        lifespan=lifespan,
        routes=[
            Route("/", endpoint=health_check),
            Mount("/sse", app=sse_app),
            Mount("/messages", app=sse.handle_post_message),
            Mount("/mcp", app=http_stream_app),
        ],
    )


# ============================================
# Main Entry Point
# ============================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MCP_PORT", "8080"))
    host = os.getenv("MCP_HOST", "0.0.0.0")

    logger.info(f"🚀 Starting Unified MCP server for project '{PROJECT_ID}' on {host}:{port}")
    logger.info(f"   Mem0: {MEM0_API_URL}")
    logger.info(f"   KB: {KNOWLEDGE_BASES_ROOT}/{PROJECT_ID}")
    logger.info(f"   Cognee: {'enabled' if COGNEE_ENABLED else 'disabled'}")
    logger.info(f"   Isolation: Memory=per_user, KB=shared, Cognee=shared")

    mcp_server = mcp._mcp_server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=host, port=port)
