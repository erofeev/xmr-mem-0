"""
Unified MCP Server v3.0
Cipher-Centric Architecture

This server:
1. Proxies memory requests to Cipher via MCP
2. Adds KB (FAISS) functionality
3. Syncs to Graphiti for temporal queries
4. Adds project/user context to all requests
"""
import os
import json
import hashlib
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import httpx
import numpy as np
from dotenv import load_dotenv
from fastmcp import FastMCP
from contextlib import asynccontextmanager

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID", "default")
CIPHER_URL = os.getenv("CIPHER_URL", "http://cipher:3000")
GRAPHITI_URL = os.getenv("GRAPHITI_URL", "http://graphiti:8000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
KNOWLEDGE_BASES_ROOT = os.getenv("KNOWLEDGE_BASES_ROOT", "/data/knowledge_bases")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "/data/faiss_indexes")

# Global
kb_manager: Optional["KnowledgeBaseManager"] = None

# Initialize FastMCP
mcp = FastMCP(f"unified-{PROJECT_ID}")


def get_full_user_id(user_id: Optional[str] = None) -> str:
    """Create project-scoped user_id for data isolation"""
    base_id = user_id or "default"
    return f"{PROJECT_ID}_{base_id}"


# ==================== CIPHER PROXY ====================
# Proxy requests to Cipher with project/user context

async def call_cipher_mcp(method: str, params: dict) -> dict:
    """Call Cipher MCP tool via JSON-RPC over HTTP"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Cipher's MCP endpoint
            response = await client.post(
                f"{CIPHER_URL}/api/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": method,
                        "arguments": params
                    }
                },
                headers={"Accept": "application/json"}
            )
            return response.json()
    except Exception as e:
        logger.error(f"Cipher MCP call error: {e}")
        return {"error": str(e)}


async def sync_to_graphiti(content: str, user_id: str, content_type: str, module: Optional[str] = None):
    """Sync important memory to Graphiti for temporal queries"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{GRAPHITI_URL}/episodes",
                json={
                    "content": content,
                    "name": f"{content_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "source": user_id,
                    "group_id": PROJECT_ID,
                    "metadata": {
                        "type": content_type,
                        "module": module,
                        "project_id": PROJECT_ID,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
            logger.debug(f"Synced to Graphiti: {content_type}")
    except Exception as e:
        logger.warning(f"Graphiti sync failed: {e}")


# ==================== MEMORY TOOLS (via Cipher) ====================

@mcp.tool()
async def memory_search(query: str, user_id: Optional[str] = None, limit: int = 10) -> str:
    """
    Search memories. Finds relevant facts, decisions, and context from past sessions.
    Uses Cipher's semantic search with deduplication.
    """
    full_user_id = get_full_user_id(user_id)
    result = await call_cipher_mcp("cipher_memory_search", {
        "query": query,
        "userId": full_user_id,
        "limit": limit
    })
    return json.dumps(result)


@mcp.tool()
async def memory_add(
    content: str,
    user_id: Optional[str] = None,
    memory_type: str = "fact",
    module: Optional[str] = None,
    tags: Optional[str] = None
) -> str:
    """
    Add memory with automatic deduplication (5-layer Cipher approach).
    
    memory_type:
    - 'fact': Quick facts, preferences (System 1)
    - 'reasoning': Complex decisions, problem-solving (System 2)
    - 'decision': Architectural/technical decisions
    
    Returns: ADD (new), UPDATE (merged), or NONE (duplicate)
    """
    full_user_id = get_full_user_id(user_id)
    
    result = await call_cipher_mcp("cipher_extract_and_operate_memory", {
        "content": content,
        "userId": full_user_id,
        "projectId": PROJECT_ID,
        "memoryType": memory_type,
        "module": module,
        "tags": tags.split(",") if tags else []
    })
    
    # Sync decisions and reasoning to Graphiti for timeline queries
    if memory_type in ["reasoning", "decision"] or "decision" in content.lower():
        await sync_to_graphiti(content, full_user_id, memory_type, module)
    
    return json.dumps(result)


@mcp.tool()
async def memory_store_reasoning(
    content: str,
    user_id: Optional[str] = None,
    module: Optional[str] = None
) -> str:
    """
    Store reasoning trace (System 2 memory).
    For complex problem-solving, debugging sessions, architectural decisions.
    Automatically synced to Graphiti for timeline queries.
    """
    full_user_id = get_full_user_id(user_id)
    
    result = await call_cipher_mcp("cipher_store_reasoning_memory", {
        "content": content,
        "userId": full_user_id,
        "projectId": PROJECT_ID,
        "module": module
    })
    
    # Always sync reasoning to Graphiti
    await sync_to_graphiti(content, full_user_id, "reasoning", module)
    
    return json.dumps(result)


# ==================== WORKSPACE TOOLS (Team Context via Cipher) ====================

@mcp.tool()
async def workspace_search(query: str, limit: int = 10) -> str:
    """
    Search workspace memory - who is working on what in the team.
    Project-scoped, shared across team members.
    """
    result = await call_cipher_mcp("cipher_workspace_search", {
        "query": query,
        "projectId": PROJECT_ID,
        "limit": limit
    })
    return json.dumps(result)


@mcp.tool()
async def workspace_store(
    content: str,
    team_member: Optional[str] = None,
    activity_type: Optional[str] = None,
    module: Optional[str] = None
) -> str:
    """
    Store workspace context - who is working on what.
    Examples: 'Alice is implementing auth module', 'Fixed bug in payment service'
    """
    result = await call_cipher_mcp("cipher_workspace_store", {
        "content": content,
        "projectId": PROJECT_ID,
        "teamMember": team_member,
        "activityType": activity_type,
        "module": module
    })
    
    # Sync to Graphiti for timeline
    if team_member and activity_type:
        activity_content = f"{team_member}: {activity_type}"
        if module:
            activity_content += f" on {module}"
        activity_content += f" - {content}"
        await sync_to_graphiti(activity_content, team_member, "activity", module)
    
    return json.dumps(result)


# ==================== GRAPHITI TOOLS (Timeline Queries) ====================

@mcp.tool()
async def timeline_search(query: str, days: int = 14, limit: int = 20) -> str:
    """
    Search what happened in the project over time.
    Examples: 
    - 'decisions about authentication'
    - 'changes in last 2 weeks'
    - 'who worked on payments'
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search facts
            facts_resp = await client.post(
                f"{GRAPHITI_URL}/search/facts",
                json={"query": query, "group_id": PROJECT_ID, "num_results": limit}
            )
            # Search nodes
            nodes_resp = await client.post(
                f"{GRAPHITI_URL}/search/nodes",
                json={"query": query, "group_id": PROJECT_ID, "num_results": limit}
            )
            
            return json.dumps({
                "facts": facts_resp.json() if facts_resp.status_code == 200 else [],
                "entities": nodes_resp.json() if nodes_resp.status_code == 200 else [],
                "query": query,
                "project": PROJECT_ID
            })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def add_decision(
    decision: str,
    reasoning: Optional[str] = None,
    module: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[str] = None
) -> str:
    """
    Add architectural/technical decision with reasoning.
    Stored in both Cipher (for search) and Graphiti (for timeline).
    
    Example: add_decision(
        decision="Use PostgreSQL RLS for tenant isolation",
        reasoning="Safer than app-level filtering, enforced at DB level",
        module="security"
    )
    """
    full_user_id = get_full_user_id(user_id)
    
    content = f"DECISION: {decision}"
    if reasoning:
        content += f"\nREASONING: {reasoning}"
    if module:
        content += f"\nMODULE: {module}"
    
    # Store in Cipher
    cipher_result = await call_cipher_mcp("cipher_extract_and_operate_memory", {
        "content": content,
        "userId": full_user_id,
        "projectId": PROJECT_ID,
        "memoryType": "decision",
        "module": module,
        "tags": tags.split(",") if tags else []
    })
    
    # Store in Graphiti for timeline
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{GRAPHITI_URL}/episodes",
                json={
                    "content": content,
                    "name": f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "source": full_user_id,
                    "group_id": PROJECT_ID,
                    "metadata": {
                        "type": "decision",
                        "module": module,
                        "tags": tags.split(",") if tags else []
                    }
                }
            )
    except Exception as e:
        logger.warning(f"Graphiti sync failed: {e}")
    
    return json.dumps({"status": "added", "decision": decision, "cipher": cipher_result})


@mcp.tool()
async def who_worked_on(module: str, days: int = 14) -> str:
    """Find who worked on a specific module in the last N days."""
    query = f"activity work development {module}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GRAPHITI_URL}/search/facts",
                json={"query": query, "group_id": PROJECT_ID, "num_results": 30}
            )
            return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def module_history(module: str, limit: int = 20) -> str:
    """Get history of decisions and changes for a specific module."""
    query = f"decision change implementation {module}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GRAPHITI_URL}/search/facts",
                json={"query": query, "group_id": PROJECT_ID, "num_results": limit}
            )
            return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def recent_activity(last_n: int = 20) -> str:
    """Get the most recent activities and decisions in the project."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{GRAPHITI_URL}/episodes",
                params={"group_id": PROJECT_ID, "last_n": last_n}
            )
            return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


# ==================== KNOWLEDGE BASE TOOLS ====================

class KnowledgeBaseManager:
    def __init__(self, kb_root: str, index_dir: str, project_id: str):
        self.kb_root = kb_root
        self.index_dir = index_dir
        self.project_id = project_id
        self.indexes = {}
        self.embeddings_cache = {}
    
    async def initialize(self):
        import faiss
        os.makedirs(self.kb_root, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        for name in os.listdir(self.index_dir):
            if name.endswith(".index"):
                kb_name = name.replace(".index", "")
                try:
                    self.indexes[kb_name] = faiss.read_index(
                        os.path.join(self.index_dir, name)
                    )
                    logger.info(f"Loaded KB: {kb_name}")
                except Exception as e:
                    logger.error(f"Failed to load {kb_name}: {e}")
        
        logger.info(f"KB initialized: {len(self.indexes)} indexes")
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
                    timeout=30.0
                )
                embedding = np.array(response.json()["embedding"], dtype=np.float32)
                self.embeddings_cache[cache_key] = embedding
                return embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(768, dtype=np.float32)
    
    async def create(self, name: str, documents: List[str]) -> Dict:
        import faiss
        
        chunks = []
        for doc in documents:
            for i in range(0, len(doc), 800):
                chunk = doc[i:i+1000]
                if chunk.strip():
                    chunks.append({"text": chunk, "source": f"doc_{len(chunks)}"})
        
        if not chunks:
            return {"error": "No valid chunks"}
        
        embeddings = [await self._get_embedding(c["text"]) for c in chunks]
        matrix = np.vstack(embeddings)
        
        index = faiss.IndexFlatIP(matrix.shape[1])
        faiss.normalize_L2(matrix)
        index.add(matrix)
        
        faiss.write_index(index, os.path.join(self.index_dir, f"{name}.index"))
        with open(os.path.join(self.kb_root, f"{name}_chunks.json"), "w") as f:
            json.dump(chunks, f)
        
        self.indexes[name] = index
        return {"status": "created", "name": name, "chunks": len(chunks)}
    
    async def search(self, name: str, query: str, top_k: int = 5) -> List[Dict]:
        import faiss
        
        if name not in self.indexes:
            return [{"error": f"KB '{name}' not found"}]
        
        qe = await self._get_embedding(query)
        qe = qe.reshape(1, -1)
        faiss.normalize_L2(qe)
        
        scores, indices = self.indexes[name].search(qe, top_k)
        
        with open(os.path.join(self.kb_root, f"{name}_chunks.json")) as f:
            chunks = json.load(f)
        
        return [{
            "text": chunks[idx]["text"],
            "score": float(score),
            "source": chunks[idx].get("source")
        } for score, idx in zip(scores[0], indices[0]) if idx < len(chunks)]


@mcp.tool()
async def kb_create(name: str, documents: str) -> str:
    """Create knowledge base from documents (JSON array of strings)."""
    try:
        docs = json.loads(documents)
        result = await kb_manager.create(name, docs)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kb_search(name: str, query: str, top_k: int = 5) -> str:
    """Search knowledge base for relevant documentation."""
    try:
        results = await kb_manager.search(name, query, top_k)
        return json.dumps({"results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kb_list() -> str:
    """List all knowledge bases."""
    return json.dumps({"knowledge_bases": list(kb_manager.indexes.keys())})


# ==================== PROJECT INFO ====================

@mcp.tool()
async def get_project_info() -> str:
    """Get project configuration and available tools."""
    return json.dumps({
        "project_id": PROJECT_ID,
        "version": "3.0.0",
        "architecture": "Cipher-Centric",
        "tools": {
            "memory": ["memory_search", "memory_add", "memory_store_reasoning"],
            "workspace": ["workspace_search", "workspace_store"],
            "timeline": ["timeline_search", "add_decision", "who_worked_on", 
                         "module_history", "recent_activity"],
            "kb": ["kb_create", "kb_search", "kb_list"]
        },
        "components": {
            "cipher": CIPHER_URL,
            "graphiti": GRAPHITI_URL,
            "ollama": OLLAMA_BASE_URL
        }
    })


# ==================== STARTUP ====================

_initialized = False

async def ensure_initialized():
    global _initialized, kb_manager
    if _initialized:
        return
    _initialized = True
    
    logger.info(f"Starting Unified MCP Server v3.0 for {PROJECT_ID}")
    kb_manager = KnowledgeBaseManager(KNOWLEDGE_BASES_ROOT, FAISS_INDEX_DIR, PROJECT_ID)
    await kb_manager.initialize()
    
    # Check components
    cipher_ok = graphiti_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{CIPHER_URL}/health")
            cipher_ok = r.status_code == 200
    except: pass
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{GRAPHITI_URL}/health")
            graphiti_ok = r.status_code == 200
    except: pass
    
    logger.info(f"Server ready: cipher={cipher_ok}, graphiti={graphiti_ok}, kb=True")

# Keep old function signature for compatibility
async def startup():
    global kb_manager
    
    kb_manager = KnowledgeBaseManager(KNOWLEDGE_BASES_ROOT, FAISS_INDEX_DIR, PROJECT_ID)
    await kb_manager.initialize()
    
    # Check components
    cipher_ok = graphiti_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{CIPHER_URL}/health")
            cipher_ok = r.status_code == 200
    except: pass
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{GRAPHITI_URL}/health")
            graphiti_ok = r.status_code == 200
    except: pass
    
    logger.info(f"Server ready: cipher={cipher_ok}, graphiti={graphiti_ok}, kb=True")


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    await ensure_initialized()
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "healthy", "project": PROJECT_ID, "version": "3.0.0", "initialized": _initialized})




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:mcp.http_app", host="0.0.0.0", port=8080, factory=True)
