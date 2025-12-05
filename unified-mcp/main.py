"""
Unified MCP Server for Mem0 + Knowledge Base + Graphiti
"""

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict

import httpx
import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("CURRENT_PROJECT_ID", "default")
MEM0_API_URL = os.getenv("MEM0_API_URL", "http://localhost:8000")
KNOWLEDGE_BASES_ROOT = Path(os.getenv("KNOWLEDGE_BASES_ROOT", "/data/knowledge_bases"))
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "/data/faiss_indexes"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
KB_CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "1000"))
KB_CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "200"))
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
GRAPHITI_ENABLED = os.getenv("GRAPHITI_ENABLED", "false").lower() == "true"
GRAPHITI_API_URL = os.getenv("GRAPHITI_API_URL", "http://graphiti:8000")

# Global instances
http_client: Optional[httpx.AsyncClient] = None
kb_manager: Optional["KnowledgeBaseManager"] = None
graphiti_client: Optional["GraphitiClient"] = None

# Initialize FastMCP
mcp = FastMCP(f"unified-{PROJECT_ID}")


def get_full_user_id(user_id: Optional[str] = None) -> str:
    uid = user_id or DEFAULT_USER_ID
    return f"{PROJECT_ID}_{uid}"


class GraphitiClient:
    """Client for Graphiti MCP Server with project isolation via group_id"""
    
    def __init__(self, api_url: str, project_id: str):
        self.api_url = api_url.rstrip("/")
        self.project_id = project_id
        self.client: Optional[httpx.AsyncClient] = None
        self.initialized = False
    
    async def initialize(self):
        try:
            self.client = httpx.AsyncClient(timeout=60.0)
            response = await self.client.get(f"{self.api_url}/health")
            self.initialized = response.status_code == 200
            if self.initialized:
                logger.info(f"✅ Graphiti client initialized (group_id={self.project_id})")
        except Exception as e:
            logger.error(f"❌ Graphiti connection failed: {e}")
    
    async def close(self):
        if self.client:
            await self.client.aclose()
    
    async def _call_mcp(self, method: str, params: Dict = None) -> Dict:
        if not self.initialized:
            return {"error": "Graphiti not initialized"}
        try:
            payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
            response = await self.client.post(f"{self.api_url}/mcp/", json=payload)
            return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def add_episode(self, content: str, name: str = None, source: str = "user") -> Dict:
        params = {
            "content": content, 
            "source": source, 
            "reference_time": datetime.now().isoformat(),
            "group_id": self.project_id
        }
        if name:
            params["name"] = name
        return await self._call_mcp("tools/add_episode", params)
    
    async def search(self, query: str, num_results: int = 10) -> Dict:
        return await self._call_mcp("tools/search", {
            "query": query, 
            "num_results": num_results,
            "group_ids": [self.project_id]
        })
    
    async def get_episodes(self, last_n: int = 10) -> Dict:
        return await self._call_mcp("tools/get_episodes", {
            "last_n": last_n,
            "group_ids": [self.project_id]
        })


class KnowledgeBaseManager:
    def __init__(self, root_dir: Path, index_dir: Path, project_id: str):
        self.root_dir = root_dir / project_id
        self.index_dir = index_dir / project_id
        self.project_id = project_id
        self.indexes: Dict[str, Any] = {}
        self.documents: Dict[str, List[Dict]] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    async def initialize(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        await self._load_existing_indexes()
        logger.info(f"✅ KB initialized: {len(self.indexes)} indexes")
    
    async def _load_existing_indexes(self):
        try:
            import faiss
            for index_file in self.index_dir.glob("*.index"):
                kb_name = index_file.stem
                try:
                    self.indexes[kb_name] = faiss.read_index(str(index_file))
                    docs_file = self.index_dir / f"{kb_name}.json"
                    if docs_file.exists():
                        with open(docs_file) as f:
                            self.documents[kb_name] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load {kb_name}: {e}")
        except ImportError:
            logger.warning("FAISS not available")
    
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
    
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + KB_CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - KB_CHUNK_OVERLAP
        return chunks
    
    async def create_kb(self, name: str, documents: List[str]) -> Dict:
        import faiss
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc)
            all_chunks.extend([{"text": c, "source": f"doc_{i}"} for i, c in enumerate(chunks)])
        
        if not all_chunks:
            return {"error": "No chunks created"}
        
        embeddings = []
        for chunk in all_chunks:
            emb = await self._get_embedding(chunk["text"])
            embeddings.append(emb)
        
        embeddings_np = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        
        self.indexes[name] = index
        self.documents[name] = all_chunks
        
        faiss.write_index(index, str(self.index_dir / f"{name}.index"))
        with open(self.index_dir / f"{name}.json", "w") as f:
            json.dump(all_chunks, f)
        
        return {"status": "created", "name": name, "chunks": len(all_chunks)}
    
    async def search_kb(self, name: str, query: str, top_k: int = 5) -> List[Dict]:
        if name not in self.indexes:
            return [{"error": f"KB {name} not found"}]
        
        query_emb = await self._get_embedding(query)
        D, I = self.indexes[name].search(query_emb.reshape(1, -1), top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.documents[name]):
                results.append({
                    "rank": i + 1,
                    "text": self.documents[name][idx]["text"],
                    "score": float(1 / (1 + dist))
                })
        return results
    
    async def list_kbs(self) -> List[str]:
        return list(self.indexes.keys())


# MEM0 TOOLS
@mcp.tool()
async def mem0_add(content: str, user_id: str = None, metadata: str = None) -> str:
    """Add memory to Mem0."""
    full_user_id = get_full_user_id(user_id)
    payload = {"messages": [{"role": "user", "content": content}], "user_id": full_user_id}
    if metadata:
        try:
            payload["metadata"] = json.loads(metadata)
        except:
            pass
    try:
        response = await http_client.post(f"{MEM0_API_URL}/v1/memories/", json=payload)
        return json.dumps(response.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def mem0_search(query: str, user_id: str = None, limit: int = 10) -> str:
    """Search memories in Mem0."""
    full_user_id = get_full_user_id(user_id)
    try:
        response = await http_client.post(
            f"{MEM0_API_URL}/v1/memories/search/",
            json={"query": query, "user_id": full_user_id, "limit": limit}
        )
        return json.dumps(response.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def mem0_get_all(user_id: str = None) -> str:
    """Get all memories for user."""
    full_user_id = get_full_user_id(user_id)
    try:
        response = await http_client.get(f"{MEM0_API_URL}/v1/memories/", params={"user_id": full_user_id})
        return json.dumps(response.json(), ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# KB TOOLS
@mcp.tool()
async def kb_create(name: str, documents: str) -> str:
    """Create knowledge base from documents (JSON array of strings)."""
    try:
        docs = json.loads(documents)
        result = await kb_manager.create_kb(name, docs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kb_search(name: str, query: str, top_k: int = 5) -> str:
    """Search knowledge base."""
    results = await kb_manager.search_kb(name, query, top_k)
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
async def kb_list() -> str:
    """List all knowledge bases."""
    return json.dumps(await kb_manager.list_kbs(), ensure_ascii=False)


# GRAPHITI TOOLS
@mcp.tool()
async def graphiti_add_episode(content: str, name: str = None, source: str = "user") -> str:
    """Add an episode to Graphiti temporal knowledge graph."""
    if not graphiti_client or not graphiti_client.initialized:
        return json.dumps({"error": "Graphiti not available"})
    result = await graphiti_client.add_episode(content, name, source)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def graphiti_search(query: str, num_results: int = 10) -> str:
    """Search the Graphiti knowledge graph."""
    if not graphiti_client or not graphiti_client.initialized:
        return json.dumps({"error": "Graphiti not available"})
    result = await graphiti_client.search(query, num_results)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def graphiti_get_episodes(last_n: int = 10) -> str:
    """Get recent episodes from Graphiti."""
    if not graphiti_client or not graphiti_client.initialized:
        return json.dumps({"error": "Graphiti not available"})
    result = await graphiti_client.get_episodes(last_n)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_project_info() -> str:
    """Get project configuration info."""
    return json.dumps({
        "project_id": PROJECT_ID,
        "features": {
            "mem0": True, 
            "kb": True, 
            "graphiti": GRAPHITI_ENABLED and graphiti_client and graphiti_client.initialized
        },
        "graphiti_group_id": PROJECT_ID if GRAPHITI_ENABLED else None
    }, ensure_ascii=False)


# MAIN
async def startup():
    global http_client, kb_manager, graphiti_client
    logger.info(f"🚀 Starting Unified MCP Server for \"{PROJECT_ID}\"")
    http_client = httpx.AsyncClient(timeout=30.0)
    kb_manager = KnowledgeBaseManager(KNOWLEDGE_BASES_ROOT, FAISS_INDEX_DIR, PROJECT_ID)
    await kb_manager.initialize()
    if GRAPHITI_ENABLED:
        graphiti_client = GraphitiClient(GRAPHITI_API_URL, PROJECT_ID)
        await graphiti_client.initialize()


async def shutdown():
    global http_client, graphiti_client
    if http_client:
        await http_client.aclose()
    if graphiti_client:
        await graphiti_client.close()
    logger.info("Server stopped")


if __name__ == "__main__":
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    async def health(request):
        return JSONResponse({"status": "healthy", "project": PROJECT_ID})
    
    @asynccontextmanager
    async def lifespan(app):
        await startup()
        yield
        await shutdown()
    
    # Get SSE app from FastMCP - it has /sse and /messages routes
    sse_app = mcp.sse_app()
    
    # Add health route to sse_app
    sse_app.add_route("/", health, methods=["GET"])
    sse_app.add_route("/health", health, methods=["GET"])
    
    # Run sse_app directly - no nested mounting
    uvicorn.run(sse_app, host="0.0.0.0", port=MCP_PORT, lifespan="on")
