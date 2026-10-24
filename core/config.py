"""Central configuration. Override via environment variables where noted."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── LLM / agent ──────────────────────────────────────────────
MODEL = os.environ.get("RAG_MODEL", "openai/gpt-oss-120b")
MAX_RETRIES = 2
TOP_K = 4

# ── Embeddings / chunking ────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── ChromaDB (persistent) ────────────────────────────────────
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_PREFIX = "doc_"
ADD_BATCH_SIZE = 500  # chunks embedded + written per batch


def get_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")
