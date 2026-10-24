"""Persistent ChromaDB store.

One Chroma collection per uploaded document. The collection name is derived
from the SHA-256 of the PDF bytes, so uploading the same file again reuses the
vectors already on disk instead of re-embedding.

Document info (filename, pages, chunks) is stored in the collection metadata,
which lets the UI list previously ingested documents after a restart.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from core.config import ADD_BATCH_SIZE, CHROMA_PERSIST_DIR, COLLECTION_PREFIX


def compute_doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


class ChromaStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, embeddings=None):
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embeddings = embeddings

    # ── helpers ──────────────────────────────────────────────
    @property
    def embeddings(self):
        if self._embeddings is None:
            from core.embeddings import get_embeddings

            self._embeddings = get_embeddings()
        return self._embeddings

    @staticmethod
    def collection_name(doc_id: str) -> str:
        return f"{COLLECTION_PREFIX}{doc_id[:32]}"

    def _vectorstore(self, doc_id: str, metadata: Optional[dict] = None) -> Chroma:
        return Chroma(
            client=self._client,
            collection_name=self.collection_name(doc_id),
            embedding_function=self.embeddings,
            collection_metadata=metadata,
        )

    # ── queries ──────────────────────────────────────────────
    def get_info(self, doc_id: str) -> Optional[Dict]:
        """Return stored info for a document, or None if it isn't fully ingested."""
        try:
            col = self._client.get_collection(self.collection_name(doc_id))
        except Exception:
            return None
        meta = col.metadata or {}
        expected = int(meta.get("chunks", -1))
        # An interrupted ingest leaves fewer vectors than expected -> treat as missing
        if expected < 0 or col.count() != expected:
            return None
        return {
            "doc_id": doc_id,
            "filename": meta.get("filename", "unknown.pdf"),
            "pages": int(meta.get("pages", 0)),
            "chunks": expected,
            "created_at": meta.get("created_at", ""),
        }

    def has_document(self, doc_id: str) -> bool:
        return self.get_info(doc_id) is not None

    def list_documents(self) -> List[Dict]:
        docs = []
        for c in self._client.list_collections():
            name = c if isinstance(c, str) else c.name
            if not name.startswith(COLLECTION_PREFIX):
                continue
            # collection name holds only the first 32 chars of the hash, which
            # is all we need to look it back up
            info = self.get_info(name[len(COLLECTION_PREFIX):])
            if info:
                docs.append(info)
        return sorted(docs, key=lambda d: d["created_at"], reverse=True)

    # ── writes ───────────────────────────────────────────────
    def add_document(
        self, doc_id: str, chunks: List[Document], filename: str, pages: int
    ) -> Chroma:
        """Embed + persist chunks. Replaces any half-finished earlier attempt."""
        self.delete_document(doc_id)
        chunks = filter_complex_metadata(chunks)
        meta = {
            "filename": filename,
            "pages": pages,
            "chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        vs = self._vectorstore(doc_id, meta)
        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start:start + ADD_BATCH_SIZE]
            ids = [f"{doc_id[:12]}-{start + i}" for i in range(len(batch))]
            vs.add_documents(batch, ids=ids)
        return vs

    def load(self, doc_id: str) -> Chroma:
        """Open an already-ingested document for retrieval (no re-embedding)."""
        if not self.has_document(doc_id):
            raise KeyError(f"Document {doc_id[:12]}… not found in ChromaDB")
        return self._vectorstore(doc_id)

    def delete_document(self, doc_id: str) -> None:
        try:
            self._client.delete_collection(self.collection_name(doc_id))
        except Exception:
            pass  # collection didn't exist
