"""PDF -> chunks -> ChromaDB."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHUNK_OVERLAP, CHUNK_SIZE
from db.chroma_store import ChromaStore, compute_doc_id


@dataclass
class IngestResult:
    doc_id: str
    vectorstore: object  # langchain_chroma.Chroma
    filename: str
    pages: int
    chunks: int
    reused: bool  # True -> loaded from disk, nothing was re-embedded


def load_pages(pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp = f.name
    try:
        pages = PyPDFLoader(tmp).load()
    finally:
        os.unlink(tmp)

    pages = [p for p in pages if p.page_content.strip()]  # drop empty pages
    if not pages:
        raise ValueError("No text extracted. PDF may be scanned/image-based.")
    for p in pages:
        p.metadata["source"] = filename  # tmp path is meaningless once persisted
    return pages


def split_pages(pages):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError("No usable content found in PDF.")
    return chunks


def ingest_pdf(pdf_bytes: bytes, filename: str, store: ChromaStore) -> IngestResult:
    doc_id = compute_doc_id(pdf_bytes)

    # Same PDF seen before -> reuse persisted vectors, skip parsing + embedding
    info = store.get_info(doc_id)
    if info:
        return IngestResult(
            doc_id=doc_id,
            vectorstore=store.load(doc_id),
            filename=info["filename"],
            pages=info["pages"],
            chunks=info["chunks"],
            reused=True,
        )

    pages = load_pages(pdf_bytes, filename)
    chunks = split_pages(pages)
    vs = store.add_document(doc_id, chunks, filename=filename, pages=len(pages))
    return IngestResult(doc_id, vs, filename, len(pages), len(chunks), reused=False)
