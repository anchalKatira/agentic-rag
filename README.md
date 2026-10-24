# Agentic RAG 🤖 — Self-Correcting Document Q&A

> A production-style RAG system built with LangGraph that self-corrects poor retrievals instead of hallucinating, shows its full reasoning trace for every answer, and stores document vectors in a **persistent ChromaDB** database.

🔗 **Live Demo:** [agentic-rag.streamlit.app](https://agentic-rag-tnpqnavtchvxixquwvnthx.streamlit.app/)

---

## What makes this different from a basic RAG chatbot

| Basic RAG                               | Agentic RAG (this project)                    |
| --------------------------------------- | --------------------------------------------- |
| Fixed pipeline — always retrieves       | Always retrieves from your uploaded document  |
| No quality check on retrieved chunks    | Grader node checks relevance before answering |
| Hallucinates when chunks are irrelevant | Self-corrects: rephrases query + retries      |
| No visibility into reasoning            | Full reasoning trace shown per answer         |
| Index rebuilt on every upload / restart | Persistent ChromaDB — ingest once, reuse      |

---

## How to Use

1. Open the **[Live Demo](https://agentic-rag-tnpqnavtchvxixquwvnthx.streamlit.app/)**
2. Upload any text-based PDF using the sidebar (or load a previously saved one)
3. Ask questions — the agent answers from the document
4. Expand **🧠 Reasoning trace** to see every decision the agent made
5. Expand **📎 Source chunks** to verify what the agent actually read

---

## Project Structure

```
agentic-rag/
├── app.py                  # Streamlit UI only
├── requirements.txt
├── core/
│   ├── config.py           # model, chunk size, top-k, Chroma path
│   ├── state.py            # LangGraph AgentState
│   ├── prompts.py          # grader / rephrase / RAG prompts
│   ├── embeddings.py       # cached MiniLM embedding model
│   ├── ingestion.py        # PDF → chunks → ChromaDB
│   └── agent.py            # LangGraph nodes, edges, run_agent()
├── db/
│   └── chroma_store.py     # persistent ChromaDB wrapper
└── evaluation/
    ├── metrics.py          # hit rate, MRR, answer keyword hit
    ├── evaluate.py         # CLI evaluation runner
    └── sample_dataset.json # template Q/A set — edit with your own
```

---

## Agent Architecture

```
User question
      ↓
[Router] — always retrieves from uploaded document
      ↓
[Retriever] — ChromaDB similarity search, top-4 chunks
      ↓
[Grader] — are chunks actually relevant to the question?
      ↓ good                    ↓ bad (retries left)         ↓ bad (no retries)
[Answer node]             [Rephrase node]                [Fallback node]
      ↓                         ↓                              ↓
Final answer          Better technical query        Best-effort answer from
                             ↓                      the chunks found
                      [Retriever] → [Grader]
                      (self-correction loop)
```

---

## Key Features

**Grader node** — evaluates whether retrieved chunks genuinely answer the question. A vector store always returns k results even for off-topic queries — the grader catches irrelevant chunks before they reach the LLM.

**Self-correction loop** — when the grader says bad, the rephrase node rewrites the query using more technical vocabulary that better matches the document's text. The rephrased query goes back to the retriever for a second attempt.

**Reasoning trace** — every answer shows an expandable panel with each node's decision: router verdict, retriever pages, grader verdict and reason, whether rephrasing fired, and the answer mode (RAG or fallback).

**Fallback** — after `MAX_RETRIES` failed attempts, the agent answers from the best chunks it found (or says clearly that nothing relevant exists) instead of making things up.

**Persistent vector storage** — document vectors, chunk text and page metadata are saved to disk by ChromaDB. Re-uploading the same PDF (even after a restart) skips parsing and embedding.

**Conversation memory** — LangGraph `MemorySaver` checkpoints state after every node within a session. Loading a new PDF starts a fresh thread.

---

## Tech Stack

| Component       | Tool                                                   |
| --------------- | ------------------------------------------------------ |
| UI              | Streamlit                                              |
| Agent framework | LangGraph                                              |
| LLM             | Groq — openai/gpt-oss-120b (free)                      |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2 (384-dim)       |
| Vector store    | **ChromaDB** (persistent, `langchain-chroma`)          |
| PDF loading     | LangChain PyPDFLoader                                  |
| Text chunking   | RecursiveCharacterTextSplitter (500 chars, 50 overlap) |
| Memory          | LangGraph MemorySaver                                  |
| Deployment      | Streamlit Community Cloud                              |

---

## Ingestion Pipeline

```
PDF upload (bytes)
      ↓
SHA-256 hash of bytes → doc_id
      ↓
Already in ChromaDB? ── yes ──→ load collection from disk (no re-embedding)
      ↓ no
PyPDFLoader → one Document per page → drop empty pages
      ↓
RecursiveCharacterTextSplitter → chunks (500 chars, 50 overlap)
      ↓
HuggingFaceEmbeddings → 384-dim vectors
      ↓
ChromaDB collection `doc_<hash>` saved to ./chroma_db/
```

---

## How ChromaDB Is Used

- **One collection per PDF**, named from the PDF's content hash, so documents never mix and duplicates are detected automatically.
- **Each chunk is stored as:** an ID, its 384-dim embedding, the chunk text, and metadata (`page`, `source`).
- **Each collection also stores** `filename`, `pages`, `chunks` and `created_at`, which powers the "Saved documents" list in the sidebar.
- **Integrity check:** on load, the stored vector count must equal the expected chunk count; a half-finished ingest is detected and rebuilt.
- **On disk:** `chroma_db/chroma.sqlite3` (text + metadata) plus an HNSW index folder per collection.

Change the storage location with an environment variable:

```
export CHROMA_PERSIST_DIR=/path/to/store
```

---

## Run Locally

```
git clone https://github.com/anchalKatira/agentic-rag
cd agentic-rag
pip install -r requirements.txt
export GROQ_API_KEY="gsk_your_key_here"   # Mac/Linux
set GROQ_API_KEY=gsk_your_key_here        # Windows
streamlit run app.py
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

---

## Evaluation

Edit `evaluation/sample_dataset.json` with questions about your PDF:

```json
[
  {
    "question": "What is gradient descent?",
    "expected_keywords": ["optimization", "loss"],
    "expected_pages": [3]
  }
]
```

Retrieval-only (no API key needed):

```
python -m evaluation.evaluate --pdf notes.pdf --dataset evaluation/sample_dataset.json
```

Full agent run (needs `GROQ_API_KEY`):

```
python -m evaluation.evaluate --pdf notes.pdf --dataset my_qa.json --agent --out report.json
```

| Metric | Meaning |
| --- | --- |
| `hit_rate@k` | % of questions where a top-k chunk contains an expected keyword / page |
| `mrr` | Mean reciprocal rank of the first correct chunk (1.0 = always first) |
| `answer_keyword_hit_rate` | % of final answers that mention an expected keyword |
| `retry_rate` / `fallback_rate` | How often the self-correction loop / fallback fired |
| `avg_latency_s` | Average seconds per question |

---

## PDF Requirements

The app works with text-based PDFs — documents created digitally where text can be selected and copied. It does not support scanned PDFs (image-based documents). To check: open the PDF and try to highlight text with your cursor. If you can → it will work.

---

## Key Design Decisions

**Why always retrieve instead of routing?** The user uploaded a PDF specifically to ask questions about it. Sending questions to the LLM's general knowledge defeats the purpose. Every question goes through the vector store — the grader then decides if the chunks are useful.

**Why a grader node instead of just using similarity scores?** Vector distance measures similarity, not whether a chunk actually answers the question. The LLM grader does a true relevance assessment.

**Why rephrase instead of retrying with the same query?** Re-running the same query returns the same chunks. The rephrase node converts conversational language to technical vocabulary that better matches the document, making the second retrieval genuinely different.

**Why ChromaDB over FAISS?** FAISS is an in-memory index: it vanishes on restart and doesn't store text or metadata. ChromaDB persists vectors, text and metadata together, so documents are embedded once and reused.

**Why a content hash for document IDs?** It makes ingestion idempotent — the same file always maps to the same collection, so duplicates are never re-embedded.

**Why one collection per document?** Hard isolation: a question about one PDF can never retrieve chunks from another.

**Why MemorySaver over a database?** For a single-user portfolio app, in-memory checkpointing is enough. For multi-user production, `SqliteSaver` or `PostgresSaver` is the upgrade path.

---

## Deployment Note

On Streamlit Community Cloud the filesystem is ephemeral: `chroma_db/` survives normal reruns but is cleared on redeploy or after the app is reclaimed. For durable persistence in production, mount a volume or switch to a hosted Chroma server.

---

## Next steps

- Hybrid search (BM25 + vectors) and a reranker
- `SqliteSaver` for chat memory that survives restarts
- Multi-document Q&A across collections
- Ragas-style LLM-judged faithfulness metrics in `evaluation/`

