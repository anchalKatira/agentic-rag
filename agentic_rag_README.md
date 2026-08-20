# Agentic RAG 🤖 — Self-Correcting Document Q&A

> A production-grade RAG system built with LangGraph that self-corrects poor retrievals instead of hallucinating, and shows its full reasoning trace for every answer.

🔗 **Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## What makes this different from a basic RAG chatbot

| Basic RAG | Agentic RAG (this project) |
|---|---|
| Fixed pipeline — always retrieves | Always retrieves from your uploaded document |
| No quality check on retrieved chunks | Grader node checks relevance before answering |
| Hallucinates when chunks are irrelevant | Self-corrects: rephrases query + retries |
| No visibility into reasoning | Full reasoning trace shown per answer |
| Answers even unanswerable questions | Graceful refusal when nothing relevant found |

---

## How to Use

1. Open the **[Live Demo](https://your-app.streamlit.app)**
2. Upload any text-based PDF using the sidebar
3. Ask questions — the agent answers from the document
4. Expand **🧠 Reasoning trace** to see every decision the agent made
5. Expand **📎 Source chunks** to verify what the agent actually read

> The app works fully out of the box — no API key setup needed on your end.

---

## Agent Architecture

```
User question
      ↓
[Router] — always retrieves from uploaded document
      ↓
[Retriever] — FAISS similarity search, top-4 chunks
      ↓
[Grader] — are chunks actually relevant to the question?
      ↓ good                    ↓ bad (retries left)         ↓ bad (no retries)
[Answer node]             [Rephrase node]                [Fallback node]
      ↓                         ↓                              ↓
Final answer          Better technical query            Graceful refusal
                             ↓
                      [Retriever] → [Grader]
                      (self-correction loop)
```

---

## Key Features

**Grader node** — evaluates whether retrieved chunks genuinely answer the question. FAISS always returns k results even for off-topic queries — the grader catches irrelevant chunks before they reach the LLM.

**Self-correction loop** — when the grader says bad, the rephrase node rewrites the query using more technical vocabulary that better matches the document's text. The rephrased query goes back to the retriever for a second attempt. Retrying with the same query returns the same chunks — rephrasing makes the retry genuinely different.

**Reasoning trace** — every answer shows an expandable panel with each node's decision: router verdict, retriever pages, grader verdict and reason, whether rephrasing fired, and the answer mode (RAG or fallback).

**Graceful fallback** — after MAX_RETRIES failed attempts, the agent refuses clearly instead of hallucinating an answer.

**Persistent memory** — MemorySaver checkpointer saves state after every node. Same conversation thread remembers previous turns. New PDF upload generates a fresh thread_id for a clean start.

---

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph |
| LLM | Groq — openai/gpt-oss-120b (free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| PDF loading | LangChain PyPDFLoader |
| Text chunking | RecursiveCharacterTextSplitter (500 chars, 50 overlap) |
| Memory | LangGraph MemorySaver |
| Deployment | Streamlit Community Cloud |

---

## Ingestion Pipeline

```
PDF upload (bytes)
      ↓
PyPDFLoader → list of Document objects (one per page)
      ↓
Filter empty pages
      ↓
RecursiveCharacterTextSplitter → chunks (500 chars, 50 overlap)
      ↓
HuggingFaceEmbeddings → 384-dim vectors
      ↓
FAISS index (stored in memory, rebuilt per session)
```

---

## Run Locally

```bash
git clone https://github.com/anchalKatira/agentic-rag
cd agentic-rag
pip install -r requirements.txt
streamlit run app.py
```

Add your Groq key for local use:
```bash
export GROQ_API_KEY="gsk_your_key_here"   # Mac/Linux
set GROQ_API_KEY=gsk_your_key_here        # Windows
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

---

## PDF Requirements

The app works with text-based PDFs — documents created digitally where text can be selected and copied. It does not support scanned PDFs (image-based documents where text cannot be highlighted). To check: open the PDF and try to highlight text with your cursor. If you can → it will work. If you cannot → it is scanned.

---

## Key Design Decisions

**Why always retrieve instead of routing?**
The user uploaded a PDF specifically to ask questions about it. Sending questions to the LLM's general knowledge instead of the document defeats the purpose of the app. Every question goes through FAISS — the grader then decides if the chunks are useful.

**Why a grader node instead of just using FAISS scores?**
FAISS distance measures vector similarity, not semantic relevance. A chunk can be close in vector space (similar vocabulary) but still not answer the question. The LLM grader does true relevance assessment — it understands whether the content addresses the question.

**Why rephrase instead of retrying with the same query?**
Re-running the same query returns the same chunks. The rephrase node converts conversational language to technical vocabulary that better matches the document's text in vector space — making the second retrieval genuinely different.

**Why refuse instead of answering from general knowledge?**
If the system answered from general knowledge when the document doesn't cover something, users have no way to know whether the answer came from the document or from LLM training data. A clear refusal is more honest and maintains trust in the system.

**Why MemorySaver over a database?**
For a single-user portfolio app, in-memory checkpointing is sufficient. The state persists across conversation turns within a session. For multi-user production with persistence across restarts, SqliteSaver or PostgresSaver would be the upgrade path.

---

## Comparison with DocChat (Week 1 project)

DocChat was a basic RAG pipeline — fixed steps, no quality checking, no self-correction. Agentic RAG adds three things DocChat doesn't have: a grader that checks retrieval quality, a self-correction loop that rephrases and retries, and a graceful fallback. The architecture is also fundamentally different — an explicit LangGraph state machine vs a simple LangChain chain.

---

*Built as Week 2 of a structured AI/ML project series — directly replicating patterns from the Oracle AI Developer Hub enterprise RAG systems.*
