# Agentic RAG 🤖 — Self-Correcting Document Q&A

> A production-grade RAG system built with LangGraph that self-corrects poor retrievals instead of hallucinating, and shows its full reasoning trace for every answer.

🔗 **Live Demo:** [agentic-rag-tnpqnavtchvxixquwvnthx.streamlit.app](https://agentic-rag-tnpqnavtchvxixquwvnthx.streamlit.app)

---

## What makes this different from a basic RAG chatbot

| Basic RAG | Agentic RAG (this project) |
|---|---|
| Fixed pipeline — always retrieves | Always retrieves from uploaded document |
| No quality check on retrieved chunks | Grader node checks relevance before answering |
| Hallucinates when chunks are irrelevant | Self-corrects: rephrases query + retries |
| No visibility into reasoning | Full reasoning trace shown per answer |
| Answers even unanswerable questions | Answers from available chunks even on fallback |

---

## How to Use

1. Open the **[Live Demo](https://agentic-rag-tnpqnavtchvxixquwvnthx.streamlit.app)**
2. Upload any text-based PDF using the sidebar
3. Ask questions — the agent answers from the document
4. Expand **🧠 Reasoning trace** to see every node decision
5. Expand **📎 Source chunks** to verify what the agent actually read

> The app works fully out of the box — no API key setup needed on your end.

---

## Agent Architecture

```
User question
      ↓
[Router] — always retrieves (document Q&A mode)
      ↓
[Retriever] — FAISS similarity search, top-4 chunks
      ↓
[Grader] — are chunks relevant to the question?
      ↓ good                 ↓ bad + retries left      ↓ bad + no retries
[Answer node]          [Rephrase node]             [Fallback node]
      ↓                      ↓                          ↓
Answer from docs     Better technical query      Answer from available
                            ↓                     chunks anyway
                     [Retriever] → [Grader]
                     (self-correction loop,
                      up to MAX_RETRIES=2)
```

---

## Key Features

**Router node** — always routes to retrieval. Since the user uploaded a PDF specifically to ask questions about it, every question goes through FAISS first. The grader then decides what to do with the results.

**Grader node** — evaluates whether retrieved chunks genuinely answer the question. FAISS always returns k results even for off-topic queries — the grader catches irrelevant chunks before they reach the LLM. Grades `good` if any relevant information is found, `bad` only if chunks are completely unrelated.

**Self-correction loop** — when the grader says bad, the rephrase node rewrites the query using more technical vocabulary that better matches the document's text. The rephrased query goes back to the retriever for a second attempt. Re-running the same query returns the same chunks — rephrasing makes the retry genuinely different.

**Fallback node** — after MAX_RETRIES (2) failed attempts, the fallback node still tries to answer from whatever chunks were retrieved rather than refusing outright. Only if no chunks exist does it say the information wasn't found.

**Reasoning trace** — every answer shows an expandable panel with each node's decision shown as plain text steps with emoji labels: 🔀 Router, 🔍 Retriever, ⚖️ Grader, ✏️ Rephrase, ✅ Answer, ⚠️ Fallback. Retries are labelled ↺.

**Persistent memory** — MemorySaver checkpointer saves state after every node. Same conversation thread remembers previous turns within a session. New PDF upload generates a fresh UUID as thread_id for a clean start.

---

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph |
| LLM | Groq — `openai/gpt-oss-120b` (free) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (top-4 chunks per query) |
| PDF loading | LangChain `PyPDFLoader` |
| Text chunking | `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) |
| Memory | LangGraph `MemorySaver` |
| Deployment | Streamlit Community Cloud |

---

## Agent State

Every node reads from and writes to a shared TypedDict:

```python
class AgentState(TypedDict):
    question:           str    # user's original question
    rephrased_question: str    # rewritten query for retry
    route:              str    # always 'retrieve'
    documents:          List   # retrieved FAISS chunks
    grade:              str    # 'good' or 'bad'
    answer:             str    # final answer
    retry_count:        int    # 0, 1, or 2
    steps:              List   # reasoning trace log
    chat_history:       List   # memory across turns
```

---

## Ingestion Pipeline

```
PDF upload (bytes)
      ↓
Write to temp file (PyPDFLoader needs a path, not bytes)
      ↓
PyPDFLoader → Document objects (one per page)
      ↓
Filter empty pages
      ↓
RecursiveCharacterTextSplitter → chunks (500 chars, 50 overlap)
      ↓
Filter empty chunks
      ↓
HuggingFaceEmbeddings → 384-dim vectors
      ↓
FAISS.from_documents() → index stored in session memory
```

---

## Run Locally

```bash
git clone https://github.com/anchalKatira/agentic-rag
cd agentic-rag
pip install -r requirements.txt
streamlit run app.py
```

Add your Groq key:
```bash
export GROQ_API_KEY="gsk_your_key_here"   # Mac/Linux
set GROQ_API_KEY=gsk_your_key_here        # Windows
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

---

## PDF Requirements

Works with **text-based PDFs** — documents created digitally where text can be selected and copied.

Does **not** support scanned PDFs (image-based documents). To check: open the PDF and try to highlight text. If you can → it works. If you cannot → it is scanned.

---

## Key Design Decisions

**Why always retrieve instead of routing?**
The user uploaded a PDF to ask questions about it. Routing questions to the LLM's general knowledge defeats the purpose. Every question goes through FAISS — the grader decides if the result is useful.

**Why a grader instead of using FAISS scores?**
FAISS distance measures vector similarity, not semantic relevance. A chunk can be close in vector space but still not answer the question. The LLM grader does true relevance assessment.

**Why rephrase instead of retrying the same query?**
Same query → same chunks. The rephrase node converts conversational language to technical vocabulary matching the document's text — making the retry genuinely different.

**Why answer from chunks even in fallback?**
A hard refusal is frustrating when partial information exists. The fallback node uses the RAG prompt on whatever chunks were retrieved — giving the user something useful even when the grader wasn't satisfied.

**Why MemorySaver over a database?**
Single-user portfolio app — in-memory state is sufficient per session. For multi-user production with persistence across restarts, `SqliteSaver` or `PostgresSaver` would replace it.

---

## Comparison with DocChat

| | DocChat (Week 1) | Agentic RAG (Week 2) |
|---|---|---|
| Pipeline | Fixed chain | LangGraph state machine |
| Quality check | None | Grader node |
| Bad retrieval | Hallucinates | Rephrases + retries |
| Memory | ConversationBufferWindowMemory | LangGraph MemorySaver |
| Trace | Hidden | Visible per answer |
