# Agentic RAG 🤖 — Self-Correcting Document Q&A

> A production-grade RAG chatbot built with LangGraph that self-corrects poor retrievals
> instead of hallucinating, and shows its full reasoning trace for every answer.

🔗 **Live Demo:** [https://agentic-rag-tnpqnavtchvxixquwvnthx.streamlit.app/]

---

## What makes this different from a basic RAG chatbot

| Basic RAG (DocChat) | Agentic RAG |
|---|---|
| Fixed pipeline — always retrieves | Agent decides: retrieve or answer directly |
| No quality check on retrieved chunks | Grader node checks relevance before answering |
| Hallucinates when chunks are irrelevant | Self-corrects: rephrases + retries |
| No visibility into reasoning | Full reasoning trace shown for every answer |
| Answers even unanswerable questions | Graceful fallback refusal |

---

## Agent Architecture

```
User question
     ↓
[Router] — retrieve or answer directly?
     ↓ retrieve              ↓ direct
[Retriever]            [Direct Answer] → END
     ↓
[Grader] — are chunks actually relevant?
     ↓ good      ↓ bad + retries left    ↓ bad + no retries
[Answer]       [Rephrase] ──────────→ [Retriever] (loop)
     ↓              ↓ exhausted
    END          [Fallback] → END
```

---

## How to Use the Live Demo

1. Open the **[Live Demo](https://your-app.streamlit.app)**
2. Get a free Groq API key at [console.groq.com](https://console.groq.com) (no credit card)
3. Paste the key in the sidebar
4. Upload any PDF
5. Ask questions — expand the ** Reasoning trace** to see every agent decision

> **The reasoning trace is the most interesting part.** You can see exactly which nodes
> ran, whether the grader triggered a retry, and what the rephrased query was.

---

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph |
| LLM | Groq — Llama 3.1 8B (free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| PDF loading | LangChain PyPDFLoader |
| Memory | LangGraph MemorySaver |
| Deployment | Streamlit Community Cloud |

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

---

## Key Design Decisions

**Why a grader node?**
FAISS always returns k results even for off-topic questions — they're just the least irrelevant chunks available. Without a grader, the LLM tries to answer from irrelevant text and hallucinate. The grader does true semantic relevance assessment, not just distance measurement.

**Why rephrase instead of just retrying with the same query?**
Re-running the same query against FAISS returns the same chunks. The rephrase node converts conversational language to technical vocabulary that better matches the paper's text — giving the retriever a genuinely different search.

**Why MAX_RETRIES = 2?**
Enough attempts to handle most phrasings, but bounded to prevent infinite loops on genuinely unanswerable questions.

**Why MemorySaver for persistent memory?**
LangGraph's checkpointer saves full state after every node. The same thread_id resumes from saved state — no need to re-ingest documents between questions in the same session.

---

*Built as Week 2 of a RAG learning project — upgrades DocChat with agentic self-correction.*
