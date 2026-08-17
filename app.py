"""
Agentic RAG 
===========================
A production-grade RAG chatbot with:
  - PDF upload and ingestion
  - LangGraph multi-node agent (router → retriever → grader → answer/rephrase/fallback)
  - Self-correction loop (rephrase + retry when retrieval quality is poor)
  - Reasoning trace panel (shows every node decision)
  - Persistent memory via MemorySaver
  - Source chunks attribution
"""

import os
import tempfile
import uuid
import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:#0f0f11; --surface:#18181c; --border:#2a2a32;
  --accent:#6c63ff; --accent2:#a78bfa;
  --text:#e8e8f0; --muted:#72728a;
  --green:#34d399; --yellow:#fbbf24;
  --red:#f87171; --blue:#60a5fa;
}

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text)}
#MainMenu,footer,header{visibility:hidden}
.main .block-container{padding-top:1.25rem;padding-bottom:2rem;max-width:900px}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)}

/* Messages */
.msg-wrap{margin-bottom:1.1rem}
.msg-role{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}
.msg-role.user{color:var(--accent2)}.msg-role.bot{color:var(--green)}
.msg-bubble{padding:.85rem 1.1rem;border-radius:10px;font-size:.92rem;line-height:1.7;border:1px solid var(--border)}
.msg-bubble.user{background:#1e1e2e}.msg-bubble.bot{background:#16161f}

/* Reasoning trace */
.trace-step{display:flex;gap:10px;align-items:flex-start;padding:5px 0;border-bottom:.5px solid var(--border);font-size:.8rem}
.trace-step:last-child{border-bottom:none}
.trace-num{width:20px;height:20px;border-radius:50%;background:var(--surface);border:.5px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:500;flex-shrink:0;margin-top:1px}
.trace-text{color:var(--muted);line-height:1.5;font-family:'DM Mono',monospace}
.trace-text b{color:var(--text)}

/* Node badges */
.node-badge{display:inline-block;font-size:.65rem;padding:1px 6px;border-radius:4px;margin-right:4px;font-weight:600;font-family:'DM Mono',monospace}
.nb-router{background:#1a1a2e;color:var(--accent2);border:.5px solid var(--accent)}
.nb-retriever{background:#0a2e1a;color:var(--green);border:.5px solid #065f46}
.nb-grader{background:#2d1a00;color:var(--yellow);border:.5px solid #78350f}
.nb-rephrase{background:#1a0a2e;color:var(--blue);border:.5px solid #1d4ed8}
.nb-answer{background:#0a2e1a;color:var(--green);border:.5px solid #065f46}
.nb-fallback{background:#2d0a0a;color:var(--red);border:.5px solid #7f1d1d}

/* Source chunks */
.src-chunk{font-size:.78rem;color:var(--muted);background:#111118;border-left:2px solid var(--accent);padding:.5rem .75rem;border-radius:0 6px 6px 0;font-family:'DM Mono',monospace;margin:4px 0 8px;white-space:pre-wrap;line-height:1.5}

/* Status pills */
.pill{display:inline-flex;align-items:center;gap:5px;font-size:.75rem;padding:3px 9px;border-radius:20px;border:.5px solid var(--border);background:var(--surface);color:var(--muted)}
.dot{width:6px;height:6px;border-radius:50%}
.dot-green{background:var(--green);box-shadow:0 0 5px var(--green)}
.dot-orange{background:var(--yellow)}

/* Divider */
.divider{height:1px;background:var(--border);margin:.9rem 0}

/* Retry badge */
.retry-badge{font-size:.72rem;background:#1a1a00;color:var(--yellow);border:.5px solid var(--yellow);border-radius:4px;padding:2px 7px;margin-left:6px}

.stButton>button{background:var(--surface);border:.5px solid var(--border);color:var(--text);border-radius:8px;font-family:'DM Sans',sans-serif;transition:all .15s}
.stButton>button:hover{border-color:var(--accent);color:var(--accent2)}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MAX_RETRIES = 2
MODEL       = "openai/gpt-oss-120b"


# ─────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:           str
    rephrased_question: str
    route:              str
    documents:          List[Document]
    grade:              str
    answer:             str
    retry_count:        int
    steps:              List[str]
    chat_history:       List[dict]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_pdf(pdf_bytes: bytes) -> tuple:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp = f.name
    try:
        pages = PyPDFLoader(tmp).load()
    finally:
        os.unlink(tmp)

    # Filter out empty pages
    pages = [p for p in pages if p.page_content.strip()]

    if not pages:
        raise ValueError("No text extracted. PDF may be scanned/image-based.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)

    # Filter out empty chunks
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        raise ValueError("No usable content found in PDF.")

    em = load_embedding_model()
    vs = FAISS.from_documents(chunks, em)
    return vs, len(pages), len(chunks)


# ─────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────
def build_graph(vectorstore: FAISS, api_key: str):
    """Build the full agentic RAG graph for a given vectorstore."""

    llm = ChatGroq(
        model=MODEL,
        temperature=0,
        groq_api_key=api_key,
    )

    # ── Structured output schemas ──────────────────────────────
    # ── Structured output schemas ──────────────────────────────
    class RouteDecision(BaseModel):
        route: Literal["retrieve", "direct"] = Field(
            description="routing decision"
        )

    class GradeDecision(BaseModel):
        grade:  Literal["good", "bad"] = Field(
            description="relevance grade"
        )
        reason: str = Field(description="reason")

    # ── Prompts ────────────────────────────────────────────────
    grader_chain = ChatPromptTemplate.from_messages([
        ("system",
         """You are grading document chunks for relevance.
         
Grade 'good' if the chunks contain ANY of these:
- The topic or concept mentioned in the question
- Related technical terms
- Partial information about the subject
- Background information that helps answer

Grade 'bad' ONLY if chunks are about a completely different subject with zero overlap.

DEFAULT to 'good' unless chunks are totally irrelevant."""),
        ("human", "Question: {question}\n\nChunks:\n{context}")
    ]) | llm.with_structured_output(GradeDecision)

    rephrase_chain = ChatPromptTemplate.from_messages([
        ("system",
         """Rewrite the question using different technical keywords
for better document search. Return ONLY the rewritten question, 5-10 words."""),
        ("human", "Original: {question}\n\nRewritten:")
    ]) | llm

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Answer using the provided document context.
Use the context as your PRIMARY source.
If context partially answers, give what you can and note gaps.
Do NOT say the information is not available if context is partially relevant.

Context:
{context}"""),
        ("human", "{question}")
    ])

    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer concisely."),
        ("human", "{question}")
    ])

    # ── Nodes ──────────────────────────────────────────────────
    def router_node(state: AgentState) -> dict:
        # Always retrieve — user uploaded PDF to ask about it
        step = "Router → retrieve (document Q&A mode)"
        return {
            "route":       "retrieve",
            "retry_count": 0,
            "steps":       state.get("steps", []) + [step]
        }

    def retriever_node(state: AgentState) -> dict:
        rc    = state.get("retry_count", 0)
        query = (state["rephrased_question"]
                 if rc > 0 and state.get("rephrased_question")
                 else state["question"])
        src   = "rephrased" if rc > 0 else "original"
        docs  = vectorstore.similarity_search(query, k=4)
        pages = [str(d.metadata.get("page", "?")) for d in docs]
        step  = f"Retriever (attempt {rc+1}, {src}) → {len(docs)} chunks from pages {', '.join(pages)}"
        return {
            "documents": docs,
            "steps":     state.get("steps", []) + [step]
        }

    def grader_node(state: AgentState) -> dict:
        docs = state["documents"]
        if not docs:
            return {"grade": "bad",
                    "steps": state.get("steps", []) + ["Grader → 'bad' | no documents retrieved"]}

        ctx = "\n\n".join([
            f"[Chunk {i+1}, Page {d.metadata.get('page','?')}]:\n{d.page_content[:400]}"
            for i, d in enumerate(docs)
        ])
        d    = grader_chain.invoke({"question": state["question"], "context": ctx})
        step = f"Grader → '{d.grade}' | {d.reason}"
        return {"grade": d.grade, "steps": state.get("steps", []) + [step]}

    def rephrase_node(state: AgentState) -> dict:
        rc    = state.get("retry_count", 0)
        new_q = rephrase_chain.invoke({"question": state["question"]}).content.strip()
        step  = f"Rephrase (attempt {rc+1}/{MAX_RETRIES}) → '{new_q}'"
        return {
            "rephrased_question": new_q,
            "retry_count":        rc + 1,
            "steps":              state.get("steps", []) + [step]
        }

    def answer_node(state: AgentState) -> dict:
        docs = state.get("documents", [])
        if docs:
            ctx  = "\n\n".join([
                f"[Page {d.metadata.get('page','?')}]: {d.page_content.strip()}"
                for d in docs
            ])
            resp = (rag_prompt | llm).invoke({
                "context":  ctx,
                "question": state["question"]
            })
            mode = "RAG"
        else:
            resp = (direct_prompt | llm).invoke({"question": state["question"]})
            mode = "Direct"

        answer  = resp.content
        history = state.get("chat_history", []) + [{
            "question": state["question"],
            "answer":   answer,
            "sources":  state.get("documents", []),
            "steps":    state.get("steps", [])
        }]
        step = f"Answer ({mode}) → {len(answer)} chars"
        return {
            "answer":       answer,
            "chat_history": history,
            "steps":        state.get("steps", []) + [step]
        }

    def direct_answer_node(state: AgentState) -> dict:
        return answer_node({**state, "documents": []})

    def fallback_node(state: AgentState) -> dict:
        # Even on fallback — answer from whatever chunks we have
        docs = state.get("documents", [])
        if docs:
            ctx  = "\n\n".join([
                f"[Page {d.metadata.get('page','?')}]: {d.page_content.strip()}"
                for d in docs
            ])
            resp = (rag_prompt | llm).invoke({
                "context":  ctx,
                "question": state["question"]
            })
            answer = resp.content
            mode   = "RAG fallback"
        else:
            answer = (f"I couldn't find specific information about "
                      f"'{state['question']}' in the document.")
            mode   = "no chunks"

        history = state.get("chat_history", []) + [{
            "question": state["question"],
            "answer":   answer,
            "sources":  docs,
            "steps":    state.get("steps", [])
        }]
        step = f"Fallback ({mode}) → answered with available context"
        return {
            "answer":       answer,
            "chat_history": history,
            "steps":        state.get("steps", []) + [step]
        }

    # ── Conditional edges ─────────────────────────────────────
    def route_decision(state):
        return "retriever"   # always retrieve

    def grade_decision(state):
        if state.get("grade") == "good":
            return "answer"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rephrase"
        return "fallback"   # fallback now answers from chunks, not refuses

    # ── Assemble ──────────────────────────────────────────────
    g = StateGraph(AgentState)
    g.add_node("router",        router_node)
    g.add_node("retriever",     retriever_node)
    g.add_node("grader",        grader_node)
    g.add_node("rephrase",      rephrase_node)
    g.add_node("answer",        answer_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("fallback",      fallback_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_decision,
                             {"retriever":"retriever","direct_answer":"direct_answer"})
    g.add_edge("retriever", "grader")
    g.add_conditional_edges("grader", grade_decision,
                             {"answer":"answer","rephrase":"rephrase","fallback":"fallback"})
    g.add_edge("rephrase",      "retriever")
    g.add_edge("answer",        END)
    g.add_edge("direct_answer", END)
    g.add_edge("fallback",      END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────────────────────
NODE_CLASSES = {
    "Router":    "nb-router",
    "Retriever": "nb-retriever",
    "Grader":    "nb-grader",
    "Rephrase":  "nb-rephrase",
    "Answer":    "nb-answer",
    "Fallback":  "nb-fallback",
    "Direct":    "nb-answer",
}


def render_trace(steps: list):
    """Render the agent's reasoning trace as styled HTML."""
    html = '<div style="padding:.25rem 0">'
    for i, step in enumerate(steps):
        # Detect which node this step is from
        node_name = step.split("→")[0].strip().split("(")[0].strip()
        css_class = NODE_CLASSES.get(node_name, "nb-router")

        # Highlight retry badge
        retry_badge = ""
        if "attempt 2" in step.lower() or "attempt 3" in step.lower():
            retry_badge = '<span class="retry-badge">↺ retry</span>'

        html += f"""
        <div class="trace-step">
          <div class="trace-num">{i+1}</div>
          <div class="trace-text">
            <span class="node-badge {css_class}">{node_name}</span>
            {retry_badge}
            {step.split('→',1)[-1].strip() if '→' in step else step}
          </div>
        </div>"""
    html += '</div>'
    return html


def render_message(role: str, content: str, sources=None, steps=None):
    label = "You" if role == "user" else "Agent"
    cls   = "user" if role == "user" else "bot"

    st.markdown(f"""
    <div class="msg-wrap">
      <div class="msg-role {cls}">{label}</div>
      <div class="msg-bubble {cls}">{content}</div>
    </div>
    """, unsafe_allow_html=True)

    if steps and role == "assistant":
        has_retry    = any("attempt 2" in s.lower() or "attempt 3" in s.lower() for s in steps)
        has_fallback = any("fallback" in s.lower() for s in steps)
        label_extra  = " ↺ self-corrected" if has_retry else (" ⚠ fallback" if has_fallback else "")

        with st.expander(f"🧠 Reasoning trace ({len(steps)} steps){label_extra}"):
            for i, step in enumerate(steps):
                # Parse node name and content
                parts     = step.split("→", 1)
                node_name = parts[0].strip().split("(")[0].strip()
                detail    = parts[1].strip() if len(parts) > 1 else step

                # Pick emoji per node
                icons = {
                    "Router":    "🔀",
                    "Retriever": "🔍",
                    "Grader":    "⚖️",
                    "Rephrase":  "✏️",
                    "Answer":    "✅",
                    "Fallback":  "⚠️",
                    "Direct":    "💬",
                }
                icon = icons.get(node_name, "▸")

                # Retry indicator
                retry_tag = " ↺ retry" if ("attempt 2" in step.lower() or "attempt 3" in step.lower()) else ""

                st.markdown(
                    f"**{i+1}. {icon} {node_name}**{retry_tag}  \n"
                    f"<span style='color:#72728a;font-size:.85rem'>{detail}</span>",
                    unsafe_allow_html=True
                )
                if i < len(steps) - 1:
                    st.markdown("---")

    if sources and role == "assistant":
        with st.expander(f"📎 {len(sources)} source chunk(s) used"):
            for i, doc in enumerate(sources):
                pg   = doc.metadata.get("page", "?")
                text = doc.page_content.strip()[:350]
                st.markdown(f"**Chunk {i+1} · page {pg}**")
                st.code(text, language=None)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "messages":    [],   # [{role, content, sources, steps}]
    "graph":       None,
    "vs":          None,
    "pdf_name":    None,
    "pdf_pages":   0,
    "pdf_chunks":  0,
    "thread_id":   str(uuid.uuid4()),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Agentic RAG")
    st.caption("Self-correcting document Q&A with LangGraph")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # API Key
    st.markdown("#### 🔑 Groq API Key")
    env_key = get_api_key()
    if env_key:
        st.markdown('<div class="pill"><span class="dot dot-green"></span>Key loaded</div>',
                    unsafe_allow_html=True)
        api_key = env_key
    else:
        api_key = st.text_input("Paste key", type="password", placeholder="gsk_...",
                                label_visibility="collapsed")
        if api_key:
            st.success("Key ready ✓")
        else:
            st.warning("Enter your Groq API key")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # PDF Upload
    st.markdown("#### 📄 Upload PDF")
    uploaded = st.file_uploader("Any PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded and api_key and uploaded.name != st.session_state.pdf_name:
        with st.spinner("Ingesting PDF…"):
            try:
                vs, pages, chunks = ingest_pdf(uploaded.read())
                graph = build_graph(vs, api_key)
                st.session_state.vs         = vs
                st.session_state.graph      = graph
                st.session_state.pdf_name   = uploaded.name
                st.session_state.pdf_pages  = pages
                st.session_state.pdf_chunks = chunks
                st.session_state.messages   = []
                st.session_state.thread_id  = str(uuid.uuid4())
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    elif uploaded and not api_key:
        st.warning("Enter API key first.")

    # Doc stats
    if st.session_state.pdf_name:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 Document")
        c1, c2 = st.columns(2)
        c1.metric("Pages",  st.session_state.pdf_pages)
        c2.metric("Chunks", st.session_state.pdf_chunks)
        st.caption(st.session_state.pdf_name)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # How it works
    st.markdown("#### ⚙️ Agent Architecture")
    st.markdown("""
    <div style="font-size:.78rem;color:#72728a;line-height:1.8">
    1. <b style="color:#a78bfa">Router</b> — retrieve or answer directly?<br>
    2. <b style="color:#34d399">Retriever</b> — FAISS top-3 chunks<br>
    3. <b style="color:#fbbf24">Grader</b> — are chunks relevant?<br>
    4. <b style="color:#60a5fa">Rephrase</b> — if not, try better query<br>
    5. <b style="color:#34d399">Answer</b> — grounded final response<br>
    6. <b style="color:#f87171">Fallback</b> — graceful refusal
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages   = []
            st.session_state.thread_id  = str(uuid.uuid4())
            st.rerun()
    with col2:
        if st.button("📄 New PDF", use_container_width=True):
            for k in ["messages","graph","vs","pdf_name","pdf_pages","pdf_chunks"]:
                st.session_state[k] = defaults[k]
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
st.markdown("# Agentic RAG")
st.markdown("*Self-correcting document Q&A — powered by LangGraph*")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Empty state
if not st.session_state.graph:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;color:#72728a'>
      <div style='font-size:2.5rem;margin-bottom:.75rem'>🤖</div>
      <div style='font-size:1.15rem;color:#e8e8f0;margin-bottom:.5rem;font-weight:500'>
        Upload a PDF to start
      </div>
      <div style='font-size:.85rem;line-height:2'>
        1. Enter your Groq API key in the sidebar<br>
        2. Upload any PDF<br>
        3. Ask questions — watch the agent reason step by step
      </div>
    </div>

    <div style='max-width:500px;margin:2rem auto 0;background:#18181c;border:.5px solid #2a2a32;border-radius:12px;padding:1rem 1.25rem'>
      <div style='font-size:.75rem;font-weight:600;color:#72728a;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px'>What makes this different from DocChat</div>
      <div style='font-size:.82rem;color:#e8e8f0;line-height:1.8'>
        ✦ <b>Grader node</b> — checks if retrieved chunks actually answer the question<br>
        ✦ <b>Self-correction</b> — rephrases and retries if quality is poor<br>
        ✦ <b>Reasoning trace</b> — see every decision the agent made<br>
        ✦ <b>Graceful fallback</b> — refuses instead of hallucinating
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Chat history
    for msg in st.session_state.messages:
        render_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
            steps=msg.get("steps"),
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("Ask a question about your document…")

    if question:
        st.session_state.messages.append({"role":"user","content":question})

        with st.spinner("Agent thinking…"):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                initial = {
                    "question":           question,
                    "rephrased_question": "",
                    "route":              "",
                    "documents":          [],
                    "grade":              "",
                    "answer":             "",
                    "retry_count":        0,
                    "steps":              [],
                    "chat_history":       [],
                }
                result = st.session_state.graph.invoke(initial, config=config)

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["answer"],
                    "sources": result.get("documents", []),
                    "steps":   result.get("steps", []),
                })

            except Exception as e:
                err = str(e)
                if "auth" in err.lower() or "api_key" in err.lower():
                    msg = "Invalid Groq API key. Check the sidebar."
                elif "rate" in err.lower():
                    msg = "Rate limit hit. Wait a moment and try again."
                elif "decommission" in err.lower():
                    msg = "Model issue — refresh and try again."
                else:
                    msg = f"Error: {err}"
                st.session_state.messages.append({
                    "role":"assistant","content":f"⚠️ {msg}",
                    "sources":[],"steps":[]
                })

        st.rerun()
