"""
Agentic RAG — Streamlit UI
==========================
Self-correcting document Q&A:
- PDF upload and ingestion into a persistent ChromaDB store
- LangGraph agent (router → retriever → grader → answer/rephrase/fallback)
- Reasoning trace panel + source chunk attribution
- Previously ingested PDFs are reloaded from disk (no re-embedding)

Logic lives in core/ (agent, ingestion), db/ (ChromaDB) and evaluation/.
"""

import uuid

import streamlit as st

from core.agent import build_graph, run_agent
from core.config import CHROMA_PERSIST_DIR, TOP_K, get_api_key
from core.ingestion import ingest_pdf
from db.chroma_store import ChromaStore

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

/* Status pills */
.pill{display:inline-flex;align-items:center;gap:5px;font-size:.75rem;padding:3px 9px;border-radius:20px;border:.5px solid var(--border);background:var(--surface);color:var(--muted)}
.dot{width:6px;height:6px;border-radius:50%}
.dot-green{background:var(--green);box-shadow:0 0 5px var(--green)}

/* Divider */
.divider{height:1px;background:var(--border);margin:.9rem 0}

.stButton>button{background:var(--surface);border:.5px solid var(--border);color:var(--text);border-radius:8px;font-family:'DM Sans',sans-serif;transition:all .15s}
.stButton>button:hover{border-color:var(--accent);color:var(--accent2)}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_store() -> ChromaStore:
    """One ChromaDB client per server process (persisted under CHROMA_PERSIST_DIR)."""
    return ChromaStore()


store = get_store()

# ─────────────────────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────────────────────
NODE_ICONS = {
    "Router": "🔀",
    "Retriever": "🔍",
    "Grader": "⚖️",
    "Rephrase": "✏️",
    "Answer": "✅",
    "Fallback": "⚠️",
    "Direct": "💬",
}


def _is_retry(step: str) -> bool:
    s = step.lower()
    return "attempt 2" in s or "attempt 3" in s


def render_message(role: str, content: str, sources=None, steps=None):
    label = "You" if role == "user" else "Agent"
    cls = "user" if role == "user" else "bot"
    st.markdown(f"""
    <div class="msg-wrap">
      <div class="msg-role {cls}">{label}</div>
      <div class="msg-bubble {cls}">{content}</div>
    </div>
    """, unsafe_allow_html=True)

    if steps and role == "assistant":
        has_retry = any(_is_retry(s) for s in steps)
        has_fallback = any("fallback" in s.lower() for s in steps)
        extra = " ↺ self-corrected" if has_retry else (" ⚠ fallback" if has_fallback else "")
        with st.expander(f"🧠 Reasoning trace ({len(steps)} steps){extra}"):
            for i, step in enumerate(steps):
                parts = step.split("→", 1)
                node_name = parts[0].strip().split("(")[0].strip()
                detail = parts[1].strip() if len(parts) > 1 else step
                icon = NODE_ICONS.get(node_name, "▸")
                retry_tag = " ↺ retry" if _is_retry(step) else ""
                st.markdown(
                    f"**{i + 1}. {icon} {node_name}**{retry_tag}  \n"
                    f"<span style='color:#72728a;font-size:.85rem'>{detail}</span>",
                    unsafe_allow_html=True,
                )
                if i < len(steps) - 1:
                    st.markdown("---")

    if sources and role == "assistant":
        with st.expander(f"📎 {len(sources)} source chunk(s) used"):
            for i, doc in enumerate(sources):
                pg = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i + 1} · page {pg}**")
                st.code(doc.page_content.strip()[:350], language=None)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],  # [{role, content, sources, steps}]
    "graph": None,
    "vs": None,
    "pdf_name": None,
    "pdf_pages": 0,
    "pdf_chunks": 0,
    "doc_id": None,
    "thread_id": str(uuid.uuid4()),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def activate_document(vs, doc_id, name, pages, chunks, api_key):
    """Point the session at an ingested document and build a fresh agent for it."""
    st.session_state.vs = vs
    st.session_state.graph = build_graph(vs, api_key)
    st.session_state.doc_id = doc_id
    st.session_state.pdf_name = name
    st.session_state.pdf_pages = pages
    st.session_state.pdf_chunks = chunks
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Agentic RAG")
    st.caption("Self-correcting document Q&A with LangGraph + ChromaDB")
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
                res = ingest_pdf(uploaded.read(), uploaded.name, store)
                activate_document(res.vectorstore, res.doc_id, res.filename,
                                  res.pages, res.chunks, api_key)
                st.session_state.just_reused = res.reused
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    elif uploaded and not api_key:
        st.warning("Enter API key first.")

    # Previously ingested documents (persisted in ChromaDB)
    saved = store.list_documents()
    if saved:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🗄️ Saved documents")
        labels = {d["doc_id"]: f'{d["filename"]} · {d["chunks"]} chunks' for d in saved}
        choice = st.selectbox("Saved documents", list(labels), format_func=labels.get,
                              label_visibility="collapsed")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Load", use_container_width=True, disabled=not api_key):
                d = next(x for x in saved if x["doc_id"] == choice)
                activate_document(store.load(choice), choice, d["filename"],
                                  d["pages"], d["chunks"], api_key)
                st.rerun()
        with b2:
            if st.button("Delete", use_container_width=True):
                store.delete_document(choice)
                if st.session_state.doc_id == choice:
                    for k in ["messages", "graph", "vs", "pdf_name", "pdf_pages",
                              "pdf_chunks", "doc_id"]:
                        st.session_state[k] = defaults[k]
                    st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

    # Doc stats
    if st.session_state.pdf_name:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 Document")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.pdf_pages)
        c2.metric("Chunks", st.session_state.pdf_chunks)
        st.caption(st.session_state.pdf_name)
        if st.session_state.pop("just_reused", False):
            st.caption("♻️ Loaded from ChromaDB — no re-embedding")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # How it works
    st.markdown("#### ⚙️ Agent Architecture")
    st.markdown(f"""
    <div style="font-size:.78rem;color:#72728a;line-height:1.8">
    1. <b style="color:#a78bfa">Router</b> — always retrieve from your PDF<br>
    2. <b style="color:#34d399">Retriever</b> — ChromaDB top-{TOP_K} chunks<br>
    3. <b style="color:#fbbf24">Grader</b> — are chunks relevant?<br>
    4. <b style="color:#60a5fa">Rephrase</b> — if not, try better query<br>
    5. <b style="color:#34d399">Answer</b> — grounded final response<br>
    6. <b style="color:#f87171">Fallback</b> — answer from best chunks found
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Vector store: `{CHROMA_PERSIST_DIR}`")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
    with col2:
        if st.button("📄 New PDF", use_container_width=True):
            for k in ["messages", "graph", "vs", "pdf_name", "pdf_pages",
                      "pdf_chunks", "doc_id"]:
                st.session_state[k] = defaults[k]
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
st.markdown("# Agentic RAG")
st.markdown("*Self-correcting document Q&A — powered by LangGraph*")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if not st.session_state.graph:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;color:#72728a'>
      <div style='font-size:2.5rem;margin-bottom:.75rem'>🤖</div>
      <div style='font-size:1.15rem;color:#e8e8f0;margin-bottom:.5rem;font-weight:500'>
        Upload a PDF to start
      </div>
      <div style='font-size:.85rem;line-height:2'>
        1. Enter your Groq API key in the sidebar<br>
        2. Upload any PDF (or load a saved one)<br>
        3. Ask questions — watch the agent reason step by step
      </div>
    </div>
    <div style='max-width:500px;margin:2rem auto 0;background:#18181c;border:.5px solid #2a2a32;border-radius:12px;padding:1rem 1.25rem'>
      <div style='font-size:.75rem;font-weight:600;color:#72728a;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px'>What makes this different from DocChat</div>
      <div style='font-size:.82rem;color:#e8e8f0;line-height:1.8'>
        ✦ <b>Grader node</b> — checks if retrieved chunks actually answer the question<br>
        ✦ <b>Self-correction</b> — rephrases and retries if quality is poor<br>
        ✦ <b>Reasoning trace</b> — see every decision the agent made<br>
        ✦ <b>Persistent ChromaDB</b> — ingest once, reuse after restarts
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
            steps=msg.get("steps"),
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    question = st.chat_input("Ask a question about your document…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Agent thinking…"):
            try:
                result = run_agent(st.session_state.graph, question,
                                   st.session_state.thread_id)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("documents", []),
                    "steps": result.get("steps", []),
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
                    "role": "assistant", "content": f"⚠️ {msg}",
                    "sources": [], "steps": [],
                })
        st.rerun()
