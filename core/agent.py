"""LangGraph agent: router -> retriever -> grader -> answer / rephrase / fallback."""

from __future__ import annotations

from typing import List, Literal

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from core.config import MAX_RETRIES, MODEL, TOP_K
from core.prompts import DIRECT_PROMPT, GRADER_PROMPT, RAG_PROMPT, REPHRASE_PROMPT
from core.state import AgentState


class GradeDecision(BaseModel):
    grade: Literal["good", "bad"] = Field(description="relevance grade")
    reason: str = Field(description="reason")


def _format_context(docs: List[Document], limit: int = 0) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]: "
        f"{d.page_content.strip()[:limit] if limit else d.page_content.strip()}"
        for d in docs
    )


def build_graph(vectorstore, api_key: str):
    """Build the agentic RAG graph for a given (Chroma) vectorstore."""
    llm = ChatGroq(model=MODEL, temperature=0, groq_api_key=api_key)

    grader_chain = GRADER_PROMPT | llm.with_structured_output(GradeDecision)
    rephrase_chain = REPHRASE_PROMPT | llm

    # ── Nodes ────────────────────────────────────────────────
    def router_node(state: AgentState) -> dict:
        # Always retrieve — user uploaded the PDF to ask about it
        step = "Router → retrieve (document Q&A mode)"
        return {
            "route": "retrieve",
            "retry_count": 0,
            "steps": state.get("steps", []) + [step],
        }

    def retriever_node(state: AgentState) -> dict:
        rc = state.get("retry_count", 0)
        query = (
            state["rephrased_question"]
            if rc > 0 and state.get("rephrased_question")
            else state["question"]
        )
        src = "rephrased" if rc > 0 else "original"
        docs = vectorstore.similarity_search(query, k=TOP_K)
        pages = [str(d.metadata.get("page", "?")) for d in docs]
        step = (
            f"Retriever (attempt {rc + 1}, {src}) → "
            f"{len(docs)} chunks from pages {', '.join(pages)}"
        )
        return {"documents": docs, "steps": state.get("steps", []) + [step]}

    def grader_node(state: AgentState) -> dict:
        docs = state["documents"]
        if not docs:
            return {
                "grade": "bad",
                "steps": state.get("steps", []) + ["Grader → 'bad' | no documents retrieved"],
            }
        ctx = "\n\n".join(
            f"[Chunk {i + 1}, Page {d.metadata.get('page', '?')}]:\n{d.page_content[:400]}"
            for i, d in enumerate(docs)
        )
        d = grader_chain.invoke({"question": state["question"], "context": ctx})
        step = f"Grader → '{d.grade}' | {d.reason}"
        return {"grade": d.grade, "steps": state.get("steps", []) + [step]}

    def rephrase_node(state: AgentState) -> dict:
        rc = state.get("retry_count", 0)
        new_q = rephrase_chain.invoke({"question": state["question"]}).content.strip()
        step = f"Rephrase (attempt {rc + 1}/{MAX_RETRIES}) → '{new_q}'"
        return {
            "rephrased_question": new_q,
            "retry_count": rc + 1,
            "steps": state.get("steps", []) + [step],
        }

    def answer_node(state: AgentState) -> dict:
        docs = state.get("documents", [])
        if docs:
            resp = (RAG_PROMPT | llm).invoke(
                {"context": _format_context(docs), "question": state["question"]}
            )
            mode = "RAG"
        else:
            resp = (DIRECT_PROMPT | llm).invoke({"question": state["question"]})
            mode = "Direct"

        answer = resp.content
        history = state.get("chat_history", []) + [{
            "question": state["question"],
            "answer": answer,
            "sources": docs,
            "steps": state.get("steps", []),
        }]
        step = f"Answer ({mode}) → {len(answer)} chars"
        return {
            "answer": answer,
            "chat_history": history,
            "steps": state.get("steps", []) + [step],
        }

    def fallback_node(state: AgentState) -> dict:
        # Retries exhausted — still answer from whatever chunks we have
        docs = state.get("documents", [])
        if docs:
            resp = (RAG_PROMPT | llm).invoke(
                {"context": _format_context(docs), "question": state["question"]}
            )
            answer = resp.content
            mode = "RAG fallback"
        else:
            answer = (
                f"I couldn't find specific information about "
                f"'{state['question']}' in the document."
            )
            mode = "no chunks"

        history = state.get("chat_history", []) + [{
            "question": state["question"],
            "answer": answer,
            "sources": docs,
            "steps": state.get("steps", []),
        }]
        step = f"Fallback ({mode}) → answered with available context"
        return {
            "answer": answer,
            "chat_history": history,
            "steps": state.get("steps", []) + [step],
        }

    # ── Conditional edge ─────────────────────────────────────
    def grade_decision(state: AgentState) -> str:
        if state.get("grade") == "good":
            return "answer"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rephrase"
        return "fallback"

    # ── Assemble ─────────────────────────────────────────────
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("grader", grader_node)
    g.add_node("rephrase", rephrase_node)
    g.add_node("answer", answer_node)
    g.add_node("fallback", fallback_node)

    g.set_entry_point("router")
    g.add_edge("router", "retriever")
    g.add_edge("retriever", "grader")
    g.add_conditional_edges(
        "grader",
        grade_decision,
        {"answer": "answer", "rephrase": "rephrase", "fallback": "fallback"},
    )
    g.add_edge("rephrase", "retriever")
    g.add_edge("answer", END)
    g.add_edge("fallback", END)

    return g.compile(checkpointer=MemorySaver())


def run_agent(graph, question: str, thread_id: str) -> AgentState:
    """Invoke the graph for one question. Shared by the Streamlit app and evaluation."""
    initial: AgentState = {
        "question": question,
        "rephrased_question": "",
        "route": "",
        "documents": [],
        "grade": "",
        "answer": "",
        "retry_count": 0,
        "steps": [],
        "chat_history": [],
    }
    return graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
