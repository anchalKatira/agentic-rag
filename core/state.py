"""LangGraph state shared by every node."""

from typing import List, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    rephrased_question: str
    route: str
    documents: List[Document]
    grade: str
    answer: str
    retry_count: int
    steps: List[str]
    chat_history: List[dict]
