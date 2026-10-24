"""Prompt templates used by the agent nodes."""

from langchain_core.prompts import ChatPromptTemplate

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are grading document chunks for relevance.

Grade 'good' if the chunks contain ANY of these:
- The topic or concept mentioned in the question
- Related technical terms
- Partial information about the subject
- Background information that helps answer

Grade 'bad' ONLY if chunks are about a completely different subject with zero overlap.
DEFAULT to 'good' unless chunks are totally irrelevant."""),
    ("human", "Question: {question}\n\nChunks:\n{context}"),
])

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """Rewrite the question using different technical keywords
for better document search. Return ONLY the rewritten question, 5-10 words."""),
    ("human", "Original: {question}\n\nRewritten:"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """Answer using the provided document context.
Use the context as your PRIMARY source.
If context partially answers, give what you can and note gaps.
Do NOT say the information is not available if context is partially relevant.

Context:
{context}"""),
    ("human", "{question}"),
])

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    ("human", "{question}"),
])
