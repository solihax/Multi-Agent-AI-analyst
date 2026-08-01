"""
state.py
Shared graph state and the LLM response helper used everywhere.
"""

from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    question: str
    plan: str
    documents: List[str]
    sql_result: Optional[str]
    code_result: Optional[str]
    answer: str
    steps: List[str]
    revisions: int


def new_state(question: str) -> AgentState:
    """Convenience factory so call sites don't repeat the same 7 empty fields."""
    return {
        "question": question,
        "plan": "",
        "documents": [],
        "sql_result": None,
        "code_result": None,
        "answer": "",
        "steps": [],
        "revisions": 0,
    }


def get_llm_text(response) -> str:
    """Gemini sometimes returns .content as a list of dicts instead of a plain
    string — this normalizes both cases."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
        return "".join(texts).strip()
    return str(content).strip()