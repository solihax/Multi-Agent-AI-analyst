"""
agents.py
The four specialist agents: retriever, web, data (SQL), code.

Each "make_*_agent" factory closes over the objects it needs (llm, df, db)
and returns a plain function of (state) -> dict, which is what LangGraph
expects as a node.
"""

import io
import contextlib
import threading

from langchain_community.utilities import SQLDatabase
from tavily import TavilyClient

from state import AgentState, get_llm_text
import vectorstore as vs


# ---------- Retriever agent ----------

def retriever_agent(state: AgentState) -> dict:
    query_vector = vs.embedder.embed_query(state["question"])
    results = vs.qdrant.query_points(collection_name=vs.DOCS_COLLECTION, query=query_vector, limit=3)
    retrieved_texts = [r.payload["text"] for r in results.points]
    return {"documents": retrieved_texts, "steps": state["steps"] + ["retriever"]}


# ---------- Web agent ----------

def web_agent(state: AgentState) -> dict:
    import os
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return {"documents": state["documents"], "steps": state["steps"] + ["web(skipped-no-key)"]}
    client = TavilyClient(api_key=key)
    hits = client.search(state["question"])["results"]
    web_texts = [h["content"] for h in hits]
    return {"documents": state["documents"] + web_texts, "steps": state["steps"] + ["web"]}


# ---------- Data (SQL) agent ----------

def make_data_agent(db: SQLDatabase, llm_lite):
    def data_agent(state: AgentState) -> dict:
        schema = db.get_table_info()
        prompt = (
            f"Schema:\n{schema}\n\nWrite ONE SQLite query to answer: {state['question']}\n"
            f"Return ONLY the SQL query, nothing else. No markdown, no explanation."
        )
        sql = get_llm_text(llm_lite.invoke(prompt)).replace("```sql", "").replace("```", "").strip()
        if not sql.lower().startswith("select"):
            return {"sql_result": "Rejected: only SELECT queries are allowed.",
                     "steps": state["steps"] + ["data(sql-rejected)"]}
        try:
            result = db.run(sql)
        except Exception as e:
            result = f"SQL execution error: {e}"
        return {"sql_result": f"Query: {sql}\nResult: {result}", "steps": state["steps"] + ["data(sql)"]}
    return data_agent


# ---------- Code agent ----------

class TimeoutException(Exception):
    pass


def run_python_with_timeout(code: str, df, pd, timeout_seconds: int = 5) -> str:
    """Thread-based timeout — works on Windows, unlike signal.alarm/SIGALRM
    which is Unix-only. Note: if the code truly hangs, the background thread
    keeps running (daemon=True so it won't block process exit), but we stop
    waiting for it and return a timeout message."""
    output = io.StringIO()
    error_holder = {}

    def target():
        try:
            with contextlib.redirect_stdout(output):
                exec(code, {"__builtins__": __builtins__, "df": df, "pd": pd})
        except Exception as e:
            error_holder["error"] = str(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        return "Error: code execution timed out"
    if "error" in error_holder:
        return f"Error: {error_holder['error']}"
    return output.getvalue().strip() or "(code ran but printed nothing)"


def make_code_agent(df, llm_lite):
    import pandas as pd

    def code_agent(state: AgentState) -> dict:
        prompt = (
            f"You have a pandas DataFrame called `df` with real telecom customer data. "
            f"Columns: {list(df.columns)}\nWrite Python code to answer: {state['question']}\n"
            f"Use print() to output the final result. Return ONLY the Python code, no markdown, no explanation."
        )
        code = get_llm_text(llm_lite.invoke(prompt)).replace("```python", "").replace("```", "").strip()
        result = run_python_with_timeout(code, df, pd)
        return {"code_result": f"Code:\n{code}\n\nOutput:\n{result}", "steps": state["steps"] + ["code"]}
    return code_agent
