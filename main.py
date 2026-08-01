"""
main.py
FastAPI wrapper around the LangGraph multi-agent pipeline.

Local run:   uvicorn main:app --reload
Render run:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_llms
from data_setup import setup_data, REPORT_PATH
from vectorstore import init_embedder, index_report, init_memory_collection, add_to_memory
from graph import build_graph
from state import new_state
from langchain_community.utilities import SQLDatabase
from data_setup import DB_PATH

app = FastAPI(title="Multi-Agent AI Analyst API")

# Allow your Vercel frontend to call this API. Tighten this to your exact
# Vercel URL once you have it (e.g. "https://your-app.vercel.app") instead
# of "*" if you want to lock it down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once at startup, reused across requests.
_graph_app = None


@app.on_event("startup")
def startup():
    global _graph_app
    llm_flash, llm_lite = get_llms()

    df = setup_data()
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    init_embedder()
    with open(REPORT_PATH) as f:
        report_text = f.read()
    embedding_dim = index_report(report_text)
    init_memory_collection(embedding_dim)

    _graph_app = build_graph(llm_flash, llm_lite, db, df)
    print("Startup complete — graph ready.")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    steps: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "graph_ready": _graph_app is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if _graph_app is None:
        raise HTTPException(status_code=503, detail="Graph is still starting up, try again in a moment.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    state = new_state(req.question)
    result = _graph_app.invoke(state, config={"recursion_limit": 25})
    add_to_memory(req.question, result["answer"])
    return AskResponse(answer=result["answer"], steps=result["steps"])
