"""
graph.py
Supervisor (router), critic (verifier), generate node, and the LangGraph
StateGraph that wires all agents together.

IMPORTANT: redefining a node function does not update an already-compiled
graph. If you change any node here, re-run build_graph() to get a fresh
compiled app.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from state import AgentState, get_llm_text
from vectorstore import get_relevant_memory
from agents import retriever_agent, web_agent, make_data_agent, make_code_agent

MAX_REVISIONS = 2


class Route(BaseModel):
    next: Literal["retriever", "web", "data", "code", "finish"] = Field(description="Next agent to run")


class Verdict(BaseModel):
    ok: bool = Field(description="True if answer is correct AND fully supported by evidence")
    reason: str = Field(description="Brief explanation")


def make_supervisor(llm_flash):
    """Memory-aware supervisor: resolves follow-up references ('that', 'it')
    using past conversation turns, and force-routes to the retriever for
    report/analysis-style questions on the first pass.

    Guards:
    1. The LLM is not allowed to pick 'finish' before any specialist agent
       has run — prevents empty-evidence answers.
    2. Hard cap on specialist calls — prevents infinite supervisor<->agent
       loops if the LLM keeps bouncing between agents without ever
       choosing finish (this caused a GraphRecursionError in testing)."""

    MAX_SPECIALIST_CALLS = 4

    def specialist_ran(steps: list[str]) -> bool:
        return any(
            s == "retriever" or s.startswith("web") or s.startswith("data(") or s == "code"
            for s in steps
        )

    def specialist_call_count(steps: list[str]) -> int:
        return sum(1 for s in steps if s.startswith("supervisor→") and "finish" not in s)

    def supervisor(state: AgentState) -> dict:
        q_lower = state["question"].lower()
        already_ran = " ".join(state["steps"])
        past_context = get_relevant_memory(state["question"])

        if any(kw in q_lower for kw in ["report", "analysis", "why", "reason"]) and "retriever" not in already_ran:
            return {"plan": "retriever", "steps": state["steps"] + ["supervisor→retriever(forced)"]}

        if specialist_call_count(state["steps"]) >= MAX_SPECIALIST_CALLS:
            return {"plan": "finish", "steps": state["steps"] + ["supervisor→finish(cap-reached)"]}

        prompt = (
            f"Relevant past conversation (may be empty): {past_context}\n\n"
            f"Question: {state['question']}\n"
            f"Agents already run: {state['steps']}\n"
            f"Documents so far: {len(state['documents'])} chunk(s)\n"
            f"SQL result: {state['sql_result']}\nCode result: {state['code_result']}\n\n"
            f"Decide next agent: retriever, web, data, code, or finish. "
            f"Use past conversation to resolve references like 'that', 'it', or follow-ups."
        )
        decision = llm_flash.with_structured_output(Route).invoke(prompt)
        next_step = decision.next

        if next_step == "finish" and not specialist_ran(state["steps"]):
            # Guard tripped: no evidence gathered yet, don't allow finish.
            # Cheap heuristic fallback since we can't trust the LLM here.
            if any(kw in q_lower for kw in ["how many", "count", "average", "total", "number of"]):
                next_step = "data"
            elif any(kw in q_lower for kw in ["calculate", "compute", "percentage", "correlation"]):
                next_step = "code"
            else:
                next_step = "retriever"
            return {"plan": next_step, "steps": state["steps"] + [f"supervisor→{next_step}(guard-override)"]}

        return {"plan": next_step, "steps": state["steps"] + [f"supervisor→{next_step}"]}

    return supervisor


def make_critic(llm_flash):
    def critic(state: AgentState) -> dict:
        no_evidence = not state["documents"] and not state["sql_result"] and not state["code_result"]

        prompt = (
            f"Question: {state['question']}\n\nEvidence:\n- Documents: {state['documents']}\n"
            f"- SQL result: {state['sql_result']}\n- Code result: {state['code_result']}\n\n"
            f"Drafted answer: {state['answer']}\n\nIs this correct AND fully supported? "
            f"Be VERY strict: any fact not in the evidence = NOT ok. "
            f"If the evidence sections above are all empty and the answer just says information "
            f"is missing, that is NOT ok — it means the wrong agent (or no agent) ran, not that "
            f"the question is unanswerable."
        )
        verdict = llm_flash.with_structured_output(Verdict).invoke(prompt)
        ok = verdict.ok and not no_evidence
        return {
            "revisions": state["revisions"] + (0 if ok else 1),
            "steps": state["steps"] + [f"critic({'approved' if ok else 'revise: ' + verdict.reason})"],
        }
    return critic


def make_generate_answer(llm_lite):
    def generate_answer(state: AgentState) -> dict:
        prompt = (
            f"Question: {state['question']}\n\nEvidence (ONLY source you may use):\n"
            f"- Documents: {state['documents']}\n- SQL result: {state['sql_result']}\n"
            f"- Code result: {state['code_result']}\n\nAnswer using ONLY facts literally in the evidence. "
            f"Do not add anything not explicitly stated."
        )
        return {"answer": get_llm_text(llm_lite.invoke(prompt)), "steps": state["steps"] + ["generate"]}
    return generate_answer


def route_after_supervisor(state):
    return state["plan"]


def route_after_critic(state):
    last = state["steps"][-1]
    return "finish" if "approved" in last or state["revisions"] >= MAX_REVISIONS else "revise"


def build_graph(llm_flash, llm_lite, db, df):
    """Assembles and compiles the full StateGraph. Call this again any time
    an agent or node function changes."""

    supervisor = make_supervisor(llm_flash)
    critic = make_critic(llm_flash)
    generate_answer = make_generate_answer(llm_lite)
    data_agent = make_data_agent(db, llm_lite)
    code_agent = make_code_agent(df, llm_lite)

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("web", web_agent)
    graph.add_node("data", data_agent)
    graph.add_node("code", code_agent)
    graph.add_node("generate", generate_answer)
    graph.add_node("critic", critic)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {"retriever": "retriever", "web": "web", "data": "data", "code": "code", "finish": "generate"},
    )
    for a in ["retriever", "web", "data", "code"]:
        graph.add_edge(a, "supervisor")
    graph.add_edge("generate", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"finish": END, "revise": "supervisor"})

    app = graph.compile()
    print("Graph compiled")
    return app