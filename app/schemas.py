from typing import Annotated, TypedDict

from pydantic import BaseModel


def _append_warnings(existing: list[str] | None, new: list[str]) -> list[str]:
    return (existing or []) + new


class HistoryTurn(BaseModel):
    query: str
    answer: str


class ResearchState(TypedDict, total=False):
    query: str
    # Prior turns in this conversation, oldest first. Only the Manager and Synthesis agents see
    # this (see app/agents/manager.py and synthesis_agent.py) — the specialists work off the
    # Manager's already-contextualized sub-questions instead of the raw history.
    history: list[dict]

    search_query: str
    paper_query: str
    benchmark_query: str

    search_findings: str
    paper_findings: str
    benchmark_findings: str

    fact_check_report: str
    final_report: str

    # Reducer: each agent that degrades to a fallback appends its own warning here instead of
    # overwriting, so warnings from parallel agents (search/paper/benchmark) all survive.
    warnings: Annotated[list[str], _append_warnings]


class ResearchRequest(BaseModel):
    query: str
    # Prior turns in this conversation, oldest first. Optional — omit for a standalone query.
    history: list[HistoryTurn] = []


class ResearchResponse(BaseModel):
    query: str
    search_findings: str
    paper_findings: str
    benchmark_findings: str
    fact_check_report: str
    final_report: str
    warnings: list[str] = []
