from typing import Annotated, TypedDict

from pydantic import BaseModel


def _append_warnings(existing: list[str] | None, new: list[str]) -> list[str]:
    return (existing or []) + new


class ResearchState(TypedDict, total=False):
    query: str

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


class ResearchResponse(BaseModel):
    query: str
    search_findings: str
    paper_findings: str
    benchmark_findings: str
    fact_check_report: str
    final_report: str
    warnings: list[str] = []
