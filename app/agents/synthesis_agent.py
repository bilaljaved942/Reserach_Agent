import logging

from app.llm import LLMCallError, call_llm, get_llm
from app.schemas import ResearchState

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are the Synthesis Agent in a multi-agent research pipeline.
You receive the original research question, findings from three specialist agents (Search, \
Paper, Benchmark), and a Fact Checker report that flags corroborated claims, contradictions, \
and unverified claims.

Write the final report as clean Markdown with these sections:
1. Summary — a short, direct answer to the research question
2. Background — from the Search Agent's findings
3. Academic Perspective — from the Paper Agent's findings
4. Benchmark Comparison — from the Benchmark Agent's findings
5. Caveats & Confidence — incorporate the Fact Checker's flags; be explicit about what is \
corroborated vs. speculative vs. contradictory, and remind the reader this system has no live \
data access so findings need independent verification

Be concise and well-organized. Do not invent information beyond what was provided.
"""


def _raw_concatenation_fallback(state: ResearchState) -> str:
    return (
        "# Research Report (auto-synthesis unavailable)\n\n"
        "The Synthesis Agent could not run, so this is a raw concatenation of unedited agent "
        "output rather than a written report.\n\n"
        f"## Question\n{state['query']}\n\n"
        f"## Search Agent Findings\n{state.get('search_findings', 'N/A')}\n\n"
        f"## Paper Agent Findings\n{state.get('paper_findings', 'N/A')}\n\n"
        f"## Benchmark Agent Findings\n{state.get('benchmark_findings', 'N/A')}\n\n"
        f"## Fact Check Report\n{state.get('fact_check_report', 'N/A')}\n"
    )


def run_synthesis_agent(state: ResearchState) -> dict:
    content = (
        f"Research question: {state['query']}\n\n"
        f"Search Agent findings:\n{state.get('search_findings', '')}\n\n"
        f"Paper Agent findings:\n{state.get('paper_findings', '')}\n\n"
        f"Benchmark Agent findings:\n{state.get('benchmark_findings', '')}\n\n"
        f"Fact Checker report:\n{state.get('fact_check_report', '')}"
    )

    try:
        llm = get_llm()
        report = call_llm(
            llm,
            [
                ("system", SYNTHESIS_SYSTEM_PROMPT),
                ("human", content),
            ],
        )
    except LLMCallError as exc:
        logger.warning("Synthesis agent LLM call failed: %s", exc)
        return {
            "final_report": _raw_concatenation_fallback(state),
            "warnings": [f"Synthesis Agent: fell back to raw concatenation ({exc})"],
        }

    return {"final_report": report}
