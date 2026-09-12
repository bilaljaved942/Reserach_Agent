import logging

from app.llm import LLMCallError, call_llm, get_llm
from app.schemas import ResearchState

logger = logging.getLogger(__name__)

FACT_CHECKER_SYSTEM_PROMPT = """You are the Fact Checker in a multi-agent research pipeline.
You receive findings independently produced by three specialist agents (Search, Paper, \
Benchmark) for the same research question. None of them have live external data access, so \
every claim is a candidate, not a confirmed fact.

Your job:
1. Identify claims that are corroborated across two or more agents (note them as higher \
confidence, but still not verified).
2. Flag direct contradictions between agents.
3. Flag claims that are speculative, unsupported, or where the originating agent already \
expressed uncertainty.
4. Explicitly remind the reader that all findings should be independently verified against \
real sources before being relied upon.

Output a concise, structured report with short sections: Corroborated Claims, Contradictions, \
Unverified / Low-Confidence Claims, Verification Notes.
"""


def run_fact_checker(state: ResearchState) -> dict:
    content = (
        f"Research question: {state['query']}\n\n"
        f"Search Agent findings:\n{state.get('search_findings', '')}\n\n"
        f"Paper Agent findings:\n{state.get('paper_findings', '')}\n\n"
        f"Benchmark Agent findings:\n{state.get('benchmark_findings', '')}"
    )

    try:
        llm = get_llm()
        report = call_llm(
            llm,
            [
                ("system", FACT_CHECKER_SYSTEM_PROMPT),
                ("human", content),
            ],
        )
    except LLMCallError as exc:
        # Synthesis can still proceed on the raw findings — just without cross-checking.
        logger.warning("Fact checker LLM call failed: %s", exc)
        return {
            "fact_check_report": (
                f"Fact-checking unavailable — Fact Checker failed after retries ({exc}). "
                "The findings below have NOT been cross-checked for contradictions."
            ),
            "warnings": [f"Fact Checker: cross-check skipped ({exc})"],
        }

    return {"fact_check_report": report}
