import logging

from app.llm import LLMCallError, call_llm, get_llm
from app.schemas import ResearchState

logger = logging.getLogger(__name__)

BENCHMARK_AGENT_SYSTEM_PROMPT = """You are the Benchmark Agent in a multi-agent research pipeline.
You do NOT have live access to benchmark leaderboards or datasets in this version of the system \
— answer using your own knowledge of how relevant approaches tend to compare quantitatively.

Rules:
- Focus on quantitative comparisons: metrics, relative performance, and known trade-offs between \
approaches.
- Any specific number you give must be flagged as approximate/from training knowledge and in \
need of independent verification against a live leaderboard.
- Be concise and use bullet points or a small comparison table.
- If you are not aware of solid quantitative data for this question, say so plainly instead of \
guessing.
"""


def run_benchmark_agent(state: ResearchState) -> dict:
    try:
        llm = get_llm()
        content = call_llm(
            llm,
            [
                ("system", BENCHMARK_AGENT_SYSTEM_PROMPT),
                ("human", state["benchmark_query"]),
            ],
        )
    except LLMCallError as exc:
        logger.warning("Benchmark agent LLM call failed: %s", exc)
        return {
            "benchmark_findings": f"Findings unavailable — Benchmark Agent failed after retries ({exc}).",
            "warnings": [f"Benchmark Agent: no findings available ({exc})"],
        }

    return {"benchmark_findings": content}
