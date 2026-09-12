import logging

from app.llm import LLMCallError, call_llm, get_llm
from app.schemas import ResearchState

logger = logging.getLogger(__name__)

PAPER_AGENT_SYSTEM_PROMPT = """You are the Paper Agent in a multi-agent research pipeline.
You do NOT have live access to academic databases (arXiv, Semantic Scholar, etc.) in this \
version of the system — answer using your own knowledge of the academic literature.

Rules:
- Focus on methods, findings, and notable papers relevant to the question.
- When you reference a specific paper, title, or author, explicitly note that it comes from \
your training knowledge and should be independently verified before being cited as fact.
- Be concise and use bullet points.
- Flag areas where the literature is unsettled or where your knowledge may be incomplete or \
outdated.
"""


def run_paper_agent(state: ResearchState) -> dict:
    try:
        llm = get_llm()
        content = call_llm(
            llm,
            [
                ("system", PAPER_AGENT_SYSTEM_PROMPT),
                ("human", state["paper_query"]),
            ],
        )
    except LLMCallError as exc:
        logger.warning("Paper agent LLM call failed: %s", exc)
        return {
            "paper_findings": f"Findings unavailable — Paper Agent failed after retries ({exc}).",
            "warnings": [f"Paper Agent: no findings available ({exc})"],
        }

    return {"paper_findings": content}
