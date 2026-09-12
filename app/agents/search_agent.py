import logging

from app.llm import LLMCallError, call_llm, get_llm
from app.schemas import ResearchState

logger = logging.getLogger(__name__)

SEARCH_AGENT_SYSTEM_PROMPT = """You are the Search Agent in a multi-agent research pipeline.
You do NOT have live internet access in this version of the system — answer using your own \
knowledge, as if summarizing what a web search would surface: general context, background, \
notable events, and common understanding of the topic.

Rules:
- Be concise and use bullet points.
- Explicitly flag anything that may be outdated relative to your knowledge cutoff, or that you \
are not confident about.
- Do not fabricate specific URLs, dates, or statistics you are not sure of.
"""


def run_search_agent(state: ResearchState) -> dict:
    try:
        llm = get_llm()
        content = call_llm(
            llm,
            [
                ("system", SEARCH_AGENT_SYSTEM_PROMPT),
                ("human", state["search_query"]),
            ],
        )
    except LLMCallError as exc:
        logger.warning("Search agent LLM call failed: %s", exc)
        return {
            "search_findings": f"Findings unavailable — Search Agent failed after retries ({exc}).",
            "warnings": [f"Search Agent: no findings available ({exc})"],
        }

    return {"search_findings": content}
