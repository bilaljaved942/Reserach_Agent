import logging

from app.history import format_history
from app.llm import LLMCallError, call_llm, get_llm
from app.schemas import ResearchState

logger = logging.getLogger(__name__)

MANAGER_SYSTEM_PROMPT = """You are the Research Manager in a multi-agent research pipeline.
Given a user's research question (and, if provided, the prior turns of this conversation), \
break the NEW question into three focused, self-contained sub-questions, one for each \
specialist agent that will investigate in parallel:

- SEARCH: a question for a Search Agent gathering general web/background context and current events
- PAPER: a question for a Paper Agent focused on academic literature, methods, and findings
- BENCHMARK: a question for a Benchmark Agent focused on quantitative comparisons and leaderboard-style results

The specialist agents will NOT see the conversation history themselves — only your sub-questions.
So if the new question uses a reference that depends on prior turns (e.g. "compare that with X",
"what about the second one", "and for images?"), resolve the reference yourself using the
conversation history and write self-contained sub-questions that make sense on their own.

Respond with EXACTLY three lines, no extra commentary, in this format:
SEARCH: <query>
PAPER: <query>
BENCHMARK: <query>
"""


def run_manager(state: ResearchState) -> dict:
    query = state["query"]
    history_text = format_history(state.get("history"))

    try:
        llm = get_llm()
        content = call_llm(
            llm,
            [
                ("system", MANAGER_SYSTEM_PROMPT),
                ("human", f"Conversation so far:\n{history_text}\n\nNew question: {query}"),
            ],
        )
    except LLMCallError as exc:
        # No sub-questions to work with — fall back to giving every specialist the raw query
        # verbatim instead of failing the whole request over a decomposition step.
        logger.warning("Manager LLM call failed, falling back to raw query: %s", exc)
        return {
            "search_query": query,
            "paper_query": query,
            "benchmark_query": query,
            "warnings": [f"Manager: could not decompose query, used raw query as-is ({exc})"],
        }

    search_q = paper_q = benchmark_q = query
    for line in content.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("SEARCH:"):
            search_q = line.split(":", 1)[1].strip()
        elif upper.startswith("PAPER:"):
            paper_q = line.split(":", 1)[1].strip()
        elif upper.startswith("BENCHMARK:"):
            benchmark_q = line.split(":", 1)[1].strip()

    return {
        "search_query": search_q,
        "paper_query": paper_q,
        "benchmark_query": benchmark_q,
    }
