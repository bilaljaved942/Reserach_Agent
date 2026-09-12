import logging
import threading

from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.config import (
    GEMINI_API_KEY,
    LLM_MAX_CONCURRENCY,
    LLM_MAX_RETRY_ATTEMPTS,
    LLM_REQUEST_TIMEOUT_SECONDS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

# There is no second model to fall back to, so a burst of parallel agent calls (manager fans
# out to 3 specialists at once) must not itself trigger a rate-limit error. This caps how many
# Gemini calls are in flight at the same time, independent of how many agents are scheduled.
_llm_semaphore = threading.Semaphore(LLM_MAX_CONCURRENCY)

_PERMANENT_ERROR_MARKERS = (
    "api key",
    "api_key",
    "invalid_argument",
    "permission",
    "unauthorized",
    "401",
    "403",
)


class LLMCallError(RuntimeError):
    """Raised when an LLM call still fails after all retries are exhausted.

    There is no fallback model to hand the request to — callers (agents) must decide how to
    degrade: skip gracefully with a placeholder, or let it propagate if the step is essential.
    """


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_output_tokens=2048,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )


def _is_retryable(exc: BaseException) -> bool:
    # Auth/bad-request errors won't be fixed by retrying, so fail fast instead of burning the
    # whole retry budget on something that will never succeed. Everything else (timeouts, 429
    # rate limits, transient 5xx) is treated as retryable.
    message = str(exc).lower()
    return not any(marker in message for marker in _PERMANENT_ERROR_MARKERS)


@retry(
    stop=stop_after_attempt(LLM_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential_jitter(initial=1, max=15),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _invoke_with_retry(llm, messages):
    with _llm_semaphore:
        return llm.invoke(messages)


def _extract_text(content) -> str:
    # Some models/providers return a plain string; others (e.g. newer Gemini models) return a
    # list of content blocks like [{"type": "text", "text": "..."}], mirroring Anthropic's
    # content-block format. Normalize both to plain text so callers never have to care.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def call_llm(llm: ChatGoogleGenerativeAI, messages: list) -> str:
    """Invoke the LLM with retry/backoff/timeout/concurrency control applied.

    Raises LLMCallError if every attempt fails. Callers decide what "no response" means for
    their step (see each agent's try/except around this call).
    """
    try:
        response = _invoke_with_retry(llm, messages)
    except Exception as exc:
        logger.error("LLM call failed after %d attempt(s): %s", LLM_MAX_RETRY_ATTEMPTS, exc)
        raise LLMCallError(str(exc)) from exc

    usage = getattr(response, "usage_metadata", None)
    if usage:
        logger.info("LLM call token usage: %s", usage)

    return _extract_text(response.content)
