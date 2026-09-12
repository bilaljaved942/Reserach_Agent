import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("RESEARCH_AGENT_MODEL", "gemini-3.5-flash-lite")

# Reliability knobs — see app/llm.py for how these are used.
LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRY_ATTEMPTS = int(os.environ.get("LLM_MAX_RETRY_ATTEMPTS", "3"))
LLM_MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "2"))
