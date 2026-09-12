# Research Agent

A multi-agent research assistant. You send it a research question; a team of LLM-backed agents,
orchestrated with LangGraph, gather findings, cross-check them against each other, and return a
single synthesized, cited-where-possible report.

Full problem statement and design rationale: [PROJECT_CHARTER.md](PROJECT_CHARTER.md).

## How it works

```
User query
    │
    ▼
Research Manager .......... breaks the query into a sub-question per specialist
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Search Agent   Paper Agent   Benchmark Agent      (run in parallel)
    │              │              │
    └──────────────┴──────────────┘
                   ▼
             Fact Checker ......... cross-checks the three sets of findings against each other
                   │
                   ▼
           Synthesis Agent ........ writes the final report
                   │
                   ▼
              Final Report
```

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`
  (`app/graph.py`). The graph's edges are what decide which agent runs after which — see the
  "Orchestration in detail" section below.
- **Agents**: each agent (`app/agents/*.py`) is a plain Python function that takes the shared
  state, calls the LLM with a role-specific system prompt, and returns its findings.
- **LLM**: [Gemini](https://ai.google.dev/) via `langchain-google-genai` (`app/llm.py`).
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (`app/main.py`), one endpoint in, one
  report out.

### v1 limitation — no external tools yet

Search/Paper/Benchmark agents currently answer from the model's own knowledge — there's no live
web search, arXiv lookup, or benchmark API wired in yet. Each is prompted to flag uncertainty and
staleness rather than assert unverified claims as current fact. Because of this, the Fact Checker
can only catch **contradictions between the three agents**, not verify claims against real
sources. Treat v1 output as a draft that needs independent verification, not a sourced report.
Adding a real tool to any agent later is a drop-in change — the graph shape doesn't need to
change.

## Project structure

```
app/
  main.py              FastAPI app: /health, /research
  graph.py             LangGraph StateGraph wiring the agents together
  llm.py               Shared LLM client factory
  config.py            Env var loading (API key, model name)
  schemas.py            Shared graph state + request/response models
  agents/
    manager.py          Breaks the query into per-specialist sub-questions
    search_agent.py      General web/background context
    paper_agent.py       Academic literature angle
    benchmark_agent.py   Quantitative/benchmark comparison angle
    fact_checker.py      Cross-checks the three specialists' findings
    synthesis_agent.py   Writes the final Markdown report
requirements.txt
.env.example
PROJECT_CHARTER.md      Problem statement, use cases, design decisions
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in GEMINI_API_KEY
```

## Run

```bash
source venv/bin/activate
uvicorn app.main:app --reload
```

## API

**`GET /health`** — liveness check.

**`POST /research`**

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "How do transformer-based and diffusion-based approaches compare for image generation?"}'
```

Response:

```json
{
  "query": "...",
  "search_findings": "...",
  "paper_findings": "...",
  "benchmark_findings": "...",
  "fact_check_report": "...",
  "final_report": "..."
}
```

## Orchestration in detail

LangGraph runs the graph in "supersteps," determined entirely by the edges in `app/graph.py`:

1. `manager` runs first.
2. `manager` has three outgoing edges, so `search_agent`, `paper_agent`, and `benchmark_agent` all
   run next, in parallel.
3. `fact_checker` has three *incoming* edges (one from each specialist), so LangGraph
   automatically waits for all three to finish before running it — no manual "wait for all" code
   needed.
4. `synthesis_agent` runs last, using everything gathered so far.

All agents read and write one shared state object (`ResearchState` in `app/schemas.py`), which is
how later agents see earlier agents' output.

## Reliability: retries, rate limits, and fallback (no second model)

There is only one LLM provider (Gemini) — no second model to fall back to on failure. Instead,
`app/llm.py` and the agents implement graceful degradation:

- **Retries**: transient errors (timeouts, 429 rate limits, 5xx) are retried a few times with
  exponential backoff + jitter (`tenacity`). Permanent errors (bad API key, invalid request) are
  not retried — they fail fast instead of burning the retry budget.
- **Timeout**: every LLM call has a timeout so a hung request can't stall the whole graph.
- **Concurrency cap**: `search_agent`/`paper_agent`/`benchmark_agent` run in parallel, which
  means up to 3 simultaneous Gemini calls per request. A semaphore (`LLM_MAX_CONCURRENCY`) caps
  how many calls are in flight at once, so that fan-out doesn't itself trigger a rate limit.
- **Per-agent fallback when an LLM call still fails after retries** — no agent's failure takes
  down the whole request:
  - Search/Paper/Benchmark → that section becomes a "findings unavailable" placeholder; the
    other two specialists' results still flow through.
  - Manager → falls back to using the user's raw query for all three specialists instead of
    tailored sub-questions.
  - Fact Checker → notes that cross-checking was skipped; Synthesis still runs on the
    un-cross-checked findings.
  - Synthesis → falls back to a plain concatenation of whatever raw findings exist, instead of
    returning nothing.
- Every degradation is recorded in the response's `warnings` list, so the caller can see exactly
  what happened rather than it failing silently.

`POST /research` itself only returns a non-2xx response for something outside this path (e.g. a
bug, or a completely unset API key) — an LLM outage alone should still produce a 200 with
warnings, not a hard failure.

Tunable via env vars (see `.env.example`): `LLM_REQUEST_TIMEOUT_SECONDS`,
`LLM_MAX_RETRY_ATTEMPTS`, `LLM_MAX_CONCURRENCY`.

### Token usage

`max_output_tokens` is capped per call (2048) to bound cost per request, and each call's token
usage is logged (`response.usage_metadata`) for visibility. There's no hard per-request token
budget or caching yet — worth adding later if cost becomes a concern.

## Frontend

A chat-style UI lives in `frontend/` (plain HTML/CSS/JS, no build step) and is served by the
same FastAPI app at **http://localhost:8000/ui/** once `uvicorn` is running.

- Looks like a chat app: user messages as bubbles, assistant replies with an avatar + timestamp,
  a rounded pill input bar with a send button.
- While a request is in flight, a single typing-indicator row is shown — a blinking status line
  (in a distinct monospace font, like a live "agent working" status) that cycles through the
  pipeline stages, plus animated dots. It's purely cosmetic, since the backend returns the whole
  result in one shot rather than streaming per-agent progress yet.
- Once the response arrives, the typing indicator is removed and **only the final report** is
  shown, rendered from Markdown — no intermediate agent findings or warnings are surfaced in the
  UI (they're still in the API response if you need them for debugging).
- **Demo mode by default**: `frontend/app.js` has `CONFIG.USE_MOCK_DATA = true`, which loads
  `frontend/sample_response.json` (a saved real response) instead of calling `/research`. This
  lets you build/preview the UI without spending Gemini quota.
- **To go live**: set `CONFIG.USE_MOCK_DATA = false` in `frontend/app.js`. It will then call
  `POST /research` on `CONFIG.API_BASE_URL` (defaults to `http://localhost:8000`) with the
  real query.

## Conversation history & respecting length instructions

- **"Write 2 lines" being ignored**: the Synthesis Agent used to always force a fixed 5-section
  report template, even if you explicitly asked for something short. Its prompt
  (`app/agents/synthesis_agent.py`) now checks for explicit length/format instructions in the
  question first (e.g. "in 2 lines", "one sentence", "be brief") and follows those instead of the
  default template when present.
- **Multi-turn context**: `POST /research` now accepts an optional `history` field — a list of
  `{query, answer}` pairs from earlier turns in the conversation. The frontend
  (`frontend/app.js`) keeps this automatically: each turn's question/answer is appended to
  `conversationHistory` and sent with the next request; "New conversation" (the back arrow)
  clears it.
  - Only the **Manager** and **Synthesis** agents see the history (`app/history.py`,
    capped at the last 5 turns to bound token growth) — the Manager resolves references like
    "compare that with X" into self-contained sub-questions before handing them to
    Search/Paper/Benchmark, and Synthesis keeps the final answer consistent with prior turns.
    The specialist agents themselves stay stateless and don't see raw history, keeping their
    prompts small.

## Status

Backend, orchestration, and a demo-mode frontend are wired up and tested, now on Gemini with the
reliability layer described above.
