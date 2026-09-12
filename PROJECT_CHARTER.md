# Project Charter: Multi-Agent Research Assistant

## 1. Problem Statement

Answering a non-trivial research question (e.g. "What's the state of the art for X, and how do the leading approaches compare?") today requires a person to:

- Search the web for context, news, and general information
- Search academic sources (arXiv, Semantic Scholar, etc.) for papers and methods
- Look up benchmark results / leaderboards to compare approaches quantitatively
- Cross-check that claims from these different sources actually agree and are cited correctly
- Write up a coherent, structured summary pulling all of it together

Doing this manually is slow, inconsistent (depends on who's doing it and how thorough they are), and error-prone — it's easy to miss a source, misquote a paper, or present a stale benchmark number as current. There's no single tool that automates the full pipeline: gather → verify → synthesize.

## 2. Proposed Solution

Build a **multi-agent research assistant** that takes a user's research question and produces a fact-checked, cited report — automating the pipeline above via a team of cooperating agents instead of one agent trying to do everything.

### Agent roles (per the attached workflow)

| Agent | Responsibility |
|---|---|
| **Research Manager** | Receives the user's query, breaks it into sub-tasks, dispatches them to the right specialist agents, and collects their results. Owns overall orchestration. |
| **Search Agent** | Runs general web search for background, context, news, and non-academic sources. |
| **Paper Agent** | Searches academic sources (e.g. arXiv, Semantic Scholar) for relevant papers, methods, and findings. |
| **Benchmark Agent** | Looks up quantitative benchmark/leaderboard data to support comparisons between approaches. |
| **Fact Checker** | Cross-validates claims and citations gathered by the three specialist agents against each other and their sources; flags contradictions or unsupported claims. |
| **Synthesis Agent** | Compiles the verified findings into a single structured, cited final report. |

### Flow

```
User → Research Manager → [Search Agent, Paper Agent, Benchmark Agent] (parallel)
                        → Fact Checker
                        → Synthesis Agent
                        → Final Report
```

## 3. Example Use Cases

- **Literature review**: "Summarize recent approaches to X and how they compare" → papers + benchmarks + synthesis.
- **Competitive/technical analysis**: "How do models A, B, C compare on Y benchmark, and what do people say about tradeoffs?" → benchmark data + web sentiment + fact-checked comparison.
- **Due-diligence style research**: quick, structured report on a technology/topic combining web + academic evidence for internal use (e.g. a design doc appendix).
- **Fact-checked Q&A**: user asks a question that needs current, sourced information rather than the model's own (possibly stale) knowledge.

## 4. Technical Direction

- **Language**: Python, end-to-end — backend, agent definitions, and orchestration.
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`, no custom orchestration logic. The graph mirrors the diagram directly: `manager -> {search_agent, paper_agent, benchmark_agent} -> fact_checker -> synthesis_agent`.
- **LLM provider**: Gemini (via `langchain-google-genai`), single provider — no second model as a fallback. Reliability instead comes from retries with backoff, timeouts, a concurrency cap, and per-agent graceful degradation (see `README.md` → "Reliability" for the full design).
- **Backend**: FastAPI service exposing `POST /research` (accepts a query, runs the graph, returns the final report + each agent's intermediate findings + any degradation `warnings`) and `GET /health`.
- **Data sources (v1)**: no external APIs. Search/Paper/Benchmark agents answer from the LLM's own knowledge, each prompted to flag uncertainty and staleness rather than assert unverified facts as current. This keeps v1 simple and gets the orchestration + endpoints working end-to-end; real tools (web search API, arXiv/Semantic Scholar, a benchmark data source) can be added to any agent later as a drop-in change, without altering the graph shape.
  - **Trade-off to keep in mind**: without real sources, the Fact Checker can only catch contradictions *between* the three agents' LLM-generated answers — it cannot verify claims against ground truth. Treat all v1 output as a draft that needs human/independent verification, not a sourced report.

## 5. Out of Scope (for now)

- Multi-turn conversational refinement of the report (v1 is single-shot: query in, report out).
- A frontend UI (v1 can be CLI or a simple API — UI is a later phase if needed).
- Support for non-text sources (images, video, datasets beyond benchmark scores).

## 6. Success Criteria

- Given a research question, the system produces a structured report with inline citations back to real sources.
- The Fact Checker demonstrably catches at least some contradictions/unsupported claims in testing (not a no-op pass-through).
- The pipeline runs end-to-end as Python code with clear boundaries between agents (each agent's logic is independently testable).

## 7. Next Steps

1. ~~Decide on the orchestration approach.~~ Done — LangGraph.
2. ~~Scaffold the Python backend and get endpoints running.~~ Done — see `README.md`.
3. ~~Decide on LLM provider and reliability strategy.~~ Done — Gemini only, no fallback model; retries/timeout/concurrency-cap/graceful-degradation instead (see `README.md`).
4. Add a real GEMINI_API_KEY and run a live end-to-end request to sanity-check report quality.
5. When ready to move beyond v1: pick concrete data sources/APIs for Search, Paper, and Benchmark agents, and add them as tools to the respective agent.
6. Frontend — to be scoped later per the user's direction.
