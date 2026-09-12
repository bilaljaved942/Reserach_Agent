from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.config import GEMINI_API_KEY
from app.graph import RESEARCH_GRAPH
from app.schemas import ResearchRequest, ResearchResponse

app = FastAPI(title="Research Agent API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(
    request: ResearchRequest,
    format: Literal["json", "markdown"] = Query(
        "json",
        description="'json' (default, for programmatic use) or 'markdown' to get back just "
        "the final report as readable plain text instead of an escaped JSON string.",
    ),
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    # Agents degrade gracefully on LLM failure rather than raising (see app/llm.py), so this
    # can still fail if something outside that path breaks (e.g. a bug, not an LLM outage).
    history = [turn.model_dump() for turn in request.history]
    try:
        result = await RESEARCH_GRAPH.ainvoke({"query": request.query, "history": history})
    except Exception as exc:  # pragma: no cover - unexpected/non-LLM failure
        raise HTTPException(status_code=502, detail=f"Research pipeline failed: {exc}") from exc

    if format == "markdown":
        report = result.get("final_report", "")
        warnings = result.get("warnings", [])
        if warnings:
            report = "> **Warnings:** " + "; ".join(warnings) + "\n\n" + report
        return PlainTextResponse(report, media_type="text/markdown")

    return ResearchResponse(
        query=request.query,
        search_findings=result.get("search_findings", ""),
        paper_findings=result.get("paper_findings", ""),
        benchmark_findings=result.get("benchmark_findings", ""),
        fact_check_report=result.get("fact_check_report", ""),
        final_report=result.get("final_report", ""),
        warnings=result.get("warnings", []),
    )


# Serves the simple chat-style UI in frontend/ at /ui (e.g. http://localhost:8000/ui/).
# Mounted last so it never shadows the /health and /research routes above.
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")
