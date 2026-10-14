from functools import lru_cache

from fastapi import FastAPI, HTTPException

from .agent import RagAgent
from .bootstrap import build_agent
from .config import get_settings
from .models import AskRequest, AskResponse
from .store import VectorStore

app = FastAPI(title="Greenwash Agentic RAG", version="0.1.0")


@lru_cache
def get_agent() -> RagAgent:
    return build_agent()


@app.get("/health")
def health() -> dict[str, object]:
    try:
        # Health checks validate local data without requiring an API key.
        stats = VectorStore(get_settings().data_dir, lambda _: []).stats()
        return {"status": "ok", "corpus": stats}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return get_agent().ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
