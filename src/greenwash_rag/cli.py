import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from openai import OpenAI

from .agentic_evaluation import AgenticClassifier, run_agentic_evaluation
from .bootstrap import build_agent, build_store
from .config import get_settings
from .evaluation import load_items, run_evaluation
from .store import CorpusError, VectorStore

app = typer.Typer(help="Agentic RAG over the Greenwash sustainability-report corpus.")


def _load_environment() -> None:
    load_dotenv()
    get_settings.cache_clear()


def _corpus_error(exc: CorpusError) -> None:
    typer.secho(f"Corpus error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2) from exc


@app.command()
def doctor() -> None:
    """Check local runtime, API-key, and corpus readiness without API calls."""
    _load_environment()
    settings = get_settings()
    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "chat_model": settings.chat_model,
        "reasoning_effort": settings.reasoning_effort,
        "retrieval_mode": settings.retrieval_mode,
        "retrieval_candidates": settings.retrieval_candidates,
        "rerank_depth": settings.rerank_depth,
        "reranker_model": settings.reranker_model,
        "data_dir": str(settings.data_dir.resolve()),
        "required_files": {},
    }
    required_files = checks["required_files"]
    assert isinstance(required_files, dict)
    for filename in VectorStore.REQUIRED_FILES:
        path = settings.data_dir / filename
        required_files[filename] = {
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }

    ready = checks["api_key_configured"] and all(
        item["present"] for item in required_files.values()
    )
    checks["ready"] = ready
    typer.echo(json.dumps(checks, indent=2))
    if not ready:
        raise typer.Exit(code=2)


@app.command()
def validate() -> None:
    """Validate corpus files without making an API request."""
    _load_environment()
    settings = get_settings()
    try:
        store = VectorStore(
            settings.data_dir,
            lambda _: [],
            retrieval_mode=settings.retrieval_mode,
            candidate_k=settings.retrieval_candidates,
            rerank_depth=settings.rerank_depth,
            reranker_model=settings.reranker_model,
        )
    except CorpusError as exc:
        _corpus_error(exc)
    typer.echo(json.dumps(store.stats(), indent=2, default=str))


@app.command()
def ask(question: str) -> None:
    """Ask a grounded question using the bounded tool-calling agent."""
    _load_environment()
    try:
        result = build_agent().ask(question)
    except CorpusError as exc:
        _corpus_error(exc)
    typer.echo(result.answer)
    typer.echo("\nAgent trace:")
    typer.echo(json.dumps([step.model_dump() for step in result.trace], indent=2))


@app.command()
def search(
    query: str,
    company: str | None = None,
    year: int | None = None,
    top_k: int = 8,
) -> None:
    """Inspect raw retrieval results without running the answering agent."""
    _load_environment()
    settings = get_settings()
    client = OpenAI()
    try:
        store = build_store(settings, client)
    except CorpusError as exc:
        _corpus_error(exc)
    hits = store.search(query, company=company, year=year, top_k=top_k)
    # Escaping non-ASCII keeps output safe in the legacy CP-1252 PowerShell console as well as UTF-8.
    typer.echo(json.dumps([hit.model_dump() for hit in hits], indent=2, ensure_ascii=True))


@app.command()
def smoke() -> None:
    """Run one real evaluation item through embedding, retrieval, and classification."""
    _load_environment()
    settings = get_settings()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        typer.secho("OPENAI_API_KEY is not configured in .env.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    items = load_items(settings.eval_path, "dev", limit=1)
    client = OpenAI()
    store = build_store(settings, client)
    metrics, results = run_evaluation(
        store=store,
        client=client,
        model=settings.chat_model,
        reasoning_effort=settings.reasoning_effort,
        embedding_model=settings.embedding_model,
        items=items,
        top_k=settings.top_k,
    )
    typer.echo(json.dumps({"metrics": metrics, "result": results[0]}, indent=2))


@app.command("evaluate")
def evaluate_command(
    mode: str = typer.Option("fixed", help="fixed or agentic"),
    split: str = typer.Option("dev", help="dev, test, or all"),
    limit: int | None = typer.Option(None, min=1, help="Maximum items; omit for full split."),
    top_k: int = typer.Option(8, min=1, max=20),
    repeats: int = typer.Option(1, min=1, max=10, help="Agentic runs per item."),
    output: str = typer.Option("runs/eval_results.jsonl"),
) -> None:
    """Measure fixed or agentic retrieval and four-label classification accuracy."""
    _load_environment()
    if split not in {"dev", "test", "all"}:
        raise typer.BadParameter("split must be dev, test, or all")
    if mode not in {"fixed", "agentic"}:
        raise typer.BadParameter("mode must be fixed or agentic")
    if mode == "fixed" and repeats != 1:
        raise typer.BadParameter("repeats applies only to agentic mode")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        typer.secho("OPENAI_API_KEY is not configured in .env.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    settings = get_settings()
    items = load_items(settings.eval_path, split, limit)
    client = OpenAI()
    store = build_store(settings, client)
    if mode == "agentic":
        agent = AgenticClassifier(
            store,
            client,
            model=settings.chat_model,
            reasoning_effort=settings.reasoning_effort,
            default_top_k=top_k,
            max_steps=settings.max_agent_steps,
        )
        metrics, _ = run_agentic_evaluation(
            agent=agent,
            store=store,
            items=items,
            top_k=top_k,
            repeats=repeats,
            output_path=Path(output),
            progress=lambda completed, total: (
                typer.echo(f"Agentic evaluation: {completed}/{total} runs complete")
                if completed == 1 or completed == total or completed % 10 == 0
                else None
            ),
        )
    else:
        metrics, _ = run_evaluation(
            store=store,
            client=client,
            model=settings.chat_model,
            reasoning_effort=settings.reasoning_effort,
            embedding_model=settings.embedding_model,
            items=items,
            top_k=top_k,
            output_path=Path(output),
        )
    typer.echo(json.dumps(metrics, indent=2))
    typer.echo(f"Per-item results: {Path(output).resolve()}")


if __name__ == "__main__":
    app()
