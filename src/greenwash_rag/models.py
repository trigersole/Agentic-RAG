from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    company: str
    year: int | str | None = None
    page: int | str | None = None
    file: str | None = None
    statement_ids: list[Any] = Field(default_factory=list)
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)


class ToolStep(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result_count: int | None = None


class AskResponse(BaseModel):
    answer: str
    trace: list[ToolStep]
