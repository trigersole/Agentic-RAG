from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Field(Path("artifacts/baseline/dev"), alias="GREENWASH_DATA_DIR")
    eval_path: Path = Field(
        Path("data/eval/gw1000.human_majority.23dc51f1d9de.jsonl"),
        alias="GREENWASH_EVAL_PATH",
    )
    chat_model: str = Field("gpt-5.6-luna", alias="OPENAI_CHAT_MODEL")
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = Field(
        "low", alias="OPENAI_REASONING_EFFORT"
    )
    embedding_model: str = Field("text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    top_k: int = Field(8, ge=1, le=20, alias="GREENWASH_TOP_K")
    max_agent_steps: int = Field(6, ge=1, le=12, alias="GREENWASH_MAX_AGENT_STEPS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
