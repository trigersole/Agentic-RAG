from dotenv import load_dotenv
from openai import OpenAI

from .agent import RagAgent
from .config import Settings, get_settings
from .store import VectorStore


def build_store(settings: Settings | None = None, client: OpenAI | None = None) -> VectorStore:
    load_dotenv()
    settings = settings or get_settings()
    client = client or OpenAI()

    def embed_query(text: str) -> list[float]:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    return VectorStore(settings.data_dir, embed_query)


def build_agent(settings: Settings | None = None) -> RagAgent:
    load_dotenv()
    settings = settings or get_settings()
    client = OpenAI()
    store = build_store(settings, client)
    return RagAgent(
        store,
        client,
        model=settings.chat_model,
        reasoning_effort=settings.reasoning_effort,
        default_top_k=settings.top_k,
        max_steps=settings.max_agent_steps,
    )
