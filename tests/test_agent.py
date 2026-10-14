import json
from types import SimpleNamespace

from greenwash_rag.agent import RagAgent


class FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            tool_call = SimpleNamespace(
                type="function_call",
                name="search_reports",
                arguments=json.dumps(
                    {"query": "emissions target", "company": None, "year": None, "top_k": 3}
                ),
                call_id="call-1",
            )
            return SimpleNamespace(output=[tool_call], output_text="")
        return SimpleNamespace(output=[], output_text="Grounded answer [a | Alpha | 2023 | p. 4].")


class FakeStore:
    def search(self, **kwargs):
        hit = SimpleNamespace(
            model_dump=lambda: {
                "chunk_id": "a",
                "text": "Alpha reports an emissions target.",
                "company": "Alpha",
                "year": 2023,
                "page": 4,
                "score": 0.9,
            }
        )
        return [hit]

    def catalog(self):
        return [{"company": "Alpha", "years": ["2023"]}]


def test_agent_executes_tool_and_returns_trace():
    client = SimpleNamespace(responses=FakeResponses())
    result = RagAgent(FakeStore(), client, model="test-model").ask("What is Alpha's target?")
    assert "[a | Alpha | 2023 | p. 4]" in result.answer
    assert result.trace[0].tool == "search_reports"
    assert result.trace[0].result_count == 1
