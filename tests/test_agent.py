import json
from types import SimpleNamespace

from greenwash_rag.agent import RagAgent
from greenwash_rag.agentic_evaluation import AgenticClassifier


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
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    result = RagAgent(FakeStore(), client, model="test-model").ask("What is Alpha's target?")
    assert "[a | Alpha | 2023 | p. 4]" in result.answer
    assert result.trace[0].tool == "search_reports"
    assert result.trace[0].result_count == 1


class FakeAgenticResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                type="function_call",
                name="search_reports",
                arguments=json.dumps(
                    {"query": "emissions target", "company": None, "year": None, "top_k": 3}
                ),
                call_id="agentic-call-1",
            )
            return SimpleNamespace(output=[tool_call], output_text="", usage=None)
        payload = {
            "label": "planning",
            "evidence_chunk_ids": ["a"],
            "rationale": "The statement is a future target.",
        }
        return SimpleNamespace(output=[], output_text=json.dumps(payload), usage=None)


def test_agentic_classifier_searches_and_returns_structured_label():
    responses = FakeAgenticResponses()
    client = SimpleNamespace(responses=responses)
    classifier = AgenticClassifier(FakeStore(), client, model="test-model", reasoning_effort="low")
    outcome = classifier.classify(
        {
            "text": "We aim to reduce emissions.",
            "aspect": "emissions",
            "category": "emissions control",
            "company": "Alpha",
            "year": 2023,
            "context_prev": "",
            "context_next": "",
        }
    )
    assert outcome["label"] == "planning"
    assert outcome["retrieved_chunk_ids"] == ["a"]
    assert outcome["trace"][0]["tool"] == "search_reports"
    assert responses.calls[0]["reasoning"] == {"effort": "low"}
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
