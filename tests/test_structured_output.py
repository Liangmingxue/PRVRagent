from types import SimpleNamespace

from prvr_agent.schemas import AtomicEvent
from prvr_agent.structured_output import parse_structured_json, request_structured_json


def test_parser_accepts_fenced_json():
    result = parse_structured_json(
        '```json\n{"id":"E1","subject":"person","action":"opens"}\n```',
        AtomicEvent,
    )
    assert result.id == "E1"


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            content = '{"id":"E1","subject":"person"}'
        else:
            content = '{"id":"E1","subject":"person","action":"opens"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_request_retries_schema_invalid_json():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = request_structured_json(
        client=client,
        model="local",
        messages=[{"role": "user", "content": "test"}],
        response_model=AtomicEvent,
        temperature=0.0,
        max_tokens=256,
        validation_retries=1,
    )
    assert result.action == "opens"
    assert completions.calls == 2
