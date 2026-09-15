from prvr_agent.schemas import ProspectiveWorldSet
from prvr_agent.structured_output import parse_structured_json


def test_parser_accepts_fenced_json():
    raw = '''```json
{"query":"q","worlds":[{"id":"W1","query_anchor":"a","prior":1.0}]}
```'''
    parsed = parse_structured_json(raw, ProspectiveWorldSet)
    assert parsed.worlds[0].id == "W1"
