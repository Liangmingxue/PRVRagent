from types import SimpleNamespace

import pytest

from prvr_agent.llm_config import LLMConfig, resolve_served_model
from prvr_agent.schemas import AtomicEvent, QueryHypothesisGraph
from prvr_agent.structured_output import parse_structured_json


def test_base_url_normalization():
    cfg = LLMConfig(base_url="http://127.0.0.1:8000")
    assert cfg.normalized_base_url() == "http://127.0.0.1:8000/v1"
    cfg = LLMConfig(base_url="http://127.0.0.1:8000/v1/")
    assert cfg.normalized_base_url() == "http://127.0.0.1:8000/v1"


def test_resolve_served_model_accepts_exact_id():
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="local-qwen3-vl")])))
    assert resolve_served_model(client, "local-qwen3-vl") == "local-qwen3-vl"


def test_resolve_served_model_rejects_mismatch():
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="served-name")])))
    with pytest.raises(RuntimeError, match="served-name"):
        resolve_served_model(client, "filesystem-path")


def test_structured_parser_accepts_json_fence():
    text = '''```json
{"query":"x","atomic_events":[{"id":"E1","subject":"p","action":"walk","object":null,"attributes":[]}],"temporal_constraints":[],"identity_constraints":[],"positive_hypothesis":"x","counterfactuals":[]}
```'''
    graph = parse_structured_json(text, QueryHypothesisGraph)
    assert graph.atomic_events == [AtomicEvent(id="E1", subject="p", action="walk")]
