from types import SimpleNamespace

import pytest

from prvr_agent.llm_config import LLMConfig, resolve_served_model


def test_base_url_normalization():
    cfg = LLMConfig(base_url="http://127.0.0.1:8000")
    assert cfg.normalized_base_url() == "http://127.0.0.1:8000/v1"

    cfg = LLMConfig(base_url="http://127.0.0.1:8000/v1/")
    assert cfg.normalized_base_url() == "http://127.0.0.1:8000/v1"


def test_config_rejects_non_finite_or_invalid_values():
    with pytest.raises(ValueError):
        LLMConfig(timeout=-1).validate()
    with pytest.raises(ValueError):
        LLMConfig(temperature=float("nan")).validate()
    with pytest.raises(ValueError):
        LLMConfig(max_tokens=0).validate()


def test_resolve_served_model_accepts_exact_id():
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="local-qwen3-vl")])
        )
    )
    assert resolve_served_model(client, "local-qwen3-vl") == "local-qwen3-vl"


def test_resolve_served_model_rejects_mismatch():
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="served-name")])
        )
    )
    try:
        resolve_served_model(client, "filesystem-path")
    except RuntimeError as exc:
        assert "served-name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected model-id mismatch to raise")
