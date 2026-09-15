from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an OpenAI-compatible chat-completions backend."""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8"
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 2048
    validation_retries: int = 2
    http_max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        cfg = cls(
            base_url=os.getenv("PRVR_LLM_BASE_URL", cls.base_url),
            api_key=os.getenv("PRVR_LLM_API_KEY", cls.api_key),
            model=os.getenv("PRVR_LLM_MODEL", cls.model),
            timeout=float(os.getenv("PRVR_LLM_TIMEOUT", str(cls.timeout))),
            temperature=float(os.getenv("PRVR_LLM_TEMPERATURE", str(cls.temperature))),
            max_tokens=int(os.getenv("PRVR_LLM_MAX_TOKENS", str(cls.max_tokens))),
            validation_retries=int(os.getenv("PRVR_LLM_VALIDATION_RETRIES", str(cls.validation_retries))),
            http_max_retries=int(os.getenv("PRVR_LLM_HTTP_MAX_RETRIES", str(cls.http_max_retries))),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("PRVR_LLM_BASE_URL must not be empty")
        if not self.model.strip():
            raise ValueError("PRVR_LLM_MODEL must not be empty")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("PRVR_LLM_TIMEOUT must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("PRVR_LLM_TEMPERATURE must be finite and non-negative")
        if self.max_tokens <= 0:
            raise ValueError("PRVR_LLM_MAX_TOKENS must be positive")
        if self.validation_retries < 0 or self.http_max_retries < 0:
            raise ValueError("LLM retry counts must be non-negative")

    def normalized_base_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"


def create_openai_compatible_client(config: LLMConfig | None = None):
    """Create an OpenAI SDK client pointed at the configured local server."""

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'llm' dependencies: pip install -e '.[llm]'") from exc

    cfg = config or LLMConfig.from_env()
    cfg.validate()
    return OpenAI(
        base_url=cfg.normalized_base_url(),
        api_key=cfg.api_key,
        timeout=cfg.timeout,
        max_retries=cfg.http_max_retries,
    )


def resolve_served_model(client, requested_model: str | None = None) -> str:
    """Resolve the model id exposed by an OpenAI-compatible `/v1/models` endpoint."""

    response = client.models.list()
    ids = [item.id for item in response.data]
    if not ids:
        raise RuntimeError("The LLM server returned no models from /v1/models")
    if requested_model is None:
        return ids[0]
    if requested_model not in ids:
        raise RuntimeError(
            f"Configured model {requested_model!r} is not served. Available model ids: {ids}. "
            "Set PRVR_LLM_MODEL to one of these ids or start vLLM with --served-model-name."
        )
    return requested_model
