from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_structured_json(text: str, response_model: type[T]) -> T:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            raise
        payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    return response_model.model_validate(payload)


def request_structured_json(
    *,
    client,
    model: str,
    messages: list[dict],
    response_model: type[T],
    temperature: float,
    max_tokens: int,
    validation_retries: int,
) -> T:
    working_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(validation_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=working_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            return parse_structured_json(raw, response_model)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            if attempt >= validation_retries:
                break
            feedback = str(exc)
            if len(feedback) > 1200:
                feedback = feedback[:1200] + "..."
            working_messages.extend(
                [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON did not satisfy the required schema. "
                            "Return a corrected JSON object only. Validation error:\n" + feedback
                        ),
                    },
                ]
            )
    raise RuntimeError(
        f"Model failed to produce valid {response_model.__name__} JSON after "
        f"{validation_retries + 1} attempt(s): {last_error}"
    ) from last_error
