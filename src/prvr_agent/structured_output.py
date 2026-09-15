from __future__ import annotations

import json
from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_structured_json(text: str, response_model: Type[T]) -> T:
    """Parse a model response defensively.

    Local multimodal models sometimes wrap valid JSON in Markdown fences or add a
    short preamble even when JSON mode is requested. Accept those harmless forms,
    but still validate the final object strictly with Pydantic.
    """

    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
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
    response_model: Type[T],
    temperature: float,
    max_tokens: int,
    validation_retries: int,
    validator: Callable[[T], T] | None = None,
) -> T:
    """Request schema-valid JSON with bounded repair retries.

    `validator` can enforce request-specific invariants that are not expressible in
    the static Pydantic schema, such as preserving a particular CQHG's event ids.
    Its ValueError is fed back to the model and retried just like schema failures.
    """

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
            parsed = parse_structured_json(raw, response_model)
            return validator(parsed) if validator is not None else parsed
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
                            "The previous JSON was invalid for the required schema or request invariants. "
                            "Return a corrected JSON object only. Validation error:\n" + feedback
                        ),
                    },
                ]
            )

    raise RuntimeError(
        f"Model failed to produce valid {response_model.__name__} JSON after "
        f"{validation_retries + 1} attempt(s): {last_error}"
    ) from last_error
