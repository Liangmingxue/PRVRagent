from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Protocol

from prvr_agent.llm_config import LLMConfig, create_openai_compatible_client
from prvr_agent.schemas import Candidate, EvidenceResult, QueryHypothesisGraph
from prvr_agent.video.sampler import DecordFrameSampler, TimeWindow


class EvidenceBackend(Protocol):
    def verify(
        self,
        *,
        query: str,
        graph: QueryHypothesisGraph,
        candidate: Candidate,
        video_path: str,
        window: TimeWindow,
        mode: str,
        num_frames: int,
    ) -> EvidenceResult: ...


class OpenAIFrameEvidenceBackend:
    """Raw-frame verifier backed by an OpenAI-compatible multimodal server.

    The default configuration targets the local Qwen3-VL vLLM endpoint defined
    by :class:`prvr_agent.llm_config.LLMConfig`.
    """

    def __init__(self, client=None, model: str | None = None, *, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig.from_env()
        self.client = client or create_openai_compatible_client(cfg)
        self.model = model or cfg.model
        self.temperature = cfg.temperature
        self._samplers: dict[str, DecordFrameSampler] = {}

    @classmethod
    def from_env(cls) -> "OpenAIFrameEvidenceBackend":
        return cls(config=LLMConfig.from_env())

    @staticmethod
    def _frame_to_data_url(frame) -> str:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'video' dependencies for image encoding") from exc
        image = Image.fromarray(frame)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def verify(
        self,
        *,
        query: str,
        graph: QueryHypothesisGraph,
        candidate: Candidate,
        video_path: str,
        window: TimeWindow,
        mode: str,
        num_frames: int = 8,
    ) -> EvidenceResult:
        if mode not in {"support", "refute"}:
            raise ValueError("mode must be 'support' or 'refute'")
        sampler = self._samplers.get(video_path)
        if sampler is None:
            sampler = DecordFrameSampler(video_path)
            self._samplers[video_path] = sampler
        timestamps, frames = sampler.sample(window, num_frames)
        schema = EvidenceResult.model_json_schema()
        graph_json = graph.model_dump_json(indent=2)
        task = (
            "Search for evidence that the positive hypothesis is fully satisfied. "
            "support measures evidence for the positive hypothesis; contradiction measures evidence against it."
            if mode == "support"
            else "Act as a falsifier: search for counterfactual or contradictory evidence showing this is a semantic near-miss. "
            "support measures evidence FOR a counterfactual (against the query); contradiction measures evidence "
            "AGAINST that counterfactual (not against the query). Absence of an event in these sampled frames alone "
            "does not prove that the event is absent from the video."
        )
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"PRVR query: {query}\nCandidate video: {candidate.video_id}\n"
                    f"Sampled frame range: {min(timestamps):.2f}s-{max(timestamps):.2f}s\nMode: {mode}\n"
                    f"Task: {task}\nHypothesis graph:\n{graph_json}\n\n"
                    "Evaluate only observable evidence. Do not infer hidden intent or unseen causes. "
                    "Return calibrated support/contradiction values and explicitly judge event completeness, "
                    "entity consistency, and temporal consistency. Use only event ids from the supplied graph. "
                    "Any reported evidence times must fall within the sampled frame range."
                ),
            }
        ]
        for ts, frame in zip(timestamps, frames):
            content.append({"type": "text", "text": f"Timestamp: {ts:.2f}s"})
            content.append({"type": "image_url", "image_url": {"url": self._frame_to_data_url(frame)}})
        content.append({"type": "text", "text": "Return only JSON matching this schema:\n" + json.dumps(schema)})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a conservative visual evidence verifier for video retrieval."},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = EvidenceResult.model_validate_json(raw)
        if result.mode != mode:
            raise ValueError(f"Verifier returned mode {result.mode!r} for requested mode {mode!r}")
        event_ids = {event.id for event in graph.atomic_events}
        if not set(result.verified_event_ids).issubset(event_ids):
            raise ValueError("Verifier returned event ids that are not in the hypothesis graph")
        return result
