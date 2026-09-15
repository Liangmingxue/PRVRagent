from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Protocol

from prvr_agent.llm_config import LLMConfig, create_openai_compatible_client
from prvr_agent.schemas import Candidate, EventWorldSet, QueryHypothesisGraph, WorldEvidenceBundle
from prvr_agent.video.sampler import DecordFrameSampler, TimeWindow


class WorldEvidenceBackend(Protocol):
    def assess(
        self,
        *,
        query: str,
        graph: QueryHypothesisGraph,
        worlds: EventWorldSet,
        candidate: Candidate,
        video_path: str,
        num_frames: int,
    ) -> WorldEvidenceBundle: ...


class OpenAIWorldEvidenceBackend:
    """Observe a candidate coarsely and revise CQHG-conditioned event-world beliefs.

    Unlike the removed peak-seeded verifier, this backend samples the candidate
    globally. It asks whether each soft imagined context is supported or visibly
    contradicted; missing context alone is not treated as a contradiction.
    """

    def __init__(
        self,
        client=None,
        model: str | None = None,
        *,
        config: LLMConfig | None = None,
        max_image_side: int = 768,
        jpeg_quality: int = 80,
    ) -> None:
        cfg = config or LLMConfig.from_env()
        self.client = client or create_openai_compatible_client(cfg)
        self.model = model or cfg.model
        self.temperature = cfg.temperature
        self.max_image_side = int(max_image_side)
        self.jpeg_quality = int(jpeg_quality)
        self._samplers: dict[str, DecordFrameSampler] = {}

    @classmethod
    def from_env(cls) -> "OpenAIWorldEvidenceBackend":
        return cls(config=LLMConfig.from_env())

    def _frame_to_data_url(self, frame) -> str:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'video' dependencies for image encoding") from exc
        image = Image.fromarray(frame).convert("RGB")
        if max(image.size) > self.max_image_side:
            image.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=self.jpeg_quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def assess(
        self,
        *,
        query: str,
        graph: QueryHypothesisGraph,
        worlds: EventWorldSet,
        candidate: Candidate,
        video_path: str,
        num_frames: int = 8,
    ) -> WorldEvidenceBundle:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        sampler = self._samplers.get(video_path)
        if sampler is None:
            sampler = DecordFrameSampler(video_path)
            self._samplers[video_path] = sampler
        timestamps, frames = sampler.sample(TimeWindow(0.0, sampler.duration), num_frames)

        schema = WorldEvidenceBundle.model_json_schema()
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"PRVR query: {query}\nCandidate video: {candidate.video_id}\n\n"
                    f"CQHG (hard query semantics):\n{graph.model_dump_json(indent=2)}\n\n"
                    f"Prospective event worlds (soft context):\n{worlds.model_dump_json(indent=2)}\n\n"
                    "Assess each imagined world using only visible evidence in the sampled frames. "
                    "The CQHG query event is immutable and the preconditions/consequences are only plausible context. "
                    "Absence of a predicted precondition or consequence is NOT contradiction. Raise contradiction only "
                    "when visible evidence conflicts with that imagined trajectory. Return one evidence item for every "
                    "world id and do not invent new world ids."
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
                {
                    "role": "system",
                    "content": "You are a conservative prospective event-world evidence assessor for video retrieval.",
                },
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = WorldEvidenceBundle.model_validate_json(raw)
        expected = {world.id for world in worlds.worlds}
        observed = {item.world_id for item in result.evidence}
        if observed != expected:
            raise ValueError(
                f"world observer must return evidence for every imagined world; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        if result.candidate_video_id != candidate.video_id:
            result = result.model_copy(update={"candidate_video_id": candidate.video_id})
        return result
