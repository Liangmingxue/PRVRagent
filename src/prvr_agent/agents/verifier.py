from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Protocol

from prvr_agent.schemas import Candidate, EvidenceResult, QueryHypothesisGraph
from prvr_agent.video.sampler import DecordFrameSampler, TimeWindow


class EvidenceBackend(Protocol):
    def verify(self, *, query: str, graph: QueryHypothesisGraph, candidate: Candidate, video_path: str, window: TimeWindow, mode: str, num_frames: int) -> EvidenceResult: ...


class OpenAIFrameEvidenceBackend:
    """Raw-frame verifier with PRVR-specific structured output."""

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

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

    def verify(self, *, query: str, graph: QueryHypothesisGraph, candidate: Candidate, video_path: str, window: TimeWindow, mode: str, num_frames: int = 8) -> EvidenceResult:
        if mode not in {"support", "refute"}:
            raise ValueError("mode must be 'support' or 'refute'")
        sampler = DecordFrameSampler(video_path)
        timestamps, frames = sampler.sample(window, num_frames)
        schema = EvidenceResult.model_json_schema()
        graph_json = graph.model_dump_json(indent=2)
        task = (
            "Search for evidence that the positive hypothesis is fully satisfied."
            if mode == "support"
            else "Act as a falsifier: search for counterfactual or contradictory evidence showing this is a semantic near-miss."
        )
        content: list[dict] = [{"type": "text", "text": (
            f"PRVR query: {query}\nCandidate video: {candidate.video_id}\n"
            f"Inspection window: {window.start:.2f}s-{window.end:.2f}s\nMode: {mode}\n"
            f"Task: {task}\nHypothesis graph:\n{graph_json}\n\n"
            "Evaluate only observable evidence. Do not infer hidden intent or unseen causes. "
            "Return calibrated support/contradiction values and explicitly judge event completeness, "
            "entity consistency, and temporal consistency."
        )}]
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
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = EvidenceResult.model_validate_json(raw)
        if result.mode != mode:
            result = result.model_copy(update={"mode": mode})
        return result
