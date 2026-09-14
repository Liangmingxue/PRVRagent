# PRVR-Agent

Research scaffold for **Partially Relevant Video Retrieval (PRVR)** that treats a strong local retrieval activation as an **evidence proposal to verify**, not as the final relevance decision.

The intended pipeline is:

```text
query
  -> counterfactual hypothesis graph
  -> DreamPRVR top-K + clip/frame peak locations
  -> support/refute raw-frame verification around each peak
  -> uncertainty-gated expansion / early stop
  -> evidence-aware reranking
```

## Research claims encoded by the scaffold

1. **Counterfactual Query Hypothesis Graph**: decompose a query into atomic events, temporal/identity constraints, a positive hypothesis, and semantic near-miss counterfactuals.
2. **Peak-Seeded Support-Refute Verification**: reuse PRVR activation peaks as search seeds, then separately seek supporting and falsifying evidence in raw video.
3. **Retrieval-Reasoning Joint Uncertainty Budget**: allocate more visual inspection only when retrieval margin, clip/frame peak disagreement, or verifier disagreement indicate uncertainty.

## What is implemented

- `DreamPRVRAdapter`: consumes a loaded upstream DreamPRVR model and cached context tensors, preserves clip/frame peak indices, and returns top-K candidates.
- `OpenAIHypothesisPlanner`: structured query-to-event-graph planner. A rule-based planner is included only for smoke tests.
- `OpenAIFrameEvidenceBackend`: raw-frame support/refute verifier with calibrated structured output.
- `PRVRAgentReranker`: peak-seeded iterative verification and score fusion.
- Unit tests for graph validation, contradiction penalties, and controller early stopping.

The repository does **not** copy upstream DreamPRVR/VideoHV/VideoSeek/A4VL/REVISE/VideoSearch-R1 sources. See `THIRD_PARTY.md`.

## Installation

Minimal development install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

For DreamPRVR integration and raw-video verification:

```bash
pip install -e '.[all]'
```

You still need the upstream DreamPRVR repository, its features/checkpoints, and the benchmark data according to its own instructions.

## Smoke test

```bash
prvr-agent plan "a man washes his hands and then opens the refrigerator"
```

## DreamPRVR integration

The adapter is non-invasive: it does not modify upstream files. After the upstream validation code has produced cached context tensors, create:

```python
from prvr_agent.retriever import DreamPRVRAdapter

adapter = DreamPRVRAdapter(
    model=dreamprvr_model,
    context_info=context_info,
    video_ids=context_info["video_metas"],
    clip_scale_weight=cfg["clip_scale_w"],
    frame_scale_weight=cfg["frame_scale_w"],
)

batch = adapter.retrieve(query_feat, query_mask, top_k=20)
candidates = batch.candidates[0]
```

Each candidate contains both video-level scores and the argmax locations that produced the strongest clip/frame responses. These locations seed verification rather than being treated as proof of relevance.

## Important dataset-specific setting

`clip_peak_index` and `frame_peak_index` are feature indices, not universally seconds. Configure `PeakMappingConfig` with the actual temporal stride used by your extracted features before running raw-video verification.

## Status

This is an audited **MVP research scaffold**, not yet a reproduction package. Before reporting benchmark numbers, the next required steps are:

- add benchmark-specific video-id -> path resolution and exact feature-index -> timestamp mappings;
- validate DreamPRVR feature tensor shapes for TVR / ActivityNet Captions / Charades-STA;
- choose a fixed VLM backend and prompts for fair reproducibility;
- calibrate fusion weights on validation data only;
- log per-candidate evidence traces, cost, and failure modes;
- add official PRVR recall/SumR evaluation wrappers.
