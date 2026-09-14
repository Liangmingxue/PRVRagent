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
- `LLMConfig`: OpenAI-compatible local backend configuration, defaulting to the project's Qwen3-VL vLLM server.
- Unit tests for graph validation, contradiction penalties, peak preservation, pipeline reranking, controller early stopping, and vLLM model-id resolution.

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

## Local Qwen3-VL / vLLM backend

The repository now defaults to an OpenAI-compatible local vLLM service rather than the public OpenAI API. The deployment currently assumed by the defaults is:

```bash
ENV_DIR=/home/omnisky/miniconda3/envs/chart-vllm

env -u PYTHONHOME -u PYTHONPATH \
  LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_VISIBLE_DEVICES=1 \
  "$ENV_DIR/bin/vllm" serve \
  /home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8 \
  --port 8000 \
  --trust-remote-code \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9
```

PRVR-Agent uses normal OpenAI-compatible chat-completions and does not currently require tool calling, so `--tool-call-parser hermes` is optional for this repository.

Configure the client explicitly on Linux:

```bash
export PRVR_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PRVR_LLM_API_KEY=EMPTY
export PRVR_LLM_MODEL=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
export PRVR_LLM_TIMEOUT=120
export PRVR_LLM_TEMPERATURE=0.1
```

Check the endpoint and exact served model id before an experiment:

```bash
prvr-agent check-llm
```

If `/v1/models` reports a different id than the filesystem path, either set `PRVR_LLM_MODEL` to the returned id or launch vLLM with an explicit `--served-model-name` and use that same name in PRVR-Agent.

Generate a hypothesis graph with the local model:

```bash
prvr-agent plan-llm "a man washes his hands and then opens the refrigerator"
```

For more details, see `docs/VLLM.md`.

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

`clip_peak_index` and `frame_peak_index` are feature indices, not universally seconds. `PeakMappingConfig` has no silent default: configure both temporal strides from the exact feature-extraction pipeline before constructing `PRVRAgentReranker`. For example:

```python
from prvr_agent.pipeline import PeakMappingConfig, PipelineConfig

cfg = PipelineConfig(
    peak_mapping=PeakMappingConfig(
        clip_seconds_per_index=CLIP_STRIDE_SECONDS,
        frame_seconds_per_index=FRAME_STRIDE_SECONDS,
    )
)
```

## Status

This is an audited **MVP research scaffold**, not yet a reproduction package. Before reporting benchmark numbers, the next required steps are:

- add benchmark-specific video-id -> path resolution and exact feature-index -> timestamp mappings;
- validate DreamPRVR feature tensor shapes for TVR / ActivityNet Captions / Charades-STA;
- fix one reproducible Qwen3-VL/vLLM configuration for all reported experiments;
- calibrate fusion weights on validation data only;
- log per-candidate evidence traces, cost, and failure modes;
- add official PRVR recall/SumR evaluation wrappers.
