# PRVR-Agent

Research scaffold for **Partially Relevant Video Retrieval (PRVR)** that treats a strong local retrieval activation as an **evidence proposal to verify**, not as the final relevance decision.

The intended pipeline is:

```text
query
  -> counterfactual hypothesis graph
  -> DreamPRVR top-K + valid clip/frame peak locations
  -> support/refute raw-frame verification around the strongest peak(s)
  -> uncertainty-gated expansion / early stop
  -> evidence-aware reranking
```

## Research claims encoded by the scaffold

1. **Counterfactual Query Hypothesis Graph**: decompose a query into atomic events, temporal/identity constraints, a positive hypothesis, and semantic near-miss counterfactuals.
2. **Peak-Seeded Support-Refute Verification**: reuse PRVR activation peaks as search seeds, then separately seek supporting and falsifying evidence in raw video.
3. **Retrieval-Reasoning Joint Uncertainty Budget**: allocate more visual inspection only when retrieval margin, clip/frame peak disagreement, or verifier disagreement indicate uncertainty.

## What is implemented

- `DreamPRVRAdapter`: consumes a loaded upstream DreamPRVR model and cached context tensors, preserves the upstream retrieval score, masks padded frame locations when selecting evidence peaks, and returns top-K candidates.
- `OpenAIHypothesisPlanner`: structured query-to-event-graph planner with bounded JSON validation/repair retries. A rule-based planner is included for smoke tests and ablations.
- `OpenAIFrameEvidenceBackend`: raw-frame support/refute verifier with bounded video-reader caching, image resizing, structured-output validation, and evidence-id sanitization.
- `PRVRAgentReranker`: peak-seeded iterative verification, coherent multi-round evidence aggregation, uncertainty-gated early stopping, and score fusion.
- `LLMConfig`: OpenAI-compatible local backend configuration, defaulting to the project's Qwen3-VL vLLM server.
- CI and unit tests for graph validation, padded-peak handling, relative time mapping, contradiction semantics, coherent evidence aggregation, score-scale consistency, controller stopping, and vLLM model-id resolution.

The repository does **not** copy upstream DreamPRVR/VideoHV/VideoSeek/A4VL/REVISE/VideoSearch-R1 sources. See `THIRD_PARTY.md`.

## Installation

The package supports Python 3.9+ so that it can coexist with the public DreamPRVR TVR/ActivityNet environment. Minimal development install:

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

The repository defaults to an OpenAI-compatible local vLLM service rather than the public OpenAI API. For a same-machine deployment, bind the service explicitly to loopback:

```bash
ENV_DIR=/home/omnisky/miniconda3/envs/chart-vllm

env -u PYTHONHOME -u PYTHONPATH \
  LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_VISIBLE_DEVICES=1 \
  "$ENV_DIR/bin/vllm" serve \
  /home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9
```

PRVR-Agent uses ordinary OpenAI-compatible chat-completions and does not currently require tool calling, so `--tool-call-parser hermes` is optional for this repository.

Configure the client explicitly on Linux:

```bash
export PRVR_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PRVR_LLM_API_KEY=EMPTY
export PRVR_LLM_MODEL=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
export PRVR_LLM_TIMEOUT=120
export PRVR_LLM_TEMPERATURE=0
export PRVR_LLM_MAX_TOKENS=2048
export PRVR_LLM_VALIDATION_RETRIES=2
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

For a server reachable by other machines, do not rely on a vLLM API key alone; place the service behind an appropriate firewall/reverse proxy and expose only the routes you need. See `docs/VLLM.md`.

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

Each candidate contains the upstream DreamPRVR video-level scores, valid evidence peak indices, and metadata describing the resampled clip/frame location counts.

## Peak index -> raw-video time mapping

DreamPRVR does **not** use one universal seconds-per-index stride in its public preprocessing. Its clip branch averages every video into `map_size` bins, while the frame branch uniformly resamples the original feature sequence up to `max_ctx_l`. Therefore PRVR-Agent defaults to **relative bin-center mapping** using each raw video's duration:

```text
time ~= (peak_index + 0.5) / num_valid_locations * video_duration
```

For candidates produced by `DreamPRVRAdapter`, this is automatic when using the raw-frame verifier:

```python
from prvr_agent.pipeline import PipelineConfig, PeakMappingConfig

cfg = PipelineConfig(
    peak_mapping=PeakMappingConfig(mode="relative")
)
```

Use `mode="fixed_stride"` only when your own feature extractor truly has a known fixed temporal stride:

```python
cfg = PipelineConfig(
    peak_mapping=PeakMappingConfig(
        mode="fixed_stride",
        clip_seconds_per_index=CLIP_STRIDE_SECONDS,
        frame_seconds_per_index=FRAME_STRIDE_SECONDS,
    )
)
```

## Status

This remains an audited **MVP research scaffold**, not yet a benchmark reproduction package. Before reporting paper numbers, the remaining required work is:

- add benchmark-specific video-id -> raw-video path resolution for TVR / ActivityNet Captions / Charades-STA;
- run live end-to-end Qwen3-VL verification on the target Linux machine and freeze one reproducible vLLM configuration;
- calibrate fusion weights on validation data only;
- add persistent per-candidate evidence/cost traces and a MetaVerifier if that remains part of the final method;
- add official PRVR R@1/R@5/R@10/R@100/SumR evaluation wrappers and full benchmark runners.
