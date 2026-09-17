# DreamPRVR + CQHG/APEI benchmark integration

This note describes the benchmark path implemented on `apei-world-modeling-20260915`.

## 1. Temporal visual source

APEI accepts either:

- a compressed video file through `DecordFrameSampler`; or
- a directory containing chronologically ordered extracted frames through `FrameDirectorySampler`.

The observer still calls the historical `DecordFrameSampler(path)` entry point. For a directory, that entry point dispatches to `FrameDirectorySampler` only when the extraction FPS is explicitly configured:

```bash
export PRVR_FRAME_DIRECTORY_FPS=3
```

Do **not** set this globally unless the frame release is actually 3 fps. The value affects event timestamps and therefore CQHG temporal-relation verification.

The public TVQA preprocessing code uses `interval2frame(..., fps=3)` as its default temporal conversion, which is why 3 fps is appropriate for the corresponding TVQA-style extracted-frame release. PRVR-Agent intentionally does not hard-code this assumption for arbitrary datasets.

Check raw-video coverage:

```bash
prvr-agent index-videos /path/to/raw/video/root \
  --expected-video-ids /path/to/video_ids.json
```

Check extracted-frame coverage:

```bash
prvr-agent index-frames /path/to/frame/root \
  --expected-video-ids /path/to/video_ids.json
```

A TVQA-style layout such as

```text
frame_root/
  friends_frames/
    friends_s01e01_seg01_clip_00/
      000001.jpg
      000002.jpg
      ...
  house_frames/
    house_s01e01_seg01_clip_01/
      ...
```

is indexed by the innermost video-directory name.

## 2. Use DreamPRVR's validation scores as the benchmark authority

The integration bridge is:

```python
from prvr_agent.integration import rerank_dreamprvr_query_loader
```

It intentionally uses the public DreamPRVR model method
`get_pred_from_raw_query(...)` to construct the full collection score matrix.
This matches the upstream validation path rather than assuming our adapter is the benchmark authority.

`DreamPRVRAdapter` is then used to construct Top-K `Candidate` objects and attach the query-agnostic semantic sidekick trace. Before Qwen/APEI computation starts, the bridge verifies that the adapter's Top-K set exactly matches DreamPRVR's own Top-K for that query. A mismatch raises an error instead of silently producing an invalid experiment.

The public DreamPRVR `TxtDataSet4PRVR` stores the original query strings in
`query_eval_loader.dataset.captions`, so the bridge uses the original query text directly. It does not try to decode or reconstruct text from the precomputed RoBERTa features.

## 3. One-command official DreamPRVR benchmark

The runnable entrypoint now performs the complete public DreamPRVR path:

```text
official checkpoint config/state_dict
  -> get_datasets(cfg)
  -> get_models(cfg) + strict checkpoint load
  -> get_validations(cfg).compute_context_info(model, test_context_dataloader)
  -> test_query_eval_loader
  -> authoritative DreamPRVR full scores
  -> CQHG/APEI Top-K reranking
  -> full matrices + R@K/Rsum manifest
```

Use the official checkout and feature layout described by DreamPRVR. `--data-root`
is the directory that contains `activitynet/`, `charades/`, or `tvr/`, not the
collection directory itself. The checkpoint loader uses PyTorch's restricted
`weights_only=True` mode and requires the official checkpoint keys `config` and
`state_dict`; it never silently performs an unrestricted pickle load.

Before the first run, configure the already-running local Qwen3-VL vLLM service:

```bash
export PRVR_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PRVR_LLM_API_KEY=EMPTY
export PRVR_LLM_MODEL=Qwen3-VL-8B-Instruct-FP8

prvr-agent check-llm
```

Use the exact model id returned by `/v1/models`. If vLLM is already occupying
physical GPU 1, run DreamPRVR on a different visible GPU to avoid model-memory
contention, for example `CUDA_VISIBLE_DEVICES=0` with `--device cuda:0`.

### TVR extracted frames

The corresponding TVQA/TVR frame release uses 3 FPS, but the rate remains an
explicit command argument because other extracted-frame releases may differ:

```bash
CUDA_VISIBLE_DEVICES=0 prvr-agent benchmark-dreamprvr \
  --dreamprvr-root /path/to/CVPR26-DreamPRVR \
  --checkpoint /path/to/tvr/best.ckpt \
  --data-root /path/to/DreamPRVR \
  --dataset tvr \
  --visual-root /path/to/tvqa_frames \
  --frame-directory-fps 3 \
  --top-k 10 \
  --max-queries 10 \
  --device cuda:0 \
  --output-dir runs/tvr-smoke-k10
```

### ActivityNet Captions raw videos

ActivityNet paths are indexed recursively by filename stem. Thus a benchmark id
such as `v_demo` must resolve to exactly one file such as `v_demo.mp4`:

```bash
CUDA_VISIBLE_DEVICES=0 prvr-agent benchmark-dreamprvr \
  --dreamprvr-root /path/to/CVPR26-DreamPRVR \
  --checkpoint /path/to/activitynet/best.ckpt \
  --data-root /path/to/DreamPRVR \
  --dataset activitynet \
  --visual-root /path/to/activitynet/raw_videos \
  --top-k 10 \
  --max-queries 10 \
  --device cuda:0 \
  --output-dir runs/activitynet-smoke-k10
```

### Charades-STA raw videos

Charades paths are also indexed recursively by exact filename stem:

```bash
CUDA_VISIBLE_DEVICES=0 prvr-agent benchmark-dreamprvr \
  --dreamprvr-root /path/to/CVPR26-DreamPRVR \
  --checkpoint /path/to/charades/best.ckpt \
  --data-root /path/to/DreamPRVR \
  --dataset charades \
  --visual-root /path/to/Charades_v1_480 \
  --top-k 10 \
  --max-queries 10 \
  --device cuda:0 \
  --output-dir runs/charades-smoke-k10
```

The command rejects duplicate filename stems, missing benchmark videos,
checkpoint/dataset mismatches, and TVR runs without an explicit FPS before any
expensive Qwen request is sent.

Each output directory contains:

```text
benchmark_result.json
dreamprvr_base_scores.npy
dreamprvr_cqhg_apei_scores.npy
query_ids.json
video_ids.json
```

The manifest records the paths/configuration, DreamPRVR fusion weights, baseline
and reranked metrics, metric deltas, query/video counts, and matrix filenames.

After the 10-query smoke test succeeds, remove `--max-queries` for a full test
split and repeat with `--top-k 10`, `20`, and `50`. Use a distinct output
directory for every dataset/K pair so score matrices are never overwritten.

## 4. Programmatic integration with public DreamPRVR objects

After loading the public DreamPRVR model/checkpoint and dataloaders in its normal environment:

```python
from prvr_agent.agents.hypothesis_planner import OpenAIHypothesisPlanner
from prvr_agent.agents.world_model import OpenAIEventWorldPlanner
from prvr_agent.agents.world_observer import OpenAIWorldEvidenceBackend
from prvr_agent.evaluation import IndexedVideoPathResolver, IndexedFrameDirectoryResolver
from prvr_agent.integration import rerank_dreamprvr_query_loader
from prvr_agent.pipeline import PRVRAgentReranker, PipelineConfig

# DreamPRVR public validation object can build the same encoded context it uses
# for its own benchmark evaluation.
context_info = val_criterion.compute_context_info(model, test_context_dataloader)

# ActivityNet / Charades example:
resolver = IndexedVideoPathResolver.from_root("/path/to/raw/videos")

# TVR-style extracted-frame example instead:
# export PRVR_FRAME_DIRECTORY_FPS=3
# resolver = IndexedFrameDirectoryResolver.from_root("/path/to/tvqa_frames")

agent = PRVRAgentReranker(
    planner=OpenAIHypothesisPlanner.from_env(),
    world_planner=OpenAIEventWorldPlanner.from_env(),
    world_evidence_backend=OpenAIWorldEvidenceBackend.from_env(),
    video_path_resolver=resolver,
    cfg=PipelineConfig(),
)

result = rerank_dreamprvr_query_loader(
    model=model,
    query_loader=test_query_eval_loader,
    context_info=context_info,
    reranker=agent,
    clip_scale_weight=cfg["clip_scale_w"],
    frame_scale_weight=cfg["frame_scale_w"],
    top_k=20,
)

print("DreamPRVR:", result.base_metrics.as_dict())
print("DreamPRVR + CQHG/APEI:", result.reranked_metrics.as_dict())
print("Delta:", result.metric_delta())
```

For a cheap local Qwen smoke test before a complete benchmark, use:

```python
result = rerank_dreamprvr_query_loader(
    ...,
    top_k=10,
    max_queries=10,
)
```

## 5. Why rank-slot reranking is used

APEI produces a transformed score for only the expensive Top-K shortlist. It is not valid to overwrite those K entries with APEI scores and then compare them numerically with untouched DreamPRVR scores outside the shortlist.

Instead, the integration keeps the original DreamPRVR Top-K score values as rank slots:

1. DreamPRVR selects the Top-K membership and supplies K original score values.
2. CQHG/APEI decides only the ordering of those K candidates.
3. The sorted original Top-K DreamPRVR values are reassigned according to the APEI ordering.
4. Every score outside Top-K is unchanged.

This makes the experiment a strict Top-K reranking experiment.

## 6. Required shortlist ablation

Because the current architecture is

```text
DreamPRVR -> Top-K -> CQHG/APEI -> rerank
```

APEI cannot recover a relevant video that DreamPRVR excludes from the shortlist. Report shortlist sensitivity, for example:

```text
K = 10, 20, 50
```

and add `K=100` when compute permits. Report both retrieval metrics and Qwen/runtime cost.
