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

## 3. Minimal integration with the public DreamPRVR validation objects

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

## 4. Why rank-slot reranking is used

APEI produces a transformed score for only the expensive Top-K shortlist. It is not valid to overwrite those K entries with APEI scores and then compare them numerically with untouched DreamPRVR scores outside the shortlist.

Instead, the integration keeps the original DreamPRVR Top-K score values as rank slots:

1. DreamPRVR selects the Top-K membership and supplies K original score values.
2. CQHG/APEI decides only the ordering of those K candidates.
3. The sorted original Top-K DreamPRVR values are reassigned according to the APEI ordering.
4. Every score outside Top-K is unchanged.

This makes the experiment a strict Top-K reranking experiment.

## 5. Required shortlist ablation

Because the current architecture is

```text
DreamPRVR -> Top-K -> CQHG/APEI -> rerank
```

APEI cannot recover a relevant video that DreamPRVR excludes from the shortlist. Report shortlist sensitivity, for example:

```text
K = 10, 20, 50
```

and add `K=100` when compute permits. Report both retrieval metrics and Qwen/runtime cost.
