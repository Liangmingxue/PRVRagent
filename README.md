# PRVR-Agent

Research scaffold for **Partially Relevant Video Retrieval (PRVR)** built around two ideas only:

1. **Counterfactual Query Hypothesis Graph (CQHG)**: represent what must be true for the query to be fully satisfied, together with structured semantic near-misses.
2. **Abductive Prospective Event World Modeling (APEI)**: before fine-grained alignment, imagine several plausible event worlds in which the query could occur, then revise their probabilities using temporally coherent candidate-video evidence.

The previous peak-seeded / local-spurious-response contribution has been removed from the agent design. Counterfactual near-miss reasoning is retained because it is part of CQHG, not a separate third contribution.

## Current pipeline

```text
query
  -> Counterfactual Query Hypothesis Graph
  -> K CQHG-anchored possible event worlds
  -> DreamPRVR top-K candidates
  -> hybrid query-agnostic sidekick scan
       -> dense low-resolution pixel-change trace
       -> cached DreamPRVR semantic features
       -> local semantic novelty + global kernel temporal segmentation
       -> robust visual/semantic fusion
  -> event-aware variable-length temporal segments
  -> coarse CQHG + event-world assessment per segment
  -> bounded relevance+coverage dense re-observation
  -> choose one short contiguous candidate span
  -> isolated single-span temporal confirmation with explicit event timestamps
  -> revise event-world beliefs from confirmed evidence
  -> CQHG + prospective evidence-aware reranking
```

The key distinction is:

```text
CQHG: What must be true?
APEI: If it is true, how could the event world unfold?
```

## Innovation 1: Counterfactual Query Hypothesis Graph

`OpenAIHypothesisPlanner` decomposes the query into atomic events, temporal constraints, identity constraints, a positive hypothesis, and structured counterfactual near-misses such as partial event, temporal reversal, identity break, wrong object, and wrong action.

CQHG is the hard semantic anchor. The visual observer reports `query_support`, `query_contradiction`, observed atomic-event ids, verified temporal/identity relation ids, supported counterfactual ids, and timestamp ranges for visible atomic events. A multi-event query therefore cannot receive full CQHG credit from event presence alone when its required temporal or identity relation is unverified.

Hard CQHG evidence is never formed by unioning distant observations. In particular, `E1` near 10 s and `E2` near 90 s cannot be mechanically composed into a complete `E1 -> E2` match. A temporal relation is accepted only when both endpoint events are verified inside the same visible confirmation span and their explicit timestamp ranges satisfy the relation.

Prospective imagination is not allowed to modify or replace CQHG atomic events or temporal/identity relations.

## Innovation 2: Abductive Prospective Event World Modeling

`OpenAIEventWorldPlanner` takes the CQHG and generates multiple structured worlds:

```text
preconditions -> immutable CQHG query event -> consequences
```

Each world contains a prior probability, every CQHG atomic-event id exactly once, and every CQHG temporal/identity relation id exactly once as hard anchors. Generated worlds that drop, add, duplicate, rename, or reverse these anchors are rejected.

### Event-aware Observe -> Refine -> Confirm -> Revise

The long-video observer no longer relies on one whole-video uniform sample or fixed DreamPRVR peak locations. It uses four APEI-internal observation stages.

**1. Hybrid query-agnostic sidekick scan.** The raw-video sidekick densely samples inexpensive low-resolution luminance changes. In parallel, `DreamPRVRAdapter` reuses the already-computed chronological `video_feat` representations and `video_mask`. It derives both a local semantic novelty curve and a global kernel-temporal segmentation over the ordered semantic features. The two visual/semantic signals are robustly normalized per video before event-boundary proposal. This lets low-motion semantic transitions survive even when raw pixel change is weak. If the upstream mask/semantic trace is unavailable, the observer safely falls back to the visual sidekick.

The semantic sidekick is deliberately **not** a query-frame similarity score: it never sees the retrieval query, never exports a DreamPRVR argmax, and cannot directly contribute hard CQHG relevance. Its only role is to improve temporal structure discovery and observation allocation.

The global semantic segmenter is implemented independently in `video/kernel_temporal.py` from the KTS mathematical objective rather than by vendoring third-party KTS code. For an L2-normalized feature sequence with linear kernel `K`, the interval cost is the within-segment kernel scatter

```text
J(i,j) = sum_t K_tt - (1 / length) * sum_{s,t} K_st,
```

and dynamic programming finds the minimum-cost contiguous segmentation. The number of change points is selected with the KTS-style complexity penalty exposed as a validation-time hyperparameter.

DreamPRVR frame positions are treated as **uniform temporal bins**, not endpoint samples. If KTS selects change point `c` in a sequence of `N` semantic bins, its approximate physical boundary is therefore

```text
t_boundary = (c / N) * video_duration
```

rather than `c / (N - 1) * video_duration`.

**2. Adaptive event segmentation.** `build_event_segments()` combines global semantic structure with local visual/semantic novelty into variable-length event proposals while enforcing minimum and maximum segment durations. Quiet regions are force-split instead of silently becoming extremely long segments. The split logic preserves nearby meaningful global boundaries rather than creating a forced split that immediately suppresses them. The default maximum duration is 20 s and the default minimum duration is 4 s.

**3. Coverage-safe dense re-observation.** Every event segment first receives a low-cost VLM pass. A small bounded number of segments are then re-observed with more frames according to a priority combining hybrid sidekick salience, partial CQHG coverage, unresolved relations, Qwen uncertainty, support/contradiction conflict, and prospective-world hints. Temporal diversity prevents the budget from collapsing onto one neighborhood.

The default refinement threshold is now `0.0`: all coarse event segments remain eligible for the bounded refinement budget (`max_refinement_chunks=3` by default). This is intentional. A hard pre-filter can terminate refinement when the sparse coarse pass is confidently wrong; keeping all coarse segments eligible lets relevance priority and temporal coverage decide how to spend the fixed refinement budget. A positive threshold remains available as an ablation/configuration option.

**4. Isolated contiguous-span confirmation.** Adjacent partial event segments may jointly propose one short contiguous span, but their evidence is not directly unioned into the final CQHG score. The proposed span is sent to a clean Qwen request that contains only that span. Final hard event/relation evidence must be re-observed from scratch in this isolated request. This prevents cross-attention between distant segments in a batched coarse request from becoming final evidence, while still allowing a true event that straddles one adaptive boundary to be confirmed.

The confirmation request asks for `event_time_ranges`. Temporal relations are checked in code against these timestamps before they are accepted.

Posterior beliefs follow the evidence-weighted form:

```text
log posterior(H_k)
  ∝ log prior(H_k)
    + beta * support(H_k, V)
    - gamma * contradiction(H_k, V)
    - uncertainty penalty
```

The final score fuses three terms: the original DreamPRVR score, hard CQHG satisfaction, and posterior-weighted prospective world evidence.

## Why the observation policy is structured this way

The implementation follows published long-video findings rather than treating sparse Qwen uncertainty as a sufficient selector:

- **DeCafNet (CVPR 2025)** motivates a cheap dense sidekick followed by expensive expert processing only where needed.
- **Kernel Temporal Segmentation as an Adaptive Tokenizer (ICCVW 2023)** motivates task-agnostic feature-space temporal segmentation into variable-length semantically consistent units instead of fixed uniform chunks. PRVR-Agent now includes an independent KTS-style global dynamic-programming segmenter plus a local semantic novelty signal.
- **KTV (AAAI 2026)** supports question-agnostic feature-based preselection to avoid coupling the prefilter to query similarity.
- **AKS (CVPR 2025)** motivates balancing informative evidence with temporal coverage under a fixed observation budget.
- **FOCUS (ICLR 2026)** motivates coarse exploration across the full temporal search space before fine exploitation; in particular, it highlights that hard pre-filtering can discard the most informative moments.
- **From Frames to Clips (2025)** motivates keeping short temporally coherent spans instead of reasoning only from isolated keyframes.

These references motivate the observation policy only. CQHG and prospective event-world belief revision remain the actual PRVR contributions.

## What was removed

The following former components are no longer part of the intended method:

- clip/frame peak indices as agent inputs;
- feature-index -> timestamp peak mapping;
- peak-seeded local windows;
- iterative window expansion around local peaks;
- retrieval/verification joint uncertainty budget;
- the old standalone peak-based support/refute reranker.

DreamPRVR still uses its own internal max-similarity mechanism to produce normal retrieval scores, but PRVR-Agent no longer exports or reasons from those argmax locations. The sidekick and adaptive segmentation are full-video APEI observation mechanisms, not retrieval-peak proposals.

## Main modules

- `src/prvr_agent/agents/hypothesis_planner.py`: CQHG generation.
- `src/prvr_agent/agents/world_model.py`: abductive prospective event-world generation.
- `src/prvr_agent/agents/world_observer.py`: hybrid sidekick fusion, coarse event assessment, coverage-aware refinement, isolated confirmation, and coherence-preserving aggregation.
- `src/prvr_agent/video/kernel_temporal.py`: independent global kernel-temporal dynamic programming and semantic-boundary mapping.
- `src/prvr_agent/video/event_segments.py`: visual/semantic sidekick fusion and adaptive event-segment construction.
- `src/prvr_agent/video/sampler.py`: raw-video frame sampling and dense low-resolution visual-change scan.
- `src/prvr_agent/prospective.py`: CQHG scoring, prior normalization, posterior belief revision, and score fusion.
- `src/prvr_agent/pipeline.py`: end-to-end CQHG + APEI reranking.
- `src/prvr_agent/retriever/dreamprvr_adapter.py`: DreamPRVR top-K adapter plus query-agnostic local/global semantic sidekick structure; no peak export.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

For DreamPRVR, raw-video observation, and the local multimodal model:

```bash
pip install -e '.[all]'
```

The package accepts Python 3.9+ and PyTorch 2.0+ so it can be installed alongside the public DreamPRVR environments; the Qwen3-VL/vLLM server can remain in its own separate environment and is accessed over the OpenAI-compatible endpoint.

## Local Qwen3-VL / vLLM

```bash
export PRVR_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PRVR_LLM_API_KEY=EMPTY
export PRVR_LLM_MODEL=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
export PRVR_LLM_TIMEOUT=120
export PRVR_LLM_TEMPERATURE=0
export PRVR_LLM_MAX_TOKENS=4096
export PRVR_LLM_VALIDATION_RETRIES=2
export PRVR_LLM_HTTP_MAX_RETRIES=2
```

Check the server:

```bash
prvr-agent check-llm
```

Generate a CQHG:

```bash
prvr-agent plan-llm "a man washes his hands and then opens the refrigerator"
```

Generate prospective event worlds:

```bash
prvr-agent imagine-llm "a man washes his hands and then opens the refrigerator" --num-worlds 3
```

Rule-based smoke-test versions are also available:

```bash
prvr-agent plan "a man washes his hands and then opens the refrigerator"
prvr-agent imagine "a man washes his hands and then opens the refrigerator" --num-worlds 3
```

## DreamPRVR integration

The adapter reproduces upstream clip/frame max-similarity scores used to rank candidates but no longer returns argmax locations. When `context_info["video_mask"]` is present, it additionally attaches query-agnostic local semantic novelty and global kernel-temporal boundaries to each returned candidate so APEI can reuse the cached video representation without another heavy encoder pass.

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

Default APEI observation uses a 2 FPS low-resolution raw-video sidekick capped at 512 frames, equal initial weights for visual/semantic sidekick fusion, adaptive event segments up to 20 s long, 4 VLM frames per coarse segment, three 12-frame dense re-observations selected from all coarse segments using relevance+coverage, and a 16-frame isolated confirmation over at most two adjacent segments / 40 s. These are research defaults and must be calibrated on validation data; the 0.5/0.5 fusion weights and KTS penalty are not claimed as literature-derived optimal constants.

## Research status

This branch is still an MVP research scaffold. Before benchmark reporting, the main remaining work is:

- validate Qwen3-VL CQHG, event timestamp, isolated confirmation, and world-generation quality on real TVR / ActivityNet Captions / Charades-STA queries;
- calibrate visual-vs-semantic sidekick fusion, KTS penalty / maximum change points, scan rate, event boundary quantile, segment duration, coarse/refinement/confirmation frame budgets, and compute/recall trade-offs on validation data;
- ablate `visual-only`, `local-semantic`, `KTS-global`, `hybrid`, fixed-chunk, and whole-video uniform observation policies;
- ablate refinement threshold `0.0` against positive-threshold pre-filtering to measure the accuracy/compute effect of coverage-safe refinement;
- add benchmark-specific video-id -> path resolution;
- calibrate `base_weight`, `graph_weight`, `world_weight`, support/contradiction scales on validation data only;
- add official PRVR R@K / SumR evaluation and ablations against single-caption/query-expansion baselines;
- log segment boundaries, visual/semantic/fused sidekick salience, refinement choices, confirmation span, CQHG event timestamps, generated worlds, and posterior changes for qualitative analysis.
