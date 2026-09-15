# PRVR-Agent

Research scaffold for **Partially Relevant Video Retrieval (PRVR)** with two focused ideas:

1. **Counterfactual Query Hypothesis Graph (CQHG)** — semantic similarity is not enough; a retrieved video should satisfy the complete queried event rather than only a near-miss.
2. **Abductive Prospective Event World Modeling (APEI)** — PRVR should not only learn query–video compatibility; it should also reason about how the queried event could plausibly unfold inside a longer video world.

The current pipeline is:

```text
query
  -> Counterfactual Query Hypothesis Graph
  -> prospective event worlds: precondition -> anchored query event -> consequence
  -> DreamPRVR top-K candidates
  -> sparse global candidate observation
  -> defeasible world-belief revision
  -> CQHG + prospective reranking
```

The previous peak-seeded support/refute controller has been removed. PRVR-Agent no longer consumes DreamPRVR argmax locations, maps feature peaks to timestamps, expands local windows, or uses peak disagreement as an uncertainty signal.

## Innovation 1: Counterfactual Query Hypothesis Graph

CQHG decomposes a query into:

- atomic events;
- temporal relations;
- identity relations;
- a positive hypothesis describing what must be true;
- structured counterfactual near-misses such as partial events, temporal reversal, identity breaks, wrong objects, and wrong actions.

The graph answers:

> **What must be true for this query to be completely satisfied?**

This keeps the first contribution focused on the gap between semantic similarity and complete event satisfaction.

## Innovation 2: Abductive Prospective Event World Modeling

APEI starts from the CQHG and generates several plausible **event worlds**:

```text
soft preconditions
  -> fixed CQHG query anchor
  -> soft consequences
```

The query anchor is immutable: imagined context may be revised, but the query semantics cannot drift. Event worlds are alternatives rather than relevance requirements.

For each DreamPRVR candidate, the agent uniformly samples a small number of frames across the whole video and asks the local Qwen3-VL backend to estimate:

- CQHG query satisfaction;
- CQHG counterfactual risk;
- support/contradiction for each optional event world.

The world prior is then updated with candidate evidence:

```text
query-side prior world belief
  -> observe candidate video
  -> support / contradict worlds
  -> posterior world belief
  -> prospective reranking score
```

Missing an imagined precondition or consequence in sparse frames is treated as **neutral evidence**, not a contradiction. Because global observation is sparse, simply failing to see the queried local event is also treated as neutral/uncertain rather than proof that the event is absent. This is important because PRVR relevance only requires that the queried local event exists; optional context must never become a hard relevance condition.

## What is implemented

- `OpenAIHypothesisPlanner`: Qwen3-VL/vLLM CQHG generation.
- `OpenAIProspectiveWorldModeler`: CQHG-anchored multi-world imagination.
- `OpenAICoarseWorldObserver`: sparse global candidate observation with no peak seeding.
- `revise_world_beliefs`: posterior update over prospective worlds.
- `PRVRAgentReranker`: additive CQHG + prospective reranking on top of DreamPRVR base scores.
- `DreamPRVRAdapter`: preserves first-stage clip/frame retrieval scores but deliberately discards local argmax indices.
- bounded structured-JSON repair for local vLLM outputs.

The repository does **not** vendor DreamPRVR or other third-party video-agent repositories. See `THIRD_PARTY.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

For DreamPRVR + raw video + local Qwen3-VL:

```bash
pip install -e '.[all]'
```

## Local Qwen3-VL / vLLM backend

The defaults target:

```text
http://127.0.0.1:8000/v1
/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
```

Recommended environment variables:

```bash
export PRVR_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PRVR_LLM_API_KEY=EMPTY
export PRVR_LLM_MODEL=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
export PRVR_LLM_TIMEOUT=120
export PRVR_LLM_TEMPERATURE=0
export PRVR_LLM_MAX_TOKENS=2048
export PRVR_LLM_VALIDATION_RETRIES=2
export PRVR_LLM_HTTP_MAX_RETRIES=2
```

Check the server:

```bash
prvr-agent check-llm
```

Generate only the CQHG:

```bash
prvr-agent plan-llm "a man washes his hands and then opens the refrigerator"
```

Generate the CQHG and prospective event worlds:

```bash
prvr-agent imagine-llm \
  "a man puts a cake into the oven" \
  --worlds 3
```

For a dependency-free smoke test:

```bash
prvr-agent imagine "a man puts a cake into the oven" --worlds 3
```

## DreamPRVR integration

The adapter remains non-invasive:

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

`Candidate` now contains only video-level retrieval information:

```text
video_id
video_index
base_score
clip_score
frame_score
metadata
```

DreamPRVR's own local max pooling is still used to reproduce its first-stage video score, but PRVR-Agent no longer exposes or reasons from the argmax location.

## Current research status

This is an MVP implementation of the new two-contribution architecture. Before reporting benchmark results, the main remaining work is:

- benchmark-specific `video_id -> raw video path` resolution for TVR / ActivityNet Captions / Charades-STA;
- live local Qwen3-VL validation of CQHG and event-world quality;
- calibration of `num_worlds`, coarse-frame count, world-revision scales, and reranking weights on validation sets only;
- official R@1/5/10/100 and SumR evaluation wrappers;
- diagnostic evaluation for semantic near-misses and rare/unusual event contexts.
