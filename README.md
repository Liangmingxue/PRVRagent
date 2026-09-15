# PRVR-Agent

Research scaffold for **Partially Relevant Video Retrieval (PRVR)** built around two ideas only:

1. **Counterfactual Query Hypothesis Graph (CQHG)**: represent what must be true for the query to be fully satisfied, together with structured semantic near-misses.
2. **Abductive Prospective Event World Modeling (APEI)**: before fine-grained alignment, imagine several plausible event worlds in which the query could occur, then revise their probabilities using coarse evidence from each candidate video.

The previous peak-seeded / local-spurious-response contribution has been removed from the agent design. Counterfactual near-miss reasoning is retained because it is part of CQHG, not a separate third contribution.

## Current pipeline

```text
query
  -> Counterfactual Query Hypothesis Graph
  -> K CQHG-anchored possible event worlds
  -> DreamPRVR top-K candidates
  -> one coarse whole-video observation per candidate
       -> hard CQHG event / temporal / identity / counterfactual evidence
       -> soft support / contradiction for each imagined world
  -> event-world posterior belief revision
  -> CQHG + prospective evidence-aware reranking
```

The key distinction is:

```text
CQHG: What must be true?
APEI: If it is true, how could the event world unfold?
```

## Innovation 1: Counterfactual Query Hypothesis Graph

`OpenAIHypothesisPlanner` decomposes the query into:

- atomic events;
- temporal constraints;
- identity constraints;
- a positive hypothesis;
- counterfactual near-misses such as partial event, temporal reversal, identity break, wrong object, and wrong action.

CQHG is the hard semantic anchor. For each candidate, the visual observer reports `query_support`, `query_contradiction`, actually observed atomic-event ids, verified temporal/identity relation ids, and supported counterfactual ids. A multi-event query therefore cannot receive full CQHG credit from event presence alone when its required temporal or identity relation is unverified.

Prospective imagination is not allowed to modify or replace CQHG atomic events **or** its temporal/identity relations.

## Innovation 2: Abductive Prospective Event World Modeling

`OpenAIEventWorldPlanner` takes the CQHG and generates multiple structured worlds:

```text
preconditions -> immutable CQHG query event -> consequences
```

Each world contains a prior probability, every CQHG atomic-event id exactly once, and every CQHG temporal/identity relation id exactly once as hard anchors. The implementation rejects generated worlds that drop, add, duplicate, rename, or reverse these CQHG anchors.

`OpenAIWorldEvidenceBackend` then samples coarse frames across the **whole candidate video** and estimates, for each imagined world:

- visual support;
- visual contradiction;
- uncertainty.

The imagined preconditions and consequences are soft context. Their absence is not treated as contradiction; only visible conflicting evidence should suppress a world. Uncertainty reduces the influence of an observation toward zero rather than acting as negative evidence. Positive soft-world evidence is also gated by explicit CQHG contradiction so an imagined context cannot rescue a candidate that visibly violates the hard query semantics.

Posterior beliefs follow the evidence-weighted form:

```text
log posterior(H_k)
  ∝ log prior(H_k)
    + beta * support(H_k, V)
    - gamma * contradiction(H_k, V)
    - uncertainty penalty
```

The final score fuses three terms: the original DreamPRVR score, hard CQHG satisfaction, and posterior-weighted prospective world evidence.

## What was removed

The following former components are no longer part of the intended method:

- clip/frame peak indices as agent inputs;
- feature-index -> timestamp peak mapping;
- peak-seeded local windows;
- iterative window expansion around local peaks;
- retrieval/verification joint uncertainty budget;
- the old standalone peak-based support/refute reranker.

DreamPRVR still uses its own internal max-similarity mechanism to produce its normal retrieval scores, but PRVR-Agent no longer exports or reasons from those argmax locations.

## Main modules

- `src/prvr_agent/agents/hypothesis_planner.py`: CQHG generation.
- `src/prvr_agent/agents/world_model.py`: abductive prospective event-world generation.
- `src/prvr_agent/agents/world_observer.py`: one coarse candidate observation for CQHG evidence and event-world evidence.
- `src/prvr_agent/prospective.py`: CQHG scoring, prior normalization, posterior belief revision, and score fusion.
- `src/prvr_agent/pipeline.py`: end-to-end CQHG + APEI reranking.
- `src/prvr_agent/retriever/dreamprvr_adapter.py`: non-invasive DreamPRVR top-K adapter without peak export.

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

The defaults target the local OpenAI-compatible vLLM endpoint already used by this project:

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

Generate a CQHG:

```bash
prvr-agent plan-llm "a man washes his hands and then opens the refrigerator"
```

Generate prospective event worlds from that query:

```bash
prvr-agent imagine-llm "a man washes his hands and then opens the refrigerator" --num-worlds 3
```

Rule-based smoke-test versions are also available:

```bash
prvr-agent plan "a man washes his hands and then opens the refrigerator"
prvr-agent imagine "a man washes his hands and then opens the refrigerator" --num-worlds 3
```

## DreamPRVR integration

The adapter reproduces the upstream clip/frame max-similarity scores used to rank candidates, but it no longer returns argmax locations:

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

These candidates can then be passed to `PRVRAgentReranker`, which evaluates hard CQHG satisfaction and candidate-specific prospective world beliefs in one coarse visual pass.

## Research status

This branch is still an MVP research scaffold. Before benchmark reporting, the main remaining work is:

- validate Qwen3-VL CQHG and world generation quality on real TVR / ActivityNet Captions / Charades-STA queries;
- validate whether uniform coarse-frame sampling is sufficient for long untrimmed videos without reintroducing peak-seeded reasoning;
- add benchmark-specific video-id -> path resolution;
- calibrate `base_weight`, `graph_weight`, `world_weight`, support/contradiction scales on validation data only;
- add official PRVR R@K / SumR evaluation and ablations against single-caption/query-expansion baselines;
- log CQHG evidence, generated worlds, and posterior changes for qualitative analysis.
