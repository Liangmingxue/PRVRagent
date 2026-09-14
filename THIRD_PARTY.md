# Third-party projects and design influences

This repository intentionally does **not** vendor source code from the projects below. It provides clean-room adapters and task-specific reimplementations around their public interfaces and published ideas.

- DreamPRVR — CVPR 2026 PRVR baseline/retrieval backbone. This repository expects users to install or clone the upstream implementation separately when using `DreamPRVRAdapter`.
- VideoHV-Agent — hypothesis/verification design inspiration. The PRVR graph, counterfactual hypotheses, evidence schema, and reranking logic here are original task-specific code.
- VideoSeek — coarse-to-fine raw-video observation design inspiration. The implementation here uses its own sampler/backend abstraction rather than copied tools.
- A4VL — cross-review and disagreement control inspiration; not a runtime dependency.
- REVISE / SparseVideoUnderstanding — multi-round observation/budget-control inspiration; not vendored.
- VideoSearch-R1 — related agentic retrieval baseline and engineering reference; not a runtime dependency.

Before redistributing any upstream checkpoints, data, or source files, review the license and model/data terms in each upstream project.
