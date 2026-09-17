# Third-party projects and design influences

This repository intentionally does **not** vendor source code from the projects below. It provides clean-room adapters and task-specific implementations around public interfaces and published ideas.

- DreamPRVR — CVPR 2026 PRVR retrieval backbone. This repository expects users to install or clone the upstream implementation separately when using `DreamPRVRAdapter`.
- VideoHV-Agent — inspiration for structured hypothesis formulation. CQHG and its PRVR-specific counterfactual schema are implemented independently here.
- I-JEPA / V-JEPA — conceptual motivation for predictive representation learning rather than purely retrospective discrimination; no code dependency.
- PlausiVL — related evidence that video-language models can reason about plausible action evolution and temporal counterfactuals; no code dependency.
- MANTA — related motivation for representing uncertain event futures with multiple alternatives instead of a single deterministic trajectory; no code dependency.
- Black Swan — related motivation for abductive and defeasible video reasoning under unexpected evidence; no code dependency.
- HyDE and multi-generation retrieval methods — important related work boundary. APEI differs by using temporally structured CQHG-anchored worlds whose beliefs are revised by candidate-video evidence rather than static pseudo-reference expansion.

The previous peak-seeded verification, multi-round window-expansion, and uncertainty-budget modules have been removed from the current design.

Before redistributing any upstream checkpoints, data, or source files, review the license and model/data terms in each upstream project.
