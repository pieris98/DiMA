# Missing Elements Audit: DiMA paper vs repository

## Scope and approach

I read the full `DiMA.md` paper draft in this repository and then scanned all files in the codebase (Python, configs, scripts, docs, and data artifacts) to map what is implemented vs. what is only described in the paper.

This report focuses on **missing or incomplete code/data artifacts** explicitly mentioned in the paper: datasets, models/methods, metrics, and experiments.

---

## Executive summary

The repository contains a strong core implementation for:
- unconditional DiMA training/inference,
- encoder support (ESM2/CHEAP/SaProt/ESMc),
- key sequence metrics (pLDDT, ESM-PPPL, FD-seq, MMD, Wasserstein, diversity, novelty),
- family-generation pipeline skeleton.

However, several paper-level claims are **not fully backed by executable code** here. The largest gaps are:
1. **Conditional structure tasks are mostly scaffold code** (motif/fold conditioning has placeholders and hardcoded behavior).
2. **Several paper metrics are missing** (TM-score, BLAST identity, ProGen perplexity, structural ProteinMPNN-based distribution metrics, scRMSD/scTM).
3. **Key datasets/benchmarks are not provisioned in-repo** (AFDBv4-90 naming mismatch, CATH S40, RFDiffusion benchmark assets).
4. **Sequence infilling experiment code is absent** despite being described in detail.
5. **Full experiment reproducibility layer is incomplete** for many tables/baselines/biological analyses.

---

## A. Data & dataset gaps

## A1) AFDB variant mismatch (paper: AFDBv4-90; repo default: AFDB-v2)
- Paper repeatedly describes AFDBv4-90 as the primary large-scale dataset.
- Repo dataset configs and automation point to `AFDB-v2` / `AFDB-v2-64-510` names.

**Impact:** Reproducing paper numbers may fail or produce non-comparable results unless dataset versions are manually reconciled.

## A2) No in-repo CATH S40 dataset preparation for fold-conditioning benchmark
- Paper reports fold-conditioned fine-tuning/evaluation on CATH S40 (27k proteins).
- No dedicated CATH data loader/preparation/evaluation pipeline is present.

**Impact:** Fold-conditioned benchmark from paper cannot be reproduced out-of-the-box.

## A3) Motif-scaffolding benchmark data assets absent
- Paper uses the 24-task RFDiffusion motif benchmark.
- Code accepts optional benchmark JSON paths, but benchmark assets/parsers are not bundled.

**Impact:** Reported solved-task counts and per-problem success rates are not directly reproducible.

## A4) Family annotations are not guaranteed in dataset; fallback is synthetic
- Paper family experiments rely on InterPro/Pfam-aware family labels.
- Code falls back to heuristic/synthetic partitioning when annotations are unavailable.

**Impact:** Family-generation results may diverge significantly from paper unless matching annotated datasets are externally prepared.

---

## B. Model / method implementation gaps

## B1) Motif scaffolding implementation is incomplete
In `src/conditional_generation/conditional_gen.py`:
- PDB motif loading raises `NotImplementedError`.
- Structure encoding path returns placeholder `None`.
- Sampling loop notes conditioning is not actually applied.

**Impact:** Motif scaffolding is not implemented to paper level (no real motif extraction/conditioning).

## B2) Fold-conditioning implementation is mostly placeholder
In the same module:
- PDB length extraction returns hardcoded length (`200`).
- Fold-conditioning reverse process does not use real structural constraints beyond target length.

**Impact:** Reported fold-conditional behavior (TM-score-targeted generation) is not faithfully implemented.

## B3) Conditional training path is not wired end-to-end
`train_conditional.py` calls `score_estimator(..., condition=condition)` but base `ScoreEstimator` does not accept a `condition` argument.

**Impact:** Conditional trainer is inconsistent with the core denoiser API and is likely non-functional without additional patches.

## B4) Conditional score estimator exists but is not integrated
A `ConditionalScoreEstimator` class is defined in conditional code, but no robust integration replaces the base denoiser in the main trainer/model stack.

**Impact:** Cross-attention conditioning described in the paper is only partially scaffolded.

## B5) Family CFG generation mode is explicitly unimplemented
`src/family_generation/generate.py` contains `NotImplementedError` for CFG generation requiring separate fine-tuned model support.

**Impact:** One of the paper’s family-generation pathways is incomplete.

## B6) Sequence infilling method (adapter + protocol) missing
Paper appendix describes a dedicated infilling setup (adapter architecture, masking protocol, evaluation criteria). No corresponding infilling module/trainer/evaluator is present.

**Impact:** Sequence infilling claims cannot be reproduced from current code.

## B7) Flow-matching ablation path not present as a runnable alternative
Paper includes diffusion-vs-flow-matching discussion/ablation, but repo dynamics/training expose SDE diffusion path only.

**Impact:** Architectural ablation coverage in paper is only partially represented in code.

---

## C. Metric gaps vs paper

## C1) Missing ProGen perplexity metric
- Paper reports ProGen2-based perplexity.
- Repo includes ESM pseudo-perplexity (`esmpppl.py`) but no ProGen perplexity implementation.

## C2) Missing TM-score evaluation implementation
- Paper uses TM-score for structural relevance and fold-conditioned success.
- No TM-score metric module / structure alignment pipeline is present.

## C3) Missing BLAST identity pipeline
- Paper includes BLAST identity and BLAST+NW novelty procedure.
- Repo novelty/diversity implementations use fast ungapped identity heuristics, not BLAST/NW.

## C4) Missing structure-distribution metrics (ProteinMPNN-based FD/MMD/OT)
- Paper describes structural analogues of distribution metrics with ProteinMPNN embeddings.
- Repo FD/MMD/Wasserstein are implemented only in sequence embedding space (ProtT5).

## C5) Missing PCD/NCD co-clustering metrics
- Paper defines positive/negative co-clustering diversity metrics.
- No explicit implementation found.

## C6) Missing structure-generation metrics for CHEAP experiments
- Paper reports scRMSD/scTM and designability success criteria for co-design/structure-only protocols.
- No dedicated implementation pipeline for these metrics exists.

## C7) Missing UMAP analysis utilities referenced in paper
- Paper includes UMAP distribution analysis.
- No script to regenerate UMAP panels from generated/reference protein sets.

---

## D. Experiment reproducibility gaps

## D1) End-to-end scripts for many reported tables are absent
The repo has convenience scripts (`auto-scripts/`) and general evaluators, but no table-locked reproducer scripts for all major paper sections (main tables + appendix benchmarks).

## D2) Baseline comparison pipeline is incomplete
Paper compares against many external models (RITA, ProGen2, ProLLAMA, EvoDiff, DPLM, DPLM2, Chroma, Multiflow, RFDiffusion, PLAID, etc.).
There is no unified, version-pinned runner in repo to regenerate those exact benchmark comparisons.

## D3) Biological relevance analyses are partially missing
Paper references InterProScan/MobiDB analyses and broader biological diagnostics.
Repo includes family-focused InterPro wrapper, but not the full paper-level biological relevance analysis pipeline.

## D4) Structure-side experiments need external tooling glue
Paper protocols rely on Foldseek/ProteinMPNN/TM-align-like structural evaluation components. These are not provided as complete reproducible workflows here.

---

## E. Present-but-partial elements (important nuance)

These are **not fully missing**, but not yet paper-complete:

- Conditional generation modules exist but include placeholders and weak wiring.
- Family generation is implemented for classifier guidance, but CFG path is not finished.
- InterPro-based fidelity evaluation exists but depends on external install and is not a full biological-analysis suite.
- Distribution metrics are available for sequence embeddings, but not for structural embeddings.

---

## F. Recommended priority fixes

If the goal is paper-level reproducibility, prioritize in this order:

1. **Complete conditional pipelines** (motif/fold): PDB parsing, conditioning integration, and proper training/eval wiring.
2. **Implement missing core metrics**: TM-score, BLAST/NW novelty/identity, ProGen perplexity.
3. **Add structure-side metric stack**: ProteinMPNN embedding metrics, scRMSD/scTM protocols.
4. **Ship benchmark data adapters**: AFDBv4-90 compatibility, CATH S40 prep, motif benchmark manifests.
5. **Add reproducibility scripts per table/section** with pinned configs/checkpoints/dependencies.
6. **Implement infilling module** matching the appendix protocol.

---

## G. Concrete missing items checklist

- [ ] AFDBv4-90 canonical dataset path/config parity.
- [ ] CATH S40 dataset loader + split scripts.
- [ ] RFDiffusion motif benchmark assets and parser utilities.
- [ ] Full motif scaffolding conditioning (remove placeholders).
- [ ] Full fold-conditioned structural conditioning (remove hardcoded length fallback).
- [ ] ScoreEstimator conditional interface integration.
- [ ] Family CFG generation end-to-end implementation.
- [ ] Sequence infilling training + inference + evaluation scripts.
- [ ] ProGen perplexity metric implementation.
- [ ] TM-score metric implementation and fold benchmark evaluator.
- [ ] BLAST identity + NW novelty metric pipeline.
- [ ] ProteinMPNN-based FD/MMD/OT structural metrics.
- [ ] PCD/NCD co-clustering metrics.
- [ ] scRMSD/scTM metrics for structure generation protocols.
- [ ] UMAP analysis script for positive-cluster visualizations.
- [ ] Repro scripts for all primary paper tables/appendix sections.

