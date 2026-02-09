# Modular ML Pipeline for DiMA

## Objective

Transform DiMA's existing monolithic training/inference workflow into a **modular, pluggable pipeline** where:

1. **Encoders and decoders** are registered via a plugin registry and swappable at configuration time
2. **Dataset setup** is automated and extensible for new data sources
3. **Training** (diffusion + optional decoder), **inference**, and **metrics** are orchestrated through a single pipeline runner
4. **Third-party modules** can be plugged in without modifying core code

---

## Current State Analysis

### What exists today
- `src/encoders/` — 4 encoder implementations (ESM2, CHEAP, SaProt, ESMc) with a base class
- `src/configs/` — Hydra YAML configs for each component
- `src/diffusion/base_trainer.py` — monolithic training loop that handles training, validation, generation, and metrics
- `src/preprocessing/` — separate scripts for statistics calculation and decoder training
- `auto-scripts/` — standalone automation scripts (setup_models, prepare_data, run_inference, calc_metrics) that are loosely coupled and hard to extend
- `src/metrics/` — 4 metric implementations (FID, MMD, pLDDT, ESM-pPPL) dispatched through `compute_ddp_metric`

### Gaps
| Gap | Impact |
|-----|--------|
| No formal plugin/registry pattern | Adding a new encoder or decoder requires editing multiple files |
| No pipeline orchestrator | Each stage (stats, decoder training, diffusion training, inference, metrics) is run manually with separate scripts |
| Decoder is tightly coupled to encoders | Each encoder file hard-codes its decoder logic |
| Metrics dispatch uses if/elif chains | Adding a new metric requires editing `metric.py` |
| No validation that components are compatible | Mismatched embedding dimensions cause runtime errors |
| Auto-scripts don't compose | No way to run "setup -> train decoder -> train diffusion -> infer -> evaluate" as one pipeline |
| No third-party plugin interface | External modules must be copy-pasted into `src/` |

---

## Architecture Design

### Directory Structure (new/modified files marked with `*`)

```
DiMA/
├── PLAN.md                              *
├── pipeline/                            * NEW top-level pipeline package
│   ├── __init__.py                      *
│   ├── registry.py                      * Component registry (encoders, decoders, metrics, datasets)
│   ├── plugin_loader.py                 * Dynamic third-party plugin loading
│   ├── orchestrator.py                  * Pipeline orchestrator (runs stages in sequence)
│   ├── stages/                          * Individual pipeline stages
│   │   ├── __init__.py                  *
│   │   ├── base_stage.py               * Abstract base class for all stages
│   │   ├── setup_data.py               * Stage: dataset download & preparation
│   │   ├── setup_models.py             * Stage: model/encoder download & caching
│   │   ├── calculate_statistics.py      * Stage: compute normalization statistics
│   │   ├── train_decoder.py            * Stage: decoder training
│   │   ├── train_diffusion.py          * Stage: diffusion model training
│   │   ├── run_inference.py            * Stage: sample generation
│   │   └── evaluate_metrics.py         * Stage: metric computation
│   ├── configs/                         * Pipeline-level run configurations
│   │   ├── full_pipeline.yaml           * End-to-end pipeline config
│   │   ├── inference_only.yaml          * Inference + metrics only
│   │   └── train_only.yaml             * Training stages only
│   └── tests/                           * Pipeline tests
│       ├── __init__.py                  *
│       ├── test_registry.py             *
│       ├── test_plugin_loader.py        *
│       ├── test_stages.py              *
│       └── test_orchestrator.py         *
├── src/
│   ├── encoders/
│   │   ├── base.py                      (unchanged — already a good interface)
│   │   └── ...
│   ├── decoders/                        * NEW — extract decoder logic from encoders
│   │   ├── __init__.py                  *
│   │   ├── base.py                      * Abstract decoder base class
│   │   ├── lm_head.py                   * Wrapper around encoder's lm_head
│   │   └── transformer.py              * Moved from src/encoders/transformer_decoder.py
│   ├── metrics/
│   │   ├── base.py                      * Abstract metric base class
│   │   └── ...
│   └── ...
└── run_pipeline.py                      * NEW top-level entry point
```

### Core Abstractions

#### 1. Component Registry (`pipeline/registry.py`)

A singleton registry that maps string keys to component classes. All built-in components are auto-registered at import time. Third-party plugins register themselves via the same mechanism.

```python
# Usage:
registry.register("encoder", "esm2", ESM2EncoderModel)
registry.register("decoder", "transformer", TransformerDecoder)
registry.register("metric", "fid", FIDMetric)

encoder_cls = registry.get("encoder", "esm2")
```

**Design decisions:**
- Simple dict-of-dicts, no metaclass magic
- Registration via decorators (`@registry.register_encoder("my_encoder")`) for convenience
- Validation: registry checks that registered classes implement the required base interface

#### 2. Plugin Loader (`pipeline/plugin_loader.py`)

Loads third-party Python modules from a configurable plugins directory or from installed packages.

```python
# In pipeline config:
plugins:
  - path: "./my_plugins/custom_encoder.py"
  - package: "dima_plugin_esmfold"  # pip-installable
```

**How it works:**
- Scans specified paths/packages for modules containing `register(registry)` functions
- Calls each plugin's `register()` to let it add components to the registry
- Plugins can register encoders, decoders, metrics, or even custom pipeline stages

#### 3. Pipeline Stages (`pipeline/stages/base_stage.py`)

Each stage is a self-contained unit with:
- `validate(config)` — check preconditions (files exist, dimensions match, etc.)
- `run(config, context)` — execute the stage
- `context` is a shared dict that stages can write to (e.g., statistics path, checkpoint path)

#### 4. Orchestrator (`pipeline/orchestrator.py`)

Reads a pipeline YAML config and runs stages in sequence:

```yaml
# pipeline/configs/full_pipeline.yaml
stages:
  - name: setup_data
    enabled: true
  - name: setup_models
    enabled: true
  - name: calculate_statistics
    enabled: true
  - name: train_decoder
    enabled: true        # set false if encoder has built-in decoder
  - name: train_diffusion
    enabled: true
  - name: run_inference
    enabled: true
  - name: evaluate_metrics
    enabled: true
    params:
      metrics: [fid, mmd, esm_pppl, plddt]
```

---

## Implementation Plan

### Commit 1: Component Registry and Plugin Loader

**Files created:**
- `pipeline/__init__.py`
- `pipeline/registry.py`
- `pipeline/plugin_loader.py`
- `pipeline/tests/__init__.py`
- `pipeline/tests/test_registry.py`
- `pipeline/tests/test_plugin_loader.py`

**What it does:**
- Implements `ComponentRegistry` with `register()`, `get()`, `list_components()`, `has()`
- Decorator support: `@registry.register_encoder("name")`
- Plugin loader that discovers and loads external modules
- Auto-registers all built-in encoders, decoders, and metrics
- Full unit tests

### Commit 2: Decoder Abstraction Layer

**Files created/modified:**
- `src/decoders/__init__.py`
- `src/decoders/base.py`
- `src/decoders/lm_head.py`
- `src/decoders/transformer.py`

**What it does:**
- Creates `BaseDecoder` abstract class with `forward(encodings, attention_mask)` and `decode_to_sequences(logits, tokenizer, attention_mask)` methods
- Wraps existing `TransformerDecoder` (from `src/encoders/transformer_decoder.py`) as a registered decoder
- Creates `LMHeadDecoder` that wraps any encoder's lm_head
- Encoders reference decoders through the registry rather than hard-coding them
- Backward-compatible: existing configs continue to work

### Commit 3: Metrics Base Class and Registry Integration

**Files created/modified:**
- `src/metrics/base.py`
- `src/metrics/fid.py` (add registry decorator)
- `src/metrics/mmd.py` (add registry decorator)
- `src/metrics/plddt.py` (add registry decorator)
- `src/metrics/esmpppl.py` (add registry decorator)
- `src/metrics/metric.py` (use registry instead of if/elif)

**What it does:**
- Creates `BaseMetric` with `compute(predictions, references, **kwargs)` interface
- Wraps each existing metric function in a class that implements `BaseMetric`
- Refactors `compute_ddp_metric` to look up metrics from the registry
- Third-party metrics just need to implement `BaseMetric` and register

### Commit 4: Pipeline Stages

**Files created:**
- `pipeline/stages/__init__.py`
- `pipeline/stages/base_stage.py`
- `pipeline/stages/setup_data.py`
- `pipeline/stages/setup_models.py`
- `pipeline/stages/calculate_statistics.py`
- `pipeline/stages/train_decoder.py`
- `pipeline/stages/train_diffusion.py`
- `pipeline/stages/run_inference.py`
- `pipeline/stages/evaluate_metrics.py`
- `pipeline/tests/test_stages.py`

**What it does:**
- Defines `BaseStage` with `validate()` and `run()` interface
- Wraps each existing automation script into a proper stage class
- Each stage reads from and writes to a shared pipeline context
- Stages are registered so third-party stages can be added

### Commit 5: Pipeline Orchestrator and Run Configs

**Files created:**
- `pipeline/orchestrator.py`
- `pipeline/configs/full_pipeline.yaml`
- `pipeline/configs/inference_only.yaml`
- `pipeline/configs/train_only.yaml`
- `run_pipeline.py`
- `pipeline/tests/test_orchestrator.py`

**What it does:**
- `PipelineOrchestrator` reads a pipeline YAML, resolves stages from registry, validates prerequisites, and runs them in order
- Pipeline context flows between stages (e.g., statistics stage outputs path that training stage consumes)
- `run_pipeline.py` is the single entry point:
  ```bash
  python run_pipeline.py --pipeline pipeline/configs/full_pipeline.yaml \
      --config-overrides encoder=esm2 datasets=afdb
  ```
- Error handling: if a stage fails, the pipeline logs the error and can optionally continue or abort
- Supports `--dry-run` to show what would happen without executing

### Commit 6: Tests and Validation

**What it does:**
- Runs all unit tests to verify registry, plugin loading, stages, and orchestrator
- Verifies backward-compatibility (existing `train_diffusion.py` and `example_simple.py` still work)
- Integration test: mock pipeline run through all stages

---

## Third-Party Plugin Interface

To create a plugin for DiMA, a third party creates a Python module with a `register` function:

```python
# my_plugin.py
from src.encoders.base import Encoder
from pipeline.registry import registry

class MyCustomEncoder(Encoder):
    def __init__(self, config, device, main_config, add_enc_normalizer=True):
        super().__init__(config=config, device=device,
                         decoder_type=main_config.decoder.decoder_type,
                         add_enc_normalizer=add_enc_normalizer)
        # Custom encoder initialization...

    def batch_encode(self, batch, max_sequence_len):
        # Custom encoding logic
        ...

    def batch_decode(self, encodings, attention_mask=None):
        # Custom decoding logic
        ...

def register(reg):
    """Called by DiMA's plugin loader."""
    reg.register("encoder", "my_custom_encoder", MyCustomEncoder)
```

Usage in pipeline config:
```yaml
plugins:
  - path: "./my_plugin.py"

# Then reference it in Hydra config overrides:
# encoder=my_custom_encoder
```

---

## Cross-Attention Support

For diffusion models with cross-attention (e.g., conditioning on auxiliary signals):

- The `ScoreEstimator` model already supports configurable attention heads
- The pipeline config can enable cross-attention via model config overrides:
  ```yaml
  model:
    config:
      add_cross_attention: true
      cross_attention_dim: 1024  # dimension of conditioning signal
  ```
- Third-party plugins can provide custom conditioning encoders that feed into the cross-attention layers

---

## Key Design Principles

1. **Backward-compatible**: All existing scripts and configs continue to work unchanged
2. **Convention over configuration**: Sensible defaults; only override what you need
3. **Fail fast**: Validate configurations before starting expensive training runs
4. **Composable**: Mix and match encoders, decoders, metrics, and stages
5. **Minimal core changes**: The `src/` directory gets small additions (base classes, registry decorators) but existing logic is not disrupted
