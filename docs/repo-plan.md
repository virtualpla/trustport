# Repository Plan

## Directory tree

```
trustport/
  __init__.py
  manifest.py           cargo manifest: frozen config dataclasses + record schemas
  harbor.py             assembled end-to-end model (Harbor) tying the four modules together
  berths/               arrival quays — incoming consignments
    __init__.py
    consignments.py     typed records (Consignment, Language, Modality, BatchTensors)
    arrivals.py         deterministic synthetic cohort + manifest loader for real corpora
    stowage.py          batching / collation (no torch DataLoader)
  customs/              declaration processing — Module 1
    __init__.py
    backbones.py        Backbone Protocol + DeterministicBackbone surrogate + guarded HF
    ledger.py           LoRA adapter (Eq 1)
    declarations.py     NER / RE / classification / QA task heads
    seals.py            structured-output enforcement: JSON schema + format penalty (Eq 2)
  appraisal/           valuation / tariff — Module 2
    __init__.py
    rubrics.py          P / M / C dimension rubrics + weights
    brokerage.py        BM25 top-5 retrieval (RAG anchors)
    tariff.py           PMC-Index scoring (Eq 3)
  consolidation/       container stuffing — Module 3
    __init__.py
    crossties.py        Hadamard cross-modal interactions
    containerize.py     attention-weighted cross-modal fusion (Eq 4) + pathway clustering
  inspection/          examination / hold-or-release — Module 4
    __init__.py
    holds.py            hallucination detection (Eq 5)
    clearance.py        composite trust score + review flagging (Eq 6) + stage detection
  tally/               tally office — evaluation + statistics
    __init__.py
    integrated.py       Integrated F1 (Eq 7) + stage weights + enforcement gain
    agreement.py        ICC(2,1) / Pearson r / MAE / Cohen's kappa / ECE
    significance.py     DeLong / BCa bootstrap / Holm-Bonferroni / BH-FDR / permutation / Cohen's d
  voyage/              sailing schedule — training
    __init__.py
    objective.py        uncertainty-weighted multi-task objective (Eq 2)
    helmsman.py         training loop, optimizer, scheduler, EMA
    logbook.py          atomic checkpoint write + seed restore
  quay/                quayside services — utilities
    __init__.py
    beacon.py           seeding
    signals.py          logging
    cargo_io.py         atomic file io
    moorings.py         device selection
    types.py            shared TypedDicts / aliases
  wheelhouse/          bridge controls — CLI (simple-parsing subcommands)
    __init__.py
    __main__.py         entry dispatch
    commands.py         train / appraise / fuse / inspect / evaluate / export dataclass commands
    settings.py         OmegaConf load -> frozen dataclass materialization
    export.py           ONNX export (dynamo=False)
configs/
  model/{llama31_8b,llama31_70b,qwen25_72b,deepseek_v3,chatglm4}.yaml
  data/{english_ner,english_re,english_qa,chinese_ner,policy_corpus}.yaml
  train/default.yaml
  experiment/{main,ablation_domain,ablation_structout,ablation_fusion,ablation_trust,
              ablation_rag,supplementary_crosslang,supplementary_pathway,_smoke}.yaml
tests/proofs/
  test_ledger_lora.py            shape + Eq-1 scaling invariant
  test_seals_enforcement.py      JSON-schema validity + format penalty monotonicity
  test_tariff_pmc.py             PMC weighting + dimension bounds
  test_fusion_crossmodal.py      fusion shape + modality-dropout invariance + gradient flow
  test_inspection_trust.py       hallucination bounds + multiplicative trust decay (Eq 6)
  test_integrated_f1.py          weakest-stage-dominates property + reference values
  test_agreement_stats.py        ICC / kappa / ECE vs reference
  test_significance.py           DeLong / bootstrap / Holm / permutation vs scipy/statsmodels
  test_overfit_single_batch.py   Harbor overfits one batch
  test_training_smoke.py         2 steps on _smoke.yaml, loss decreases
  test_house_style.py            AST+tokenize no-comment / no-docstring + forbidden-phrase/emoji
docs/
  project-context.md
  implementation-map.md
  repo-plan.md
  deviations.md
scripts/
  launch_train.sh
  launch_eval.sh
  prepare_data.sh
```

## Module responsibilities

- `berths` owns all data: typed `Consignment` records, a deterministic synthetic cohort whose
  latent extraction signal is recoverable, and a manifest loader for the real credentialed corpora.
- `customs` owns Module 1: the `Backbone` Protocol, LoRA adapters (Eq 1), the four task heads, and
  structured-output enforcement (Eq 2). Real LLM backbones are guarded; the default is a
  deterministic hashed-feature surrogate.
- `appraisal` owns Module 2: BM25 retrieval anchors and PMC-Index dimension scoring (Eq 3).
- `consolidation` owns Module 3: Hadamard cross-modal interactions and attention-weighted fusion
  (Eq 4), plus retrospective pathway clustering.
- `inspection` owns Module 4: per-stage hallucination detection (Eq 5) and composite trust scoring
  with review flagging (Eq 6).
- `tally` owns evaluation: Integrated F1 (Eq 7) and the full statistical battery.
- `voyage` owns optimization; `quay` owns cross-cutting utilities; `wheelhouse` owns the CLI.

## Dependencies (pinned)

- torch >= 2.2, < 2.10
- numpy >= 1.26, < 3
- scipy >= 1.11
- omegaconf >= 2.3
- simple-parsing >= 0.1.6
- rank-bm25 >= 0.2.2
- typing-extensions >= 4.9

Dev: ruff, black, isort, mypy, pytest, hypothesis. Optional `[hf]`: transformers, accelerate.
Optional `[export]`: onnx, onnxscript.

## Expected test coverage

Eleven test modules spanning shape, invariant, overfit-single-batch, metric-vs-reference,
gradient-flow, numerical-regression, and end-to-end smoke, plus a house-style guard (no comments /
docstrings, no forbidden phrases / emoji). Targets ≥ 35 test functions; every science module has at
least one numerical-correctness check.
