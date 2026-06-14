# Project Context

    project_name       : trustport                                    [HIGH]
    domain             : hospital information integration NLP —        [HIGH]
                         trustworthy multi-source extraction + policy
                         evaluation for smart public hospitals
    framework          : PyTorch 2.x + plain torch.nn                  [HIGH]
    venue              : Scientific Reports                            [HIGH]
    primary_datasets   : 9 datasets (see §6)                           [HIGH]
    compute_target     : ~120 A100 GPU-hours total                     [HIGH]
    hparams_reference  : Table 8 (+ Table 7, Extended Methods)         [HIGH]
    supp_path          : none (SI embedded, pp. 16-22)

    NEEDS_USER_DECISION: 0

The framework unifies four modules behind one domain-adapted LLM backbone: (1) multi-task
information extraction with structured-output enforcement, (2) automated PMC-Index policy
scoring, (3) cross-modal fusion of clinical / policy / knowledge-based features, and (4)
stage-wise trustworthiness verification. End-to-end extraction quality is measured by a weighted
harmonic Integrated F1.

## 1. project_name
`trustport`. From the title "Trustworthy LLM-Augmented Information **Integration** ... Smart
Hospitals" — `trust` (trustworthiness, the headline contribution) + `port` (the port-of-entry /
customs-clearance metaphor used throughout the package, since the system clears heterogeneous
document streams through declaration → appraisal → consolidation → inspection). Abstract L1-3,
Methods "Framework overview". Confidence HIGH.

## 2. supp_path
`none`. The Supplementary Information is embedded in the same PDF (pp. 16-22: Extended
Introduction / Results / Discussion / Methods, Supplementary Tables 7-17). A sibling glob
(`./paper/*supp*`, `./*_si.*`, `./appendix*`) found no separate file. Confidence HIGH.

## 3. domain
Hospital information integration NLP for smart public hospitals: trustworthy multi-task clinical
extraction (NER → RE → classification → QA), automated quantitative scoring of healthcare policy
text (PMC-Index), cross-modal fusion across clinical NLP / policy / structured KBI signals, and
hallucination-aware trustworthiness verification across a multi-stage pipeline. Abstract,
Introduction, Methods "Framework overview". Confidence HIGH.

## 4. framework
PyTorch 2.x with plain `torch.nn`. The paper fine-tunes open-source LLMs (Llama 3.1, Qwen 2.5,
DeepSeek-V3, ChatGLM-4) via LoRA (Eq 1), uses JSON-schema constrained decoding + a format-aware
loss (Eq 2), attention-weighted fusion (Eq 4), and ONNX export for deployment (Table 5/15). The
70B/MoE backbones cannot be run here; this release places a `Backbone` Protocol in front of a
deterministic, recoverable-signal surrogate plus guarded real-HF adapters (raise honestly when
`transformers`/weights are absent). The LoRA math, multi-task loss, PMC scoring, fusion,
trustworthiness, Integrated F1 and statistics are implemented exactly. Methods, Fig. 4/5, Table 8.
Confidence HIGH.

## 5. venue
Scientific Reports. Evidence: structured-abstract single-block layout, the "Author contributions
/ Competing interests / Ethics declarations / Additional information" trailer that is the Nature
Portfolio (Scientific Reports) house structure, and self-citations to "*Sci. Reports*" (refs 12,
15, 39). Confidence HIGH.

## 6. primary_datasets
From Table 7 (p. 19) and Data availability (p. 13).

| name | version | task | lang | license / access |
|---|---|---|---|---|
| BC5CDR | BioCreative V | NER (2 ent) | En | public — biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus |
| n2c2 2022 | track-1 | NER (16 ent) | En | restricted DUA — n2c2.dbmi.hms.harvard.edu/2022-track-1 |
| ChemProt | BioCreative VI | RE (13 rel) | En | public — BioCreative VI |
| MedQA (MedBench) | OpenCompass | QA (4 cls) | Zh | public — medbench.opencompass.org.cn |
| CliMedBench | EMNLP 2024 | QA (14 cls) | Zh | public — github.com/Optifine-TAT/CliMedBench (cross-lingual validation only) |
| CCKS-2017 | CCKS shared task | NER (5 ent) | Zh | public — biendata.xyz/competition/CCKS2017_2 |
| CCKS-2019 | CCKS shared task | NER (6 ent) | Zh | public — CCKS shared task |
| MIMIC-IV | v2.2 | Multi | En | PhysioNet Credentialed Health Data License — physionet.org/content/mimiciv/2.2 |
| Policy Corpus | 77 docs (50/27) | Scoring (3-dim) | Zh | NHC gazette (public); PMC-Index upon written request to the corresponding author |

Confidence HIGH.

## 7. compute_target
~120 A100 GPU-hours total budget (Methods p. 13). 8B fine-tune ≈ 15 min/config; 70B-class ≈ 90
min; bf16 mixed precision (Table 8). Deployment (Table 5/15): Llama 3.1-8B FP16 18 GB / INT8 10
GB / INT4 6 GB; 8B-INT4 processes ~2,000 hospital documents in ~7 h on a single GPU at
$0.0006/1K tokens; A100 80 GB latency 95-1,200 ms. COMPUTE fully reported. Confidence HIGH.

## 8. hparams_reference
Table 8 (full per-backbone configuration: LoRA rank r ∈ {16,32}, α = 2r, dropout 0.05, lr
2e-4 [Qwen 1e-4], batch 8 [70B 4], grad-accum 2 [70B/MoE 4], max-epochs 10, early-stop patience 3,
warmup 0.06 [72B 0.03], weight-decay 0.01, max-seq-len 2,048 [72B/MoE 4,096], bf16). Table 7
(dataset splits / class counts). Extended Methods (initial task weights λ_NER 1.0 / λ_RE 1.5 /
λ_Cls 0.8 / λ_QA 0.8 → converged 0.28 / 0.31 / 0.22 / 0.19; γ_format 0.3; PMC α_P 0.35 / α_M 0.30
/ α_C 0.35; K=3 self-consistency at τ 0.7; τ_trust 0.6; fusion weights w_NER 0.35 / w_RE 0.40 /
w_Cls 0.25). 5 seeds {42,123,456,789,1024}. Confidence HIGH.

## 9. extra_signals
- Released artifacts: pre-trained LoRA adapters for the 4 backbones under Apache-2.0 (Code
  availability statement, p. 14).
- Algorithm content: 7 numbered equations (Eq 1-7); no boxed algorithms.
- SI-only experiments: per-seed Integrated F1 (Table 9), estimation methodology (Table 10),
  cross-language comparison (Table 11), per-category PMC ICC (Table 12), stage-wise error
  detection (Table 13), trust-score calibration / ECE (Table 14), extended deployment (Table 15),
  baseline reproduction (Table 16), per-cluster pathway analysis (Table 17).
- Deterministic seeds and the requirement that every checkpoint stores and restores its seed.
- Code-availability statement (verbatim, kept out of README per house rule): "We will make
  available code for our LLM augmented Information Integration Framework, which includes: LoRA
  fine-tuning scripts, a module to enforce structured output, an automated scoring pipeline for
  use with PMC-Index, and a verification of Trustworthiness by stage. Publication will be on
  https://github.com/[anonymized]/trustworthy-llm-hospital, and will include pre-trained LoRA
  adapters for each of the 4 backbone models under the Apache 2.0 licence."
- Ethics: retrospective de-identified public data; no additional ethical approval required;
  MIMIC-IV / n2c2 require credentialed access / DUA.
