# Implementation Map

Paper provenance lives here (source files carry no comments or docstrings). Each row links a
paper artifact to the file and symbol that realizes it.

## Metaphor key (maritime cargo port / customs-clearance terminal)

| package path | port role | paper module |
|---|---|---|
| `trustport/berths/` | arrival quays — incoming consignments | document streams + datasets (clinical / policy / KBI) |
| `trustport/customs/` | declaration processing | Module 1 domain-adapted LLM extraction |
| `trustport/appraisal/` | valuation / tariff | Module 2 PMC-Index scoring |
| `trustport/consolidation/` | container stuffing | Module 3 cross-modal fusion |
| `trustport/inspection/` | examination / hold-or-release | Module 4 trustworthiness verification |
| `trustport/tally/` | tally office — audit of cleared goods | evaluation metrics + statistics |
| `trustport/voyage/` | sailing schedule | training loop / optim / checkpoint |
| `trustport/quay/` | quayside services | utilities (seed / log / io / device / types) |
| `trustport/wheelhouse/` | bridge controls | CLI |
| `trustport/manifest.py` | cargo manifest | typed configuration + record schemas |
| `trustport/harbor.py` | the harbor | assembled end-to-end model |

## Equations

| paper | eq/fig/table | file | symbol | notes |
|---|---|---|---|---|
| Methods "Domain-adapted ... extraction" | Eq 1 | `customs/ledger.py` | `LoraAdapter`, `lora_delta` | θ_t = θ_base + B_tA_t; r∈{8,16,32,64}; scaling α/r with α=2r; dropout 0.05 |
| Methods "Domain-adapted ... extraction" | Eq 2 | `customs/seals.py`, `voyage/objective.py` | `format_penalty`, `MultiTaskObjective` | L = Σ λ_t L_t + γ L_format; γ=0.3; uncertainty-weighted λ; converged λ 0.28/0.31/0.22/0.19 |
| Methods "Automated PMC-Index ..." | Eq 3 | `appraisal/tariff.py` | `pmc_index`, `DimensionScorer` | PMC = Σ α_j s_j(d\|rubric_j, RAG_j); α_P 0.35 / α_M 0.30 / α_C 0.35 |
| Methods "Automated PMC-Index ..." | BM25 top-5 + CoT | `appraisal/brokerage.py` | `Bm25Retriever`, `retrieve` | up to 5 expert-scored reference policies as calibration anchors |
| Methods "Information fusion ..." | Eq 4 | `consolidation/containerize.py`, `consolidation/crossties.py` | `CrossModalFusion`, `hadamard_interactions` | z_fused = Σ w_i f_i + W_cross[f_c⊙f_p; f_p⊙f_k; f_c⊙f_k]; attention-weighted w_i |
| Methods "Information fusion ..." | Eq 5 | `inspection/holds.py` | `hallucination_score` | ĥ_s = 1 − ⅓[ref_s + consist_s + schema_s]; K=3 self-consistency at τ=0.7 |
| Methods "Information fusion ..." | Eq 6 | `inspection/clearance.py` | `trust_score`, `flag_for_review` | T(x)=Π_s[(1−ĥ_s)·c_s]; τ_trust 0.6; cross-document consistency c_s |
| Methods "Evaluation metrics" | Eq 7 | `tally/integrated.py` | `integrated_f1` | IntF1=(Σ_k w_k/F1_k)^−1; w_NER 0.35 / w_RE 0.40 / w_Cls 0.25 |

## Reported tables / figures

| paper | item | file | symbol | notes |
|---|---|---|---|---|
| Table 1 | multi-task extraction (7 datasets, 14 baselines) | `tally/integrated.py`, `tally/significance.py` | `integrated_f1`, `holm_bonferroni` | Ours Integrated 0.867; per-dataset BC5CDR 0.953 / n2c2 0.922 / ChemProt 0.752 / MedQA 0.787 / CCKS-17 0.954 / CCKS-19 0.890 |
| Table 2 / Fig 2 | cross-model comparison (Δ_enforce) | `customs/seals.py`, `tally/integrated.py` | `enforcement_gain` | 8B 0.842 (+5.1pp) / 70B 0.867 (+2.2) / Qwen 0.861 / DeepSeek 0.858 / ChatGLM 0.849; GPT-4 0.795 |
| Table 3 / Table 12 | PMC-Index ICC vs expert | `tally/agreement.py` | `icc_2_1`, `pearson_r`, `mae` | Overall ICC 0.84 [0.74,0.91]; Content 0.89 / Policy 0.81 / Measure 0.76 |
| Table 4 | component ablation + interaction ratio | `harbor.py`, `configs/experiment/ablation_*.yaml` | `Harbor.toggle` | drops: DomainAdapt −4.9 / StructOut −3.3 / Fusion −2.0 / Trust −1.5 / RAG −0.9; IR DomainAdapt∩StructOut 1.21 |
| Table 5 / Table 15 / Fig 3 | deployment feasibility | `wheelhouse/export.py`, `docs/...` | `export_onnx` | INT4/INT8/FP16 footprints; Pareto frontier |
| Table 6 / Table 17 | retrospective pathway clusters | `consolidation/containerize.py`, `tally/agreement.py` | `pathway_clusters`, `cohen_kappa` | 4 clusters; concordance 0.82 / κ 0.72 |
| Table 8 | full hyperparameters | `configs/experiment/main.yaml`, `configs/model/*.yaml` | — | per-backbone r/α/lr/batch/accum/epochs/warmup/wd/seq-len/precision |
| Table 9 | per-seed Integrated F1 | `tally/integrated.py` | `integrated_f1` | seeds {42,123,456,789,1024}; mean 0.867 std 0.0018 |
| Table 13 | stage-wise error detection | `inspection/clearance.py` | `stage_detection` | end-to-end error 12.3% / detection 71.4% / FP 4.8% / trust 0.76 |
| Table 14 | trust-score calibration / ECE | `tally/agreement.py` | `expected_calibration_error` | overall ECE 0.042 |
| Table 16 | top-3 baseline reproduction | `tests/proofs/test_significance.py` | — | reproductions within 1.5pp |

## Datasets and stage weights

| paper | item | file | symbol |
|---|---|---|---|
| Table 7 | dataset registry + splits | `berths/arrivals.py`, `configs/data/*.yaml` | `DATASETS`, `SyntheticArrivals` |
| Methods "Evaluation metrics" | calibrated stage weights | `tally/integrated.py` | `STAGE_WEIGHTS` (w_NER 0.35 / w_RE 0.40 / w_Cls 0.25) |
| Extended Methods "Multilingual" | En/Zh routing | `berths/consignments.py` | `Language` |

## Statistics

| paper | method | file | symbol |
|---|---|---|---|
| Table 3 | ICC(2,1) two-way random single measures | `tally/agreement.py` | `icc_2_1` |
| Table 6/17 | Cohen's κ | `tally/agreement.py` | `cohen_kappa` |
| Discussion | Cohen's d (pooled-seed std) | `tally/significance.py` | `cohens_d` |
| Methods "Statistical analysis" | DeLong AUROC test | `tally/significance.py` | `delong_test` |
| Methods "Statistical analysis" | BCa bootstrap (1,000) | `tally/significance.py` | `bca_bootstrap_ci` |
| Methods "Statistical analysis" | Holm-Bonferroni (35 comparisons) | `tally/significance.py` | `holm_bonferroni` |
| SI "exploratory analyses" | BH-FDR | `tally/significance.py` | `benjamini_hochberg` |
| Methods "Statistical analysis" | paired permutation (10,000) | `tally/significance.py` | `permutation_test` |
| Table 14 | expected calibration error | `tally/agreement.py` | `expected_calibration_error` |
