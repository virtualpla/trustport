# trustport — Bill of Lading

A shipping-document take on the trustworthy information-integration framework. Three hospital
document streams (clinical notes, NHC policy text, structured knowledge indicators) arrive as
cargo and are cleared through four ports of call: declaration (domain-adapted multi-task
extraction), appraisal (PMC-Index policy scoring), consolidation (cross-modal fusion), and
inspection (stage-wise trustworthiness verification). End-to-end clearance quality is read off a
weighted-harmonic Integrated F1.

---

## B/L No. TP-0001 — particulars

| field | entry |
|---|---|
| Shipper | Health Informatics authoring team |
| Consignee | Smart public hospital information desk |
| Vessel | trustport 0.1.0 |
| Port of loading | clinical NLP / policy / knowledge-based indicators |
| Port of discharge | integrated hospital policy evaluation |
| Carrier | PyTorch 2.x, single domain-adapted LLM backbone with LoRA task adapters |
| Marks | Llama 3.1 / Qwen 2.5 / DeepSeek-V3 / ChatGLM-4 (interchangeable backbones) |

---

## Cargo description — holds and contents

| hold (package) | contents | manuscript anchor |
|---|---|---|
| `berths/` | arriving consignments: typed records, synthetic cohort, manifest reader | Table 7 |
| `customs/` | declaration desk: backbone, LoRA adapters, four task heads, structured-output seals | Eq 1-2 |
| `appraisal/` | valuation desk: BM25 retrieval anchors, PMC-Index dimension scoring | Eq 3 |
| `consolidation/` | container yard: Hadamard interactions, attention-weighted fusion, pathway clusters | Eq 4 |
| `inspection/` | examination bay: hallucination detection, composite trust, review flagging | Eq 5-6 |
| `tally/` | tally office: Integrated F1, ICC / kappa / ECE, DeLong / bootstrap / Holm / permutation | Eq 7 |
| `voyage/` | sailing schedule: multi-task objective, trainer, atomic checkpoints | Eq 2 |
| `quay/` | quayside services: seeding, logging, atomic io, device selection | — |
| `wheelhouse/` | bridge controls: simple-parsing CLI, OmegaConf loader, ONNX export | — |

---

## Loading instructions — installation

pip:

    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e .

conda:

    conda env create -f environment.yml
    conda activate trustport
    python -m pip install --no-deps -e .

docker:

    docker build -t trustport:0.1.0 .
    docker run --rm trustport:0.1.0 evaluate --config configs/experiment/_smoke.yaml

---

## Stowage plan — where cargo sits

    trustport/
      manifest.py harbor.py
      berths/      consignments.py arrivals.py stowage.py
      customs/     backbones.py ledger.py declarations.py seals.py
      appraisal/   rubrics.py brokerage.py tariff.py
      consolidation/ crossties.py containerize.py
      inspection/  holds.py clearance.py
      tally/       integrated.py agreement.py significance.py
      voyage/      objective.py helmsman.py logbook.py
      quay/        beacon.py signals.py moorings.py cargo_io.py types.py
      wheelhouse/  __main__.py commands.py settings.py export.py

Experiment papers sit in `configs/experiment/`. Each `ablation_*` and `model/*` file extends
`main.yaml` through an OmegaConf `extends:` key, so only the changed keys appear.

---

## Routing — bridge commands

    python -m trustport.wheelhouse train    --config configs/experiment/main.yaml --out runs/main --steps 200
    python -m trustport.wheelhouse evaluate  --config configs/experiment/main.yaml --checkpoint runs/main/checkpoint.pt
    python -m trustport.wheelhouse appraise  --config configs/experiment/main.yaml
    python -m trustport.wheelhouse fuse      --config configs/experiment/main.yaml
    python -m trustport.wheelhouse inspect   --config configs/experiment/main.yaml
    python -m trustport.wheelhouse export    --config configs/experiment/main.yaml --out runs/main/ner.onnx

---

## Clearance and inspection — declared values

The figures below are the values declared in the manuscript. Each row names the endpoint that
exercises it and the command that runs it. Reproducing the declared figures needs the credentialed
corpora and a production backbone; the bundled synthetic cohorts exercise the same code paths at a
reduced scale (see `docs/deviations.md`).

| endpoint | command | declared value |
|---|---|---|
| Integrated F1 (Llama 3.1-70B) | `wheelhouse evaluate experiment=main` | 0.867 ± 0.001 |
| Integrated F1 (Llama 3.1-8B) | `wheelhouse evaluate model=llama31_8b` | 0.842 ± 0.005 |
| BC5CDR NER F1 | `wheelhouse evaluate experiment=main` | 0.953 ± 0.002 |
| n2c2 NER F1 | `wheelhouse evaluate experiment=main` | 0.922 ± 0.004 |
| ChemProt RE F1 | `wheelhouse evaluate experiment=main` | 0.752 ± 0.006 |
| MedQA accuracy | `wheelhouse evaluate experiment=main` | 0.787 ± 0.005 |
| CCKS-2017 / CCKS-2019 NER F1 | `wheelhouse evaluate experiment=main` | 0.954 / 0.890 |
| PMC-Index ICC vs expert panel | `wheelhouse appraise experiment=main` | 0.84 [0.74, 0.91] |
| Pathway concordance / kappa | `wheelhouse fuse experiment=supplementary_pathway` | 0.82 / 0.72 |
| End-to-end trust / flagged / ECE | `wheelhouse inspect experiment=main` | 0.76 / 12.3% / 0.042 |

Structured-output enforcement gain Delta_enforce ranges +2.2 pp (70B) to +5.1 pp (8B); all four
open-source backbones clear the GPT-4 reading of 0.795 (Table 2).

---

## Demurrage and freight — compute budget

| stage | hardware | figure |
|---|---|---|
| Fine-tuning budget | A100 80 GB | ~120 GPU-hours total |
| 8B fine-tune | 1 x A100 | ~15 min / configuration |
| 70B-class fine-tune | 2 x A100 | ~90 min |
| Deployment 8B FP16 / INT8 / INT4 | 1 x A100 | 18 / 10 / 6 GB |
| Hospital-scale clearance | 8B INT4, 1 GPU | ~2,000 documents in ~7 h, $0.0006 / 1K tokens |
| Precision | — | bf16 training; INT8 retains 99.2% of FP16 quality |

---

## Accompanying documents — cargo manifest (datasets)

| dataset | language | task | access |
|---|---|---|---|
| BC5CDR | en | NER | public (BioCreative V CDR) |
| n2c2 2022 | en | NER | restricted, data use agreement |
| ChemProt | en | RE | public (BioCreative VI) |
| MedQA / MedBench | zh | QA | public (OpenCompass) |
| CliMedBench | zh | QA | public (cross-lingual validation only) |
| CCKS-2017 / 2019 | zh | NER | public (CCKS shared task) |
| MIMIC-IV | en | multi | PhysioNet credentialed |
| Policy corpus (77 docs) | zh | scoring | NHC gazette; PMC-Index on written request |

Run `bash scripts/prepare_data.sh data` for the expected on-disk layout. Synthetic cohorts need no
download.

---

## Dangerous goods and compliance — ethics

The study is retrospective and uses de-identified, publicly released or governing-body-supplied
records; no additional ethical approval was required. MIMIC-IV and n2c2 2022 require credentialed
access and a signed data use agreement before any clinical text is loaded. The trustworthiness
desk flags low-confidence outputs for human oversight (threshold tau = 0.6) and does not act as an
automated clinical decision maker.

---

## Tally checks — tests and gates

    make lint    # ruff + black + isort
    make type    # mypy --strict trustport
    make test    # pytest

The suite spans shape, invariant, overfit-single-batch, metric-versus-reference, gradient-flow,
numerical-regression, and an end-to-end smoke check, plus a house-style guard.
