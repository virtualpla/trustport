from __future__ import annotations

from dataclasses import dataclass, field, replace

from trustport.quay.types import LanguageCode, ModalityName, Task


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "llama31_8b"
    vocab_size: int = 4096
    hidden_dim: int = 128
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_len: int = 2048


@dataclass(frozen=True)
class TaskConfig:
    ner_tags: int = 9
    relations: int = 13
    classes: int = 4
    answers: int = 4


@dataclass(frozen=True)
class PmcConfig:
    alpha_policy: float = 0.35
    alpha_measure: float = 0.30
    alpha_content: float = 0.35
    top_k: int = 5


@dataclass(frozen=True)
class FusionConfig:
    clinical_dim: int = 128
    policy_dim: int = 128
    kbi_dim: int = 76
    fused_dim: int = 128
    pathways: int = 4


@dataclass(frozen=True)
class TrustConfig:
    tau_trust: float = 0.6
    consistency_k: int = 3
    consistency_tau: float = 0.7


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 10
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float | None = None
    ema_decay: float = 0.999
    precision: str = "bf16"


@dataclass(frozen=True)
class DataConfig:
    datasets: tuple[str, ...] = ("bc5cdr", "n2c2", "chemprot", "medqa", "ccks17", "ccks19")
    documents: int = 256
    seq_len: int = 64
    signal_strength: float = 3.0


@dataclass(frozen=True)
class Toggles:
    domain_adapt: bool = True
    struct_out: bool = True
    fusion: bool = True
    trust: bool = True
    rag: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "main"
    seed: int = 42
    format_gamma: float = 0.3
    init_task_weights: tuple[float, float, float, float] = (1.0, 1.5, 0.8, 0.8)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    pmc: PmcConfig = field(default_factory=PmcConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    toggles: Toggles = field(default_factory=Toggles)

    def with_seed(self, seed: int) -> ExperimentConfig:
        return replace(self, seed=seed)


@dataclass(frozen=True)
class Consignment:
    doc_id: str
    language: LanguageCode
    modality: ModalityName
    token_ids: tuple[int, ...]
    ner_labels: tuple[int, ...]
    relation: int
    category: int
    answer: int
    pmc_targets: tuple[float, float, float]
    kbi: tuple[float, ...]


def stage_weight_for(task: Task) -> float:
    table: dict[Task, float] = {"ner": 0.35, "re": 0.40, "classification": 0.25, "qa": 0.0}
    return table[task]
