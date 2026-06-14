from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from trustport.appraisal.rubrics import dimension_weights
from trustport.appraisal.tariff import DimensionScorer, pmc_index
from trustport.berths.consignments import BatchTensors
from trustport.consolidation.containerize import CrossModalFusion, pathway_clusters
from trustport.customs.backbones import DeterministicBackbone
from trustport.customs.declarations import DeclarationBlock
from trustport.manifest import ExperimentConfig
from trustport.quay.types import TaskHeads


class HarborOutputs(TypedDict):
    heads: TaskHeads
    dim_scores: torch.Tensor
    pmc: torch.Tensor
    fused: torch.Tensor
    fusion_weights: torch.Tensor
    pathways: torch.Tensor


class Harbor(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.backbone.hidden_dim
        self.backbone = DeterministicBackbone(config.backbone.vocab_size, hidden)
        self.declarations = DeclarationBlock(
            hidden_dim=hidden,
            ner_tags=config.tasks.ner_tags,
            relations=config.tasks.relations,
            classes=config.tasks.classes,
            answers=config.tasks.answers,
            lora_rank=config.backbone.lora_rank,
            lora_alpha=config.backbone.lora_alpha,
            lora_dropout=config.backbone.lora_dropout,
            domain_adapt=config.toggles.domain_adapt,
        )
        self.scorer = DimensionScorer(hidden, config.fusion.kbi_dim)
        self.to_policy = nn.Linear(hidden, config.fusion.policy_dim)
        self.fusion = CrossModalFusion(
            clinical_dim=hidden,
            policy_dim=config.fusion.policy_dim,
            kbi_dim=config.fusion.kbi_dim,
            fused_dim=config.fusion.fused_dim,
            pathways=config.fusion.pathways,
        )
        self.pmc_weights = dimension_weights(config.pmc)

    def forward(self, batch: BatchTensors) -> HarborOutputs:
        token_states, pooled = self.backbone.encode(batch.token_ids)
        heads = self.declarations(token_states, pooled)
        dim_scores = self.scorer(pooled, batch.kbi)
        pmc = pmc_index(dim_scores, self.pmc_weights)
        if self.config.toggles.fusion:
            policy_feat = self.to_policy(pooled)
            fused, weights = self.fusion(pooled, policy_feat, batch.kbi)
        else:
            fused = pooled
            weights = pooled.new_full((pooled.shape[0], 3), 1.0 / 3.0)
        pathway_logits = self.fusion.pathway_logits(fused)
        return HarborOutputs(
            heads=heads,
            dim_scores=dim_scores,
            pmc=pmc,
            fused=fused,
            fusion_weights=weights,
            pathways=pathway_clusters(pathway_logits),
        )


def build_harbor(config: ExperimentConfig) -> Harbor:
    return Harbor(config)
