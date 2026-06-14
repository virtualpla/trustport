from __future__ import annotations

import torch
from torch import nn

from trustport.customs.ledger import LoraLinear
from trustport.quay.types import TaskHeads


class DeclarationBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ner_tags: int,
        relations: int,
        classes: int,
        answers: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        domain_adapt: bool,
    ) -> None:
        super().__init__()
        self.adapt_token = LoraLinear(
            hidden_dim, hidden_dim, lora_rank, lora_alpha, lora_dropout, domain_adapt
        )
        self.adapt_pool = LoraLinear(
            hidden_dim, hidden_dim, lora_rank, lora_alpha, lora_dropout, domain_adapt
        )
        self.ner = nn.Linear(hidden_dim, ner_tags)
        self.relation = nn.Linear(hidden_dim, relations)
        self.category = nn.Linear(hidden_dim, classes)
        self.answer = nn.Linear(hidden_dim, answers)

    def forward(self, token_states: torch.Tensor, pooled: torch.Tensor) -> TaskHeads:
        token_repr: torch.Tensor = torch.tanh(self.adapt_token(token_states))
        pool_repr: torch.Tensor = torch.tanh(self.adapt_pool(pooled))
        return TaskHeads(
            ner=self.ner(token_repr),
            re=self.relation(pool_repr),
            classification=self.category(pool_repr),
            qa=self.answer(pool_repr),
        )
