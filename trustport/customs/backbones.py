from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Backbone(Protocol):
    hidden_dim: int

    def encode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class DeterministicBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def encode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded: torch.Tensor = self.embed(token_ids)
        token_states: torch.Tensor = self.norm(embedded + self.act(self.mix(embedded)))
        pooled: torch.Tensor = self.act(self.pool_proj(token_states.mean(dim=1)))
        return token_states, pooled

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(token_ids)


class GuardedHFBackbone(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self._encoder = self._build(model_name)

    @staticmethod
    def _build(model_name: str) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "real backbone requested but transformers is not installed; "
                "install the 'hf' extra or use the deterministic backbone"
            ) from exc
        model: nn.Module = AutoModel.from_pretrained(model_name)
        return model

    def encode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._encoder(input_ids=token_ids)
        token_states: torch.Tensor = outputs.last_hidden_state
        pooled: torch.Tensor = token_states.mean(dim=1)
        return token_states, pooled

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(token_ids)


def load_hf_backbone(model_name: str, hidden_dim: int) -> GuardedHFBackbone:
    return GuardedHFBackbone(model_name, hidden_dim)
