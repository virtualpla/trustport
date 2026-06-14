from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from trustport.harbor import Harbor


class _NerExport(nn.Module):
    def __init__(self, harbor: Harbor) -> None:
        super().__init__()
        self.backbone = harbor.backbone
        self.declarations = harbor.declarations

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_states, pooled = self.backbone.encode(token_ids)
        heads = self.declarations(token_states, pooled)
        logits: torch.Tensor = heads["ner"]
        return logits


def export_onnx(harbor: Harbor, sample_tokens: torch.Tensor, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _NerExport(harbor).eval()
    torch.onnx.export(
        wrapper,
        (sample_tokens,),
        str(target),
        input_names=["token_ids"],
        output_names=["ner_logits"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"token_ids": {0: "batch", 1: "seq"}},
    )
