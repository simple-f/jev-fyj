from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DecisionEncoder(nn.Module):
    """Encoder-only dynamic candidate scorer.

    Unlike a fixed `Linear(hidden, num_labels)` classifier, this head scores
    each candidate marker with the same `Linear(hidden, 1)` scorer. That keeps
    the model compatible with dynamic labels supplied at inference time.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        encoder: nn.Module | None = None,
        hidden_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder is None:
            if model_name is None:
                raise ValueError("model_name is required when encoder is not provided")
            encoder = AutoModel.from_pretrained(model_name)
        if hidden_size is None:
            hidden_size = int(getattr(encoder.config, "hidden_size"))

        self.model_name = model_name
        self.encoder = encoder
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if marker_positions.ndim != 2:
            raise ValueError("marker_positions must have shape [batch_size, num_options]")
        if marker_positions.max().item() >= hidden.size(1):
            raise ValueError("marker_positions contains an index outside the sequence length")

        gather_index = marker_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        marker_hidden = torch.gather(hidden, dim=1, index=gather_index)
        return self.scorer(marker_hidden).squeeze(-1)

    def save_decision_model(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if hasattr(self.encoder, "save_pretrained"):
            self.encoder.save_pretrained(output / "encoder")
        torch.save(self.scorer.state_dict(), output / "scorer.pt")
        metadata = {"model_name": self.model_name}
        (output / "decision_model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_decision_dir(cls, model_dir: str | Path) -> "DecisionEncoder":
        model_dir = Path(model_dir)
        metadata_path = model_dir / "decision_model.json"
        model_name = None
        if metadata_path.exists():
            model_name = json.loads(metadata_path.read_text(encoding="utf-8")).get("model_name")
        model = cls(model_name=model_name, encoder=AutoModel.from_pretrained(model_dir / "encoder"))
        model.scorer.load_state_dict(torch.load(model_dir / "scorer.pt", map_location="cpu"))
        return model


def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    max_entropy = math.log(logits.size(-1))
    if max_entropy == 0:
        return torch.ones_like(entropy)
    return 1 - entropy / max_entropy
