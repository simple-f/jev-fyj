import torch
import torch.nn as nn

from decision_model.model import DecisionEncoder, confidence_from_logits


class FakeOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.emb = nn.Embedding(128, hidden_size)

    def forward(self, input_ids, attention_mask):
        return FakeOutput(self.emb(input_ids))


def test_decision_encoder_outputs_one_logit_per_candidate_marker():
    model = DecisionEncoder(encoder=FakeEncoder(), hidden_size=8)
    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    marker_positions = torch.tensor([[1, 3, 5], [2, 4, 6]])

    logits = model(input_ids, attention_mask, marker_positions)

    assert logits.shape == (2, 3)


def test_decision_encoder_supports_different_candidate_count():
    model = DecisionEncoder(encoder=FakeEncoder(), hidden_size=8)
    input_ids = torch.randint(0, 100, (1, 12))
    attention_mask = torch.ones_like(input_ids)
    marker_positions = torch.tensor([[1, 3, 5, 7, 9]])

    logits = model(input_ids, attention_mask, marker_positions)

    assert logits.shape == (1, 5)


def test_confidence_is_high_for_peaked_distribution():
    flat = confidence_from_logits(torch.tensor([[1.0, 1.0, 1.0]]))
    peaked = confidence_from_logits(torch.tensor([[0.0, 0.0, 8.0]]))

    assert peaked.item() > flat.item()
