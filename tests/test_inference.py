import torch
import torch.nn as nn
from pathlib import Path
from src.inference import DecisionEngine
from tests.test_model import MockTokenizer


class DummyDecisionModel(nn.Module):
    def __init__(self, model_name="dummy"):
        super().__init__()
        self.model_name = model_name
        self.dummy_param = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, input_ids, attention_mask):
        # Return predetermined scores based on pair index
        return torch.tensor([1.0, 3.0][:input_ids.shape[0]], dtype=torch.float)


def test_decision_engine_decide():
    tokenizer = MockTokenizer()
    model = DummyDecisionModel()
    engine = DecisionEngine(
        model=model,
        tokenizer=tokenizer,
        temperature=1.0,
        model_name="dummy",
        device="cpu",
    )

    choice, conf = engine.decide(
        state="Test state",
        question="Is this urgent?",
        options=["no", "yes"],
    )

    # Logits: [1.0, 3.0] -> index 1 has higher logit -> "yes"
    assert choice == "yes"
    assert conf > 0.5
    assert conf <= 1.0


def test_checkpoint_save_and_load(tmp_path: Path):
    ckpt_path = tmp_path / "model.pt"
    tokenizer = MockTokenizer()
    model = DummyDecisionModel()
    engine = DecisionEngine(
        model=model,
        tokenizer=tokenizer,
        temperature=1.45,
        model_name="dummy",
        device="cpu",
    )

    engine.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["temperature"] == 1.45
    assert ckpt["model_name"] == "dummy"
    assert "dummy_param" in ckpt["model_state_dict"]
