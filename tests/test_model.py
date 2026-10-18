import torch
import torch.nn as nn
from src.model import create_collate_fn, logits_for


class MockTokenizer:
    def __call__(self, texts_a, texts_b, truncation=True, max_length=256, padding=True, return_tensors="pt"):
        n = len(texts_a)
        seq_len = 8
        return {
            "input_ids": torch.zeros((n, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((n, seq_len), dtype=torch.long),
        }


class MockModel(nn.Module):
    def forward(self, input_ids, attention_mask):
        # Return 1 logit per pair
        return torch.arange(input_ids.shape[0], dtype=torch.float)


def test_collate_fn_variable_options():
    tokenizer = MockTokenizer()
    collate = create_collate_fn(tokenizer, max_len=32)

    batch = [
        {"state": "state 1", "question": "q1?", "options": ["opt1", "opt2", "opt3"], "label": 1},
        {"state": "state 2", "question": "q2?", "options": ["yes", "no"], "label": 0},
    ]

    res = collate(batch)
    # Total pairs = 2 examples * 3 max options = 6
    assert res["input_ids"].shape[0] == 6
    assert res["opt_mask"].shape == (2, 3)
    # First example has 3 valid options
    assert res["opt_mask"][0].tolist() == [True, True, True]
    # Second example has 2 valid options, 1 padding
    assert res["opt_mask"][1].tolist() == [True, True, False]
    assert res["labels"].tolist() == [1, 0]


def test_logits_for_masking():
    tokenizer = MockTokenizer()
    collate = create_collate_fn(tokenizer, max_len=32)

    batch = [
        {"state": "state 1", "question": "q1?", "options": ["opt1"], "label": 0},
        {"state": "state 2", "question": "q2?", "options": ["yes", "no"], "label": 1},
    ]
    batch_dict = collate(batch)
    model = MockModel()

    logits = logits_for(model, batch_dict, device="cpu")
    assert logits.shape == (2, 2)
    # First example had only 1 option, so option index 1 must be masked with -1e9
    assert logits[0, 1].item() <= -1e8
    # Valid options should have normal values
    assert logits[0, 0].item() > -1e8
    assert logits[1, 0].item() > -1e8
    assert logits[1, 1].item() > -1e8
