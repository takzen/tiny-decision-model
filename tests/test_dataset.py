import json
import pytest
from pathlib import Path
from src.dataset import DecisionDataset


def test_dataset_loading(tmp_path: Path):
    data_file = tmp_path / "sample.jsonl"
    rows = [
        {"state": "Message 1", "question": "Q1?", "options": ["opt1", "opt2"], "label": 0},
        {"state": "Message 2", "question": "Q2?", "options": ["yes", "no"], "label": 1},
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ds = DecisionDataset(data_file)
    assert len(ds) == 2
    assert ds[0]["state"] == "Message 1"
    assert ds[1]["label"] == 1


def test_dataset_missing_field(tmp_path: Path):
    data_file = tmp_path / "invalid.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"state": "hello", "question": "world"}) + "\n")

    with pytest.raises(ValueError, match="must contain 'state', 'question', and 'options'"):
        DecisionDataset(data_file)


def test_dataset_invalid_label(tmp_path: Path):
    data_file = tmp_path / "invalid_label.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"state": "s", "question": "q", "options": ["a", "b"], "label": 5}) + "\n")

    with pytest.raises(ValueError, match="out of bounds"):
        DecisionDataset(data_file)
