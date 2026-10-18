import json
from pathlib import Path
from typing import List, Dict, Any, Union
from torch.utils.data import Dataset


class DecisionDataset(Dataset):
    """
    Dataset that loads examples from a JSONL file.
    Each row should have:
    - state: str
    - question: str
    - options: List[str]
    - label: int (optional, index of the target option in options)
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._validate_row(data, line_no)
                    self.rows.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} in {self.path}: {e}")

    @staticmethod
    def _validate_row(row: Dict[str, Any], line_no: int):
        if "state" not in row or "question" not in row or "options" not in row:
            raise ValueError(
                f"Line {line_no} must contain 'state', 'question', and 'options'."
            )
        if not isinstance(row["options"], list) or len(row["options"]) == 0:
            raise ValueError(
                f"Line {line_no}: 'options' must be a non-empty list."
            )
        if "label" in row:
            label = row["label"]
            if not isinstance(label, int) or label < 0 or label >= len(row["options"]):
                raise ValueError(
                    f"Line {line_no}: 'label' {label} is out of bounds for options length {len(row['options'])}."
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]
