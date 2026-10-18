import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"


class DecisionModel(nn.Module):
    """
    A small decision model that scores each option independently
    given the question and state, then normalizes over options.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        super().__init__()
        self.model_name = model_name
        self.enc = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Takes input_ids and attention_mask for each (example, option) pair.
        Returns a scalar logit for each pair.
        """
        outputs = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the representation of the first token ([CLS])
        cls_rep = outputs.last_hidden_state[:, 0]
        logits = self.head(cls_rep).squeeze(-1)
        return logits


def create_collate_fn(tokenizer, max_len: int = 256):
    """
    Creates a collate function bound to the given tokenizer.
    Packs all options for each example into a single batch and returns
    input_ids, attention_mask, opt_mask, and labels (if present).
    """

    def collate_fn(batch):
        K = max(len(r["options"]) for r in batch)
        texts = []
        mask = []

        for r in batch:
            opts = r["options"]
            sep_context = f"{r['question']} [SEP] {r['state']}"
            for k in range(K):
                if k < len(opts):
                    texts.append((sep_context, str(opts[k])))
                    mask.append(1)
                else:
                    # Padding when an example has fewer options
                    texts.append(("", ""))
                    mask.append(0)

        enc = tokenizer(
            [a for a, _ in texts],
            [b for _, b in texts],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )

        batch_dict = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "opt_mask": torch.tensor(mask).view(len(batch), K).bool(),
        }

        if len(batch) > 0 and "label" in batch[0]:
            batch_dict["labels"] = torch.tensor([r["label"] for r in batch], dtype=torch.long)

        return batch_dict

    return collate_fn


def logits_for(model: nn.Module, batch: dict, device: torch.device | str) -> torch.Tensor:
    """
    Computes logits for all options in the batch and masks out padded options with -1e9.
    Returns tensor of shape (batch_size, max_options).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    opt_mask = batch["opt_mask"].to(device)

    out = model(input_ids, attention_mask)
    out = out.view(opt_mask.shape)  # Shape: (B, K)
    return out.masked_fill(~opt_mask, -1e9)
