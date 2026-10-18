from pathlib import Path
from typing import List, Tuple, Union, Optional
import torch
from transformers import AutoTokenizer

from .model import DecisionModel, create_collate_fn, logits_for, DEFAULT_MODEL_NAME


class DecisionEngine:
    """
    Inference engine for calibrated decision making.
    Picks one option from the provided list and returns the choice with its calibrated probability.
    """

    def __init__(
        self,
        model: Optional[DecisionModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        temperature: float = 1.0,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.temperature = max(1e-4, float(temperature))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = DecisionModel(model_name).to(self.device)
        else:
            self.model = model.to(self.device)

        self.model.eval()
        self._collate_fn = create_collate_fn(self.tokenizer)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "DecisionEngine":
        """
        Loads model weights, configuration, and calibrated temperature from a saved checkpoint file.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = ckpt.get("model_name", DEFAULT_MODEL_NAME)
        temperature = ckpt.get("temperature", 1.0)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DecisionModel(model_name)
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            model_name=model_name,
            device=device,
        )

    def save_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Saves model weights, model name, and temperature to a checkpoint file.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )

    @torch.no_grad()
    def decide(
        self,
        state: str,
        question: str,
        options: List[str],
    ) -> Tuple[str, float]:
        """
        Given state, question, and candidate options:
        Scores each option, applies temperature scaling, and returns the top choice and confidence.
        """
        if not options:
            raise ValueError("Options list cannot be empty.")

        batch = self._collate_fn([{"state": state, "question": question, "options": options}])
        raw_logits = logits_for(self.model, batch, self.device)  # shape (1, K)
        scaled_logits = raw_logits / self.temperature
        probs = scaled_logits.softmax(dim=-1)[0]

        best_idx = int(probs.argmax().item())
        confidence = float(probs[best_idx].item())
        return options[best_idx], confidence
