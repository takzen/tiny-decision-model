from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
import torch
from transformers import AutoTokenizer

from .model import DecisionModel, create_collate_fn, logits_for, DEFAULT_MODEL_NAME, DEFAULT_MAX_LEN
from .calibrate import apply_calibration


class DecisionEngine:
    """
    Inference engine for calibrated decision making.
    Picks one option from the provided list and returns the choice with its calibrated probability.
    Supports temperature scaling and Platt scaling (scale + bias).
    """

    def __init__(
        self,
        model: Optional[DecisionModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        temperature: float = 1.0,
        platt_scale: Optional[float] = None,
        platt_bias: Optional[float] = None,
        split_type: Optional[str] = None,
        max_len: int = DEFAULT_MAX_LEN,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.temperature = max(1e-4, float(temperature))
        self.platt_scale = platt_scale
        self.platt_bias = platt_bias
        self.split_type = split_type
        self.max_len = max_len

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = DecisionModel(model_name).to(self.device)
        else:
            self.model = model.to(self.device)

        self.model.eval()
        self._collate_fn = create_collate_fn(self.tokenizer, max_len=self.max_len)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "DecisionEngine":
        """
        Loads model weights, configuration, calibration parameters, and split metadata
        from a saved checkpoint file.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = ckpt.get("model_name", DEFAULT_MODEL_NAME)
        temperature = ckpt.get("temperature", 1.0)
        platt_scale = ckpt.get("platt_scale", None)
        platt_bias = ckpt.get("platt_bias", None)
        split_type = ckpt.get("split_type", None)
        max_len = ckpt.get("max_len", DEFAULT_MAX_LEN)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DecisionModel(model_name)
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            platt_scale=platt_scale,
            platt_bias=platt_bias,
            split_type=split_type,
            max_len=max_len,
            model_name=model_name,
            device=device,
        )

    def save_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Saves model weights, model name, calibration, and split metadata to a checkpoint file.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "platt_scale": self.platt_scale,
                "platt_bias": self.platt_bias,
                "split_type": self.split_type,
                "max_len": self.max_len,
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )

    @torch.no_grad()
    def get_calibrated_probabilities(
        self,
        state: str,
        question: str,
        options: List[str],
    ) -> torch.Tensor:
        """
        Returns calibrated probabilities tensor for the given options.
        """
        if not options:
            raise ValueError("Options list cannot be empty.")

        batch = self._collate_fn([{"state": state, "question": question, "options": options}])
        raw_logits = logits_for(self.model, batch, self.device)  # shape (1, K)
        cal_logits = apply_calibration(
            raw_logits,
            temperature=self.temperature,
            platt_scale=self.platt_scale,
            platt_bias=self.platt_bias,
        )
        return cal_logits.softmax(dim=-1)[0]

    @torch.no_grad()
    def decide(
        self,
        state: str,
        question: str,
        options: List[str],
    ) -> Tuple[str, float]:
        """
        Given state, question, and candidate options:
        Scores each option, applies calibration, and returns the top choice and confidence.
        """
        probs = self.get_calibrated_probabilities(state, question, options)
        best_idx = int(probs.argmax().item())
        confidence = float(probs[best_idx].item())
        return options[best_idx], confidence

    @torch.no_grad()
    def decide_detailed(
        self,
        state: str,
        question: str,
        options: List[str],
    ) -> Dict[str, Any]:
        """
        Returns detailed decision output including choice, confidence, per-option probabilities,
        and P(vulnerable) (option 0).
        """
        probs = self.get_calibrated_probabilities(state, question, options)
        best_idx = int(probs.argmax().item())
        prob_list = [float(p.item()) for p in probs[: len(options)]]

        return {
            "choice": options[best_idx],
            "confidence": prob_list[best_idx],
            "probabilities": {opt: prob_list[i] for i, opt in enumerate(options)},
            "p_vulnerable": prob_list[0] if len(prob_list) > 0 else 0.0,
        }
