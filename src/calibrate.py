from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import logits_for


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module, loader: DataLoader, device: torch.device | str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the model over the dataset loader and collects raw logits and labels.
    """
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        logits = logits_for(model, batch, device)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def calibrate_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.1,
) -> float:
    """
    Fits a single temperature T > 0 on validation logits using LBFGS to minimize
    cross-entropy loss (Temperature Scaling).
    Note: Temperature scaling scales logits by 1/T but does not include a bias term,
    so it cannot correct for constant shifts introduced by class-weighted loss (pos_weight).
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / log_t.exp()
        loss = nn.functional.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = log_t.exp().item()
    return float(temperature)


def calibrate_platt(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.1,
) -> Tuple[float, float]:
    """
    Fits Platt scaling (scale a > 0, bias b in R) for binary decisions:
    P(class 0) = sigmoid(a * (z0 - z1) + b).
    This jointly corrects both over/under-confidence (scale a) and shift from class weights (bias b),
    matching the methodology used by kanari-decision-bench.
    """
    diff = logits[:, 0] - logits[:, 1]
    target = (labels == 0).float()

    log_a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_a, b], lr=lr, max_iter=max_iter)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = log_a.exp() * diff + b
        loss = bce(scaled, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = log_a.exp().item()
    bias = b.item()
    return float(scale), float(bias)


def apply_calibration(
    logits: torch.Tensor,
    temperature: float = 1.0,
    platt_scale: Optional[float] = None,
    platt_bias: Optional[float] = None,
) -> torch.Tensor:
    """
    Applies either Platt scaling (if scale and bias are provided) or temperature scaling.
    Returns calibrated logits suitable for standard softmax(dim=-1).
    """
    if platt_scale is not None and platt_bias is not None and logits.shape[-1] == 2:
        diff = logits[..., 0] - logits[..., 1]
        scaled = platt_scale * diff + platt_bias
        return torch.stack([scaled / 2.0, -scaled / 2.0], dim=-1)
    else:
        temp = max(1e-4, float(temperature))
        return logits / temp


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE):
    Groups predictions into confidence bins and computes the weighted average of
    |accuracy - confidence| across bins.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).float()
    edges = torch.linspace(0, 1, bins + 1)

    total_ece = 0.0
    total_samples = len(labels)

    if total_samples == 0:
        return 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        bin_samples = mask.sum().item()
        if bin_samples > 0:
            bin_acc = accuracies[mask].mean().item()
            bin_conf = confidences[mask].mean().item()
            bin_weight = bin_samples / total_samples
            total_ece += bin_weight * abs(bin_acc - bin_conf)

    return float(total_ece)
