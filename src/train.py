import argparse
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .model import DecisionModel, create_collate_fn, logits_for, DEFAULT_MODEL_NAME, DEFAULT_MAX_LEN
from .dataset import DecisionDataset
from .calibrate import (
    collect_logits_and_labels,
    calibrate_temperature,
    calibrate_platt,
    apply_calibration,
    compute_ece,
)


def compute_auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    if not scores_pos or not scores_neg:
        return 0.0
    return sum((p > n) + 0.5 * (p == n) for p in scores_pos for n in scores_neg) / (
        len(scores_pos) * len(scores_neg)
    )


def train_model(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 3,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    batch_size: int = 16,
    max_len: int = DEFAULT_MAX_LEN,
    pos_weight: float = 3.0,
    split_type: Optional[str] = None,
    checkpoint_path: Optional[str] = "checkpoints/model.pt",
    device: Optional[str] = None,
):
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    print(f"Using device: {device_obj}")
    if device_obj.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading tokenizer & encoder: {model_name} (max_len={max_len})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collate = create_collate_fn(tokenizer, max_len=max_len)

    train_ds = DecisionDataset(train_path)
    val_ds = DecisionDataset(val_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    if test_path:
        test_ds = DecisionDataset(test_path)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
        print(f"Test samples: {len(test_ds)}")
    else:
        test_loader = None

    model = DecisionModel(model_name).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Vulnerabilities (label 0) are a minority (~19%), so pos_weight weights class 0
    if pos_weight > 1.0:
        class_weights = torch.tensor([pos_weight, 1.0], device=device_obj)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted CrossEntropyLoss: vulnerable={pos_weight:.1f}, safe=1.0")
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state_dict = None

    print("\n--- Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch["labels"].to(device_obj)
            logits = logits_for(model, batch, device_obj)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                preds = logits_for(model, batch, device_obj).argmax(dim=-1).cpu()
                correct += (preds == batch["labels"]).sum().item()
                total += len(preds)

        val_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model weights for calibration
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("\n--- Calibration ---")
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device_obj)
    temperature = calibrate_temperature(val_logits, val_labels)
    print(f"Fitted Temperature (T): {temperature:.4f}")

    platt_scale, platt_bias = None, None
    if val_logits.shape[-1] == 2:
        platt_scale, platt_bias = calibrate_platt(val_logits, val_labels)
        print(f"Fitted Platt Scaling: scale={platt_scale:.4f}, bias={platt_bias:.4f}")

    # Evaluate on test set if available
    if test_loader is not None:
        print("\n--- Test Set Evaluation ---")
        test_logits, test_labels = collect_logits_and_labels(model, test_loader, device_obj)
        raw_probs = test_logits.softmax(dim=-1)
        temp_probs = (test_logits / temperature).softmax(dim=-1)
        platt_probs = (
            apply_calibration(test_logits, platt_scale=platt_scale, platt_bias=platt_bias).softmax(dim=-1)
            if platt_scale is not None
            else temp_probs
        )

        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds == test_labels).float().mean().item()

        # Detailed metrics: missed vulnerabilities (FN) and false alarms (FP)
        is_vuln_true = (test_labels == 0)
        is_safe_true = (test_labels == 1)
        num_vuln = int(is_vuln_true.sum().item())
        num_safe = int(is_safe_true.sum().item())

        p_vuln_raw = raw_probs[:, 0]
        p_vuln_temp = temp_probs[:, 0]
        p_vuln_platt = platt_probs[:, 0]

        fn_raw = int(((p_vuln_raw < 0.5) & is_vuln_true).sum().item())
        fp_raw = int(((p_vuln_raw >= 0.5) & is_safe_true).sum().item())

        fn_temp = int(((p_vuln_temp < 0.5) & is_vuln_true).sum().item())
        fp_temp = int(((p_vuln_temp >= 0.5) & is_safe_true).sum().item())

        fn_platt = int(((p_vuln_platt < 0.5) & is_vuln_true).sum().item())
        fp_platt = int(((p_vuln_platt >= 0.5) & is_safe_true).sum().item())

        # Asymmetric risk threshold 0.3 (lower threshold catches more attacks)
        fn_risk = int(((p_vuln_platt < 0.3) & is_vuln_true).sum().item())
        fp_risk = int(((p_vuln_platt >= 0.3) & is_safe_true).sum().item())

        ece_raw = compute_ece(raw_probs, test_labels)
        ece_temp = compute_ece(temp_probs, test_labels)
        ece_platt = compute_ece(platt_probs, test_labels)

        auc = compute_auc(
            p_vuln_platt[is_vuln_true].tolist(),
            p_vuln_platt[is_safe_true].tolist(),
        )

        print(f"Test Set: {len(test_labels)} answers ({num_vuln} vulnerable, {num_safe} safe)")
        print(f"Accuracy: {test_acc:.1%}")
        print(f"Raw at 0.5:             FN (missed): {fn_raw}/{num_vuln} | FP (false alarms): {fp_raw}/{num_safe} | ECE: {ece_raw:.4f}")
        print(f"Temperature at 0.5:     FN (missed): {fn_temp}/{num_vuln} | FP (false alarms): {fp_temp}/{num_safe} | ECE: {ece_temp:.4f}")
        print(f"Platt Scaled at 0.5:    FN (missed): {fn_platt}/{num_vuln} | FP (false alarms): {fp_platt}/{num_safe} | ECE: {ece_platt:.4f}")
        print(f"Risk Policy (P>=0.30):  FN (missed): {fn_risk}/{num_vuln} | FP (false alarms): {fp_risk}/{num_safe}")
        print(f"AUC: {auc:.3f}")

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": model_name,
                "temperature": temperature,
                "platt_scale": platt_scale,
                "platt_bias": platt_bias,
                "split_type": split_type,
                "max_len": max_len,
                "model_state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        print(f"\nModel, Platt calibration and metadata saved to: {ckpt_path}")

    return model, temperature


def main():
    parser = argparse.ArgumentParser(description="Train tiny-decision-model on chatbot audit task")
    parser.add_argument(
        "--split",
        choices=["default", "group", "benchmark"],
        default="default",
        help="Split preset to use: 'group' (stratified unseen prompts), 'benchmark' (Half A / Half B), or 'default'",
    )
    parser.add_argument("--train", type=str, default=None, help="Path to train.jsonl (overrides --split)")
    parser.add_argument("--val", type=str, default=None, help="Path to val.jsonl (overrides --split)")
    parser.add_argument("--test", type=str, default=None, help="Path to test.jsonl (overrides --split)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Transformer encoder model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="Max sequence length (default 512)")
    parser.add_argument("--pos-weight", type=float, default=3.0, help="Loss weight for minority vulnerable class")
    parser.add_argument("--output", type=str, default=None, help="Checkpoint save path")
    args = parser.parse_args()

    if args.split == "benchmark":
        train_path = args.train or "data/benchmark/train.jsonl"
        val_path = args.val or "data/benchmark/val.jsonl"
        test_path = args.test or "data/benchmark/test.jsonl"
        out_path = args.output or "checkpoints/model_benchmark.pt"
        split_type = "benchmark"
    elif args.split == "group":
        train_path = args.train or "data/group/train.jsonl"
        val_path = args.val or "data/group/val.jsonl"
        test_path = args.test or "data/group/test.jsonl"
        out_path = args.output or "checkpoints/model_group.pt"
        split_type = "group"
    else:
        train_path = args.train or "data/train.jsonl"
        val_path = args.val or "data/val.jsonl"
        test_path = args.test or "data/test.jsonl"
        out_path = args.output or "checkpoints/model.pt"
        split_type = "default"

    train_model(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path if Path(test_path).exists() else None,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_len=args.max_len,
        pos_weight=args.pos_weight,
        split_type=split_type,
        checkpoint_path=out_path,
    )


if __name__ == "__main__":
    main()
