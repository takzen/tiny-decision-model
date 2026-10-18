"""
CLI entrypoint to run inference using a trained decision model checkpoint or pretrained encoder.

Usage:
    uv run python decide.py --state "Faktura nie dotarła, termin mija jutro." --question "Czy ta wiadomość jest pilna?" --options "tak" "nie"
    uv run python decide.py --checkpoint checkpoints/model.pt --state "Invoice never arrived, deadline is tomorrow." --question "Is this message urgent?" --options "yes" "no"
"""

import argparse
from pathlib import Path
from src.inference import DecisionEngine


def main():
    parser = argparse.ArgumentParser(description="Run calibrated decision inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="Path to trained model checkpoint (.pt). If not found, uses default unfinetuned model.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="Faktura nie dotarła, termin mija jutro.",
        help="Context / state input",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Czy ta wiadomość jest pilna?",
        help="Decision question",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=["tak", "nie"],
        help="List of candidate options",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for automated decision",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading calibrated model from checkpoint: {ckpt_path}")
        engine = DecisionEngine.load_from_checkpoint(ckpt_path)
    else:
        print(f"Checkpoint {ckpt_path} not found. Running with base pretrained encoder...")
        engine = DecisionEngine()

    choice, confidence = engine.decide(
        state=args.state,
        question=args.question,
        options=args.options,
    )

    print("\n--- Decision Result ---")
    print(f"Question:    {args.question}")
    print(f"State:       {args.state}")
    print(f"Options:     {args.options}")
    print(f"Choice:      {choice}")
    print(f"Confidence:  {confidence:.1%}")

    if confidence < args.threshold:
        print(f"Action:      ESCALATE (confidence {confidence:.1%} < threshold {args.threshold:.1%})")
    else:
        print(f"Action:      ACT (confidence {confidence:.1%} >= threshold {args.threshold:.1%})")


if __name__ == "__main__":
    main()
