"""
CLI entrypoint to train, calibrate and evaluate the tiny-decision-model.
Example:
    uv run python train.py --epochs 3 --batch-size 16
"""

from src.train import main

if __name__ == "__main__":
    main()
