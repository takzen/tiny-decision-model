import torch
from src.calibrate import calibrate_temperature, compute_ece


def test_calibrate_temperature():
    torch.manual_seed(42)
    # Simulate overconfident model with some validation errors
    logits = torch.tensor([
        [5.0, 1.0],
        [1.0, 5.0],
        [4.0, 1.0],
        [1.0, 4.0],
    ])
    # 2 correct, 2 incorrect (overconfident on errors)
    labels = torch.tensor([1, 0, 0, 1])

    T = calibrate_temperature(logits, labels, max_iter=20)
    assert T > 0.0
    # Overconfident network with errors should lead to T > 1.0 to soften logits
    assert T > 1.0



def test_compute_ece():
    # Perfect calibration synthetic test
    probs = torch.tensor([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.1, 0.9],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    ece = compute_ece(probs, labels, bins=5)
    # Confidences are 0.9, accuracies are 1.0 -> difference is ~0.1
    assert 0.0 <= ece <= 0.2
