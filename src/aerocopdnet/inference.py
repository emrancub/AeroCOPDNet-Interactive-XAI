from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import torch


def load_torchscript(path: Optional[str], device: torch.device):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    model = torch.jit.load(str(p), map_location=device)
    model.eval()
    return model


def to_probability(output: torch.Tensor) -> torch.Tensor:
    output = output.reshape(-1)
    if torch.all((output >= 0) & (output <= 1)):
        return output
    return torch.sigmoid(output)


def predict_probability(model, x: torch.Tensor) -> torch.Tensor:
    return to_probability(model(x))


def predicted_class(prob: float, threshold: float = 0.5) -> Tuple[str, float]:
    if prob >= threshold:
        return "COPD", prob
    return "Non-COPD", 1.0 - prob
