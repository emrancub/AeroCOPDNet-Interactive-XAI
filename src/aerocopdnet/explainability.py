from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np
import torch
from .inference import predict_probability


@dataclass
class SaliencyConfig:
    smoothgrad_samples: int = 24
    noise_std: float = 0.08
    deletion_fractions: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
    random_trials: int = 20


def _target_probability(model, x: torch.Tensor, predicted_copd: bool) -> torch.Tensor:
    p = predict_probability(model, x)[0]
    return p if predicted_copd else (1.0 - p)


def smoothgrad_saliency(model, x: torch.Tensor, cfg: SaliencyConfig, predicted_copd: bool) -> np.ndarray:
    sal = torch.zeros_like(x)
    scale = x.detach().std().clamp_min(1e-6)
    for _ in range(int(cfg.smoothgrad_samples)):
        noisy = (x.detach() + torch.randn_like(x) * float(cfg.noise_std) * scale).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        target = _target_probability(model, noisy, predicted_copd)
        target.backward()
        sal += noisy.grad.detach().abs() * noisy.detach().abs()
    sal /= float(cfg.smoothgrad_samples)
    s = sal[0, 0].detach().cpu().numpy()
    s -= np.nanmin(s)
    denom = np.nanmax(s)
    if denom > 1e-12:
        s /= denom
    return np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def deletion_curve(model, x: torch.Tensor, saliency: np.ndarray, cfg: SaliencyConfig, predicted_copd: bool) -> Dict[str, object]:
    with torch.no_grad():
        base = float(_target_probability(model, x, predicted_copd).cpu())
    flat_sal = saliency.reshape(-1)
    order = np.argsort(flat_sal)[::-1]
    n = flat_sal.size
    baseline_value = float(x.detach().median().cpu())
    rng = np.random.default_rng(2026)
    fractions = [float(f) for f in cfg.deletion_fractions]
    salient_prob, random_prob_mean, random_prob_std = [], [], []

    def prob_after(indices: np.ndarray) -> float:
        if indices.size == 0:
            return base
        xm = x.detach().clone()
        flat = xm[0, 0].reshape(-1)
        flat[torch.as_tensor(indices, device=flat.device, dtype=torch.long)] = baseline_value
        with torch.no_grad():
            return float(_target_probability(model, xm, predicted_copd).cpu())

    for frac in fractions:
        k = int(round(n * frac))
        salient_prob.append(prob_after(order[:k]))
        vals = []
        for _ in range(int(cfg.random_trials)):
            idx = rng.choice(n, size=k, replace=False) if k else np.array([], dtype=int)
            vals.append(prob_after(idx))
        random_prob_mean.append(float(np.mean(vals)))
        random_prob_std.append(float(np.std(vals)))

    salient_drop = [base - p for p in salient_prob]
    random_drop = [base - p for p in random_prob_mean]
    # Trapezoidal area of probability drop vs deletion fraction.
    sal_auc = float(np.trapz(salient_drop, fractions))
    rnd_auc = float(np.trapz(random_drop, fractions))
    return {
        "base_target_probability": base,
        "fractions": fractions,
        "salient_probability": salient_prob,
        "random_probability_mean": random_prob_mean,
        "random_probability_std": random_prob_std,
        "salient_drop": salient_drop,
        "random_drop": random_drop,
        "salient_drop_auc": sal_auc,
        "random_drop_auc": rnd_auc,
        "faithfulness_delta_auc": sal_auc - rnd_auc,
    }
