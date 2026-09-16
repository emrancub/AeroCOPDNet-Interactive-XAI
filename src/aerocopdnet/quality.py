from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class QualityConfig:
    silence_db: float = -45.0
    clipping_threshold: float = 0.995
    max_clipping_fraction: float = 0.01
    max_silence_fraction: float = 0.80


def audio_quality(y: np.ndarray, sr: int, cfg: QualityConfig) -> dict:
    eps = 1e-12
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y)) + eps))
    frame = max(64, int(0.05 * sr))
    hop = max(32, frame // 2)
    if y.size < frame:
        frame_rms = np.array([rms], dtype=np.float32)
    else:
        starts = range(0, y.size - frame + 1, hop)
        frame_rms = np.array([np.sqrt(np.mean(np.square(y[s:s+frame])) + eps) for s in starts], dtype=np.float32)
    db = 20.0 * np.log10(np.maximum(frame_rms, eps))
    silence_fraction = float(np.mean(db < cfg.silence_db))
    clipping_fraction = float(np.mean(np.abs(y) >= cfg.clipping_threshold))
    duration = float(y.size / sr)
    flags = []
    if clipping_fraction > cfg.max_clipping_fraction:
        flags.append("high clipping fraction")
    if silence_fraction > cfg.max_silence_fraction:
        flags.append("mostly silent / very low energy")
    if duration < 1.0:
        flags.append("very short recording")
    return {
        "duration_s": duration,
        "peak_amplitude": peak,
        "rms": rms,
        "silence_fraction": silence_fraction,
        "clipping_fraction": clipping_fraction,
        "status": "review" if flags else "pass",
        "flags": flags,
    }
