from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass
class AudioConfig:
    sample_rate: int = 4000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    fmin: int = 50
    fmax: int = 2000
    highpass_hz: int = 50
    max_seconds: int = 30


def _highpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return y.astype(np.float32)
    nyq = sr / 2.0
    cutoff = min(max(cutoff_hz / nyq, 1e-6), 0.95)
    sos = butter(2, cutoff, btype="highpass", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def load_audio(file_obj, cfg: AudioConfig) -> Tuple[np.ndarray, int]:
    y, _ = librosa.load(file_obj, sr=cfg.sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("The audio is empty.")
    max_len = int(cfg.sample_rate * cfg.max_seconds)
    if y.size > max_len:
        y = y[:max_len]
    peak = float(np.max(np.abs(y)))
    if peak > 1e-8:
        y = y / peak
    y = _highpass(y, cfg.sample_rate, cfg.highpass_hz)
    return y.astype(np.float32), cfg.sample_rate


def logmel(y: np.ndarray, sr: int, cfg: AudioConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=min(cfg.fmax, sr // 2),
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def normalize_logmel(x: np.ndarray, stats_path: Optional[str] = None) -> tuple[np.ndarray, str]:
    if stats_path and Path(stats_path).exists():
        stats = np.load(stats_path)
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if mean.ndim == 1:
            mean = mean[:, None]
        if std.ndim == 1:
            std = std[:, None]
        return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32), "training-statistics normalization"
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32), "per-recording band normalization (fallback)"
