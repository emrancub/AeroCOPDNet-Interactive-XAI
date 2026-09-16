from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def waveform_figure(y: np.ndarray, sr: int):
    t = np.arange(y.size) / float(sr)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t, y, linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Uploaded waveform")
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def logmel_figure(db: np.ndarray, sr: int, hop_length: int, title: str = "Log-Mel spectrogram"):
    duration = (db.shape[1] - 1) * hop_length / float(sr)
    fig, ax = plt.subplots(figsize=(8, 2.7))
    im = ax.imshow(db, origin="lower", aspect="auto", extent=[0, duration, 0, sr / 2])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="dB")
    fig.tight_layout()
    return fig


def saliency_overlay_figure(db: np.ndarray, saliency: np.ndarray, sr: int, hop_length: int):
    duration = (db.shape[1] - 1) * hop_length / float(sr)
    fig, ax = plt.subplots(figsize=(8, 2.7))
    ax.imshow(db, origin="lower", aspect="auto", extent=[0, duration, 0, sr / 2])
    ax.imshow(saliency, origin="lower", aspect="auto", alpha=0.48, extent=[0, duration, 0, sr / 2])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Saliency overlay: acoustic model sensitivity")
    fig.tight_layout()
    return fig


def deletion_curve_figure(audit: dict):
    fractions = np.asarray(audit["fractions"]) * 100.0
    sal = np.asarray(audit["salient_probability"])
    rnd = np.asarray(audit["random_probability_mean"])
    std = np.asarray(audit["random_probability_std"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(fractions, sal, marker="o", label="Top-saliency deletion")
    ax.plot(fractions, rnd, marker="o", label="Random deletion mean")
    ax.fill_between(fractions, rnd-std, rnd+std, alpha=0.15)
    ax.set_xlabel("Deleted time-frequency bins (%)")
    ax.set_ylabel("Predicted-class probability")
    ax.set_title("Deletion faithfulness audit")
    ax.legend()
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return fig
