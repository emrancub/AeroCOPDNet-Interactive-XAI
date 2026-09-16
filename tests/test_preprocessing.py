import numpy as np
from aerocopdnet.audio import AudioConfig, logmel
from aerocopdnet.quality import QualityConfig, audio_quality


def test_logmel_shape():
    cfg = AudioConfig()
    t = np.arange(cfg.sample_rate * 2) / cfg.sample_rate
    y = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    x = logmel(y, cfg.sample_rate, cfg)
    assert x.shape[0] == 64
    assert x.shape[1] > 0


def test_quality_fields():
    y = np.zeros(4000, dtype=np.float32)
    q = audio_quality(y, 4000, QualityConfig())
    assert "silence_fraction" in q
    assert q["duration_s"] == 1.0
