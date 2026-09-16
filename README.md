# AeroCOPDNet Interactive XAI Dashboard v2

A research-grade Streamlit project for **reviewable COPD screening from respiratory audio** using the AeroCOPDNet acoustic model.

This repository is designed to accompany the forum manuscript:

> **Interpretable AeroCOPDNet: Lightweight and Explainable COPD Screening from Respiratory Sounds with Saliency-Guided Validation and an Interactive Review Dashboard**

## What this project provides

- WAV upload and standardized 4 kHz preprocessing
- audio quality checks: duration, clipping fraction, silence fraction, RMS and peak amplitude
- waveform and 64-band log-Mel visualization
- plug-in AeroCOPDNet TorchScript inference
- COPD probability, predicted class and predicted-class confidence
- SmoothGrad-style time-frequency saliency
- saliency overlay for human review
- **deletion faithfulness curves** comparing top-saliency deletion with random deletion
- downloadable audit results as JSON/CSV
- optional age, sex, chest location and device fields shown as **context only**
- command-line batch evaluation for prediction + saliency audit

## Scientific scope

The repository does **not** include fabricated clinical data or substitute model weights. Real prediction and saliency require a compatible trained AeroCOPDNet checkpoint. The dashboard is a research prototype for transparent review and is **not an autonomous diagnostic device**.

The acoustic model and benchmark results are described in:

M. E. Hasan, Y.-F. Wu, D.-J. Yu, *AeroCOPDNet: A deep learning framework for COPD detection from lung sounds*, Biomedical Signal Processing and Control 119 (2026) 109939. https://doi.org/10.1016/j.bspc.2026.109939

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Optional environment variables:

```bash
AEROCOPDNET_CHECKPOINT=/absolute/path/to/aerocopdnet.ts
AEROCOPDNET_NORM=/absolute/path/to/normalization.npz
```

`normalization.npz` should contain arrays named `mean` and `std`, normally shaped `(64, 1)` or otherwise broadcastable to the log-Mel tensor.

## Expected TorchScript interface

The model should accept a float tensor shaped:

```text
(B, 1, 64, T)
```

and return a tensor shaped `(B,)` or `(B, 1)`. Outputs may be logits or probabilities; the loader detects `[0,1]` outputs and otherwise applies a sigmoid.

## Repository layout

```text
AeroCOPDNet-Interactive-XAI-v2/
├── app.py
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── MODEL_CARD.md
├── REPRODUCIBILITY.md
├── DISCLAIMER.md
├── config/default.yaml
├── src/aerocopdnet/
│   ├── audio.py
│   ├── inference.py
│   ├── explainability.py
│   ├── plots.py
│   └── quality.py
├── scripts/
│   ├── export_torchscript.py
│   └── evaluate_folder.py
├── checkpoints/README.md
├── assets/
│   ├── dashboard_concept.png
│   └── saliency_method.png
├── tests/test_preprocessing.py
├── run_windows.bat
└── run.sh
```

## Saliency method

For normalized log-Mel input `X` and predicted-class probability `p_c(X)`, SmoothGrad-style saliency is computed as the average absolute input gradient over noisy perturbations:

```text
S(X) = mean_k |∂p_c(X + ε_k) / ∂X| · |X|
```

The saliency matrix is normalized to `[0,1]` for visualization.

## Deletion faithfulness audit

A visually plausible heat map is not automatically faithful. The project therefore compares:

1. progressive deletion of the **highest-saliency** bins; and
2. deletion of the same number of **random** bins.

For each deletion fraction, the predicted-class probability is recomputed. Stronger probability degradation under saliency-guided deletion supports a closer relationship between the explanation and model decision. This is a model-behavior audit, not proof of clinical causality.

## Batch evaluation

Create a CSV with at least a `path` column:

```csv
path,label
/path/to/sample1.wav,1
/path/to/sample2.wav,0
```

Then run:

```bash
python scripts/evaluate_folder.py \
  --csv samples.csv \
  --checkpoint checkpoints/aerocopdnet.ts \
  --output outputs/audit.csv
```

The output includes COPD probability, predicted class, audio-quality indicators, and saliency-vs-random deletion summary metrics.

## Reproducibility

The default front end follows the reported acoustic pipeline:

- sample rate: 4 kHz
- high-pass filter: ~50 Hz
- STFT: 1024 samples
- hop: 512 samples (50% overlap)
- Mel bands: 64
- frequency range: 50-2000 Hz

See `REPRODUCIBILITY.md` and `config/default.yaml` for details.

## Safety note

This software is for research, education, demonstration, and model auditing. It must not be used as a stand-alone diagnostic system or as a substitute for spirometry, clinician assessment, or appropriate medical evaluation.
