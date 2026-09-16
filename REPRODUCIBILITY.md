# Reproducibility notes

## Front end
- 4 kHz sample rate
- ~50 Hz high-pass filtering
- peak normalization
- STFT window = 1024 samples
- hop = 512 samples
- 64 Mel bands
- 50-2000 Hz range
- log-Mel representation

## Normalization
For exact reproduction, provide training-derived per-band statistics in an NPZ file with `mean` and `std`. The app can fall back to per-recording band normalization but warns that this may not reproduce the validated pipeline exactly.

## Saliency
Default: 24 SmoothGrad perturbations, noise scale 0.08 × input standard deviation.

## Deletion audit
Default fractions: 0%, 5%, 10%, 15%, 20%, 30%; 20 random masks per fraction. The audit reports the area under the probability-drop curve for saliency-guided and random deletion and their difference.

## Randomness
Random deletion uses a fixed seed (2026) for repeatability.
