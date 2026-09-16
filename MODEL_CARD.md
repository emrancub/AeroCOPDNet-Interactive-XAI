# AeroCOPDNet model card — dashboard integration

## Intended task
Binary respiratory-sound screening: COPD versus non-COPD.

## Reported acoustic model performance
Subject-wise five-fold results reported in the AeroCOPDNet study include accuracy 0.9622 ± 0.0091, AUROC 0.9955 ± 0.0021, AUPR 0.9979 ± 0.0008, F1 0.9706 ± 0.0073, sensitivity 0.9508 ± 0.0152, specificity 0.9839 ± 0.0198, and Brier score 0.0266 ± 0.0055.

## Important limitations
- Non-COPD is heterogeneous and includes healthy and other diseases.
- Cross-dataset transport is materially lower than pooled cross-validation.
- This dashboard is not prospectively clinically validated.
- Saliency is a model-behavior explanation, not a causal biomarker.
- Demographic/context fields are not acoustic model inputs in this repository.
