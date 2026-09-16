# Checkpoints

Place a compatible TorchScript checkpoint here:

```text
checkpoints/aerocopdnet.ts
```

Expected input: `(B, 1, 64, T)` float tensor.
Expected output: `(B,)` or `(B,1)` logits/probabilities.

No fabricated or substitute model weights are bundled in this repository.
