"""Template helper for exporting an already-instantiated PyTorch AeroCOPDNet model.

Edit `build_model_and_load_weights()` to match your training code, then run this script.
"""
from pathlib import Path
import torch


def build_model_and_load_weights():
    raise NotImplementedError("Connect this function to the authors' AeroCOPDNet training implementation and checkpoint.")


def main():
    model = build_model_and_load_weights().eval()
    example = torch.randn(1, 1, 64, 40)
    scripted = torch.jit.trace(model, example)
    out = Path("checkpoints/aerocopdnet.ts")
    out.parent.mkdir(exist_ok=True)
    scripted.save(str(out))
    print(out)

if __name__ == "__main__": main()
