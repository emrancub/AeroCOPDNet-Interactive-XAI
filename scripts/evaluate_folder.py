from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aerocopdnet.audio import AudioConfig, load_audio, logmel, normalize_logmel
from aerocopdnet.quality import QualityConfig, audio_quality
from aerocopdnet.inference import load_torchscript, predict_probability, predicted_class
from aerocopdnet.explainability import SaliencyConfig, smoothgrad_saliency, deletion_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV containing at least a path column")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--norm", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    ac, qc, sc = AudioConfig(**cfg["audio"]), QualityConfig(**cfg["quality"]), SaliencyConfig(**cfg["saliency"])
    threshold = float(cfg["model"]["threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torchscript(args.checkpoint, device)
    if model is None:
        raise SystemExit("Checkpoint not found or invalid")
    df = pd.read_csv(args.csv)
    if "path" not in df.columns:
        raise SystemExit("Input CSV must contain a path column")
    rows=[]
    for _,r in df.iterrows():
        path=str(r["path"])
        try:
            y,sr=load_audio(path,ac); q=audio_quality(y,sr,qc); db=logmel(y,sr,ac); xn,norm=normalize_logmel(db,args.norm or None)
            x=torch.from_numpy(xn).unsqueeze(0).unsqueeze(0).to(device).float()
            with torch.no_grad(): p=float(predict_probability(model,x)[0].cpu())
            label,conf=predicted_class(p,threshold)
            sal=smoothgrad_saliency(model,x,sc,label=="COPD"); audit=deletion_curve(model,x,sal,sc,label=="COPD")
            rows.append({**r.to_dict(),"copd_probability":p,"predicted_class":label,"confidence":conf,"quality_status":q["status"],"silence_fraction":q["silence_fraction"],"clipping_fraction":q["clipping_fraction"],"faithfulness_delta_auc":audit["faithfulness_delta_auc"],"salient_drop_auc":audit["salient_drop_auc"],"random_drop_auc":audit["random_drop_auc"],"normalization":norm})
        except Exception as exc:
            rows.append({**r.to_dict(),"error":str(exc)})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(out)

if __name__ == "__main__": main()
