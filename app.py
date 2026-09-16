from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aerocopdnet.audio import AudioConfig, load_audio, logmel, normalize_logmel
from aerocopdnet.quality import QualityConfig, audio_quality
from aerocopdnet.inference import load_torchscript, predict_probability, predicted_class
from aerocopdnet.explainability import SaliencyConfig, smoothgrad_saliency, deletion_curve
from aerocopdnet.plots import waveform_figure, logmel_figure, saliency_overlay_figure, deletion_curve_figure

with open(ROOT / "config" / "default.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
AUDIO_CFG = AudioConfig(**CFG["audio"])
QUALITY_CFG = QualityConfig(**CFG["quality"])
SAL_CFG = SaliencyConfig(**CFG["saliency"])
THRESHOLD = float(CFG["model"]["threshold"])

st.set_page_config(page_title="AeroCOPDNet Interactive XAI", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(90deg,#043d50,#00616a); color:white; padding:18px 24px; border-radius:14px; margin-bottom:14px;}
.hero h1 {margin:0; font-size:2rem;} .hero p {margin:4px 0 0 0; opacity:.9;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>AeroCOPDNet Interactive Dashboard</h1><p>Respiratory-audio screening with reviewable acoustic evidence</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("1  Respiratory audio")
    uploaded = st.file_uploader("Upload .wav", type=["wav"])
    st.subheader("2  Optional reviewer context")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=67)
    sex = st.selectbox("Sex", ["Not specified", "Female", "Male", "Other"])
    chest = st.text_input("Chest location", value="")
    device_name = st.text_input("Recording equipment", value="")
    st.caption("These fields are context only; they are not acoustic model inputs in this repository.")
    st.subheader("3  Model files")
    checkpoint = st.text_input("TorchScript checkpoint", value=os.getenv("AEROCOPDNET_CHECKPOINT", str(ROOT / "checkpoints" / "aerocopdnet.ts")))
    norm_path = st.text_input("Normalization stats (.npz)", value=os.getenv("AEROCOPDNET_NORM", ""))

if uploaded is None:
    st.info("Upload a respiratory WAV file. Add a compatible TorchScript checkpoint for prediction, saliency and deletion auditing.")
    st.image(str(ROOT / "assets" / "dashboard_concept.png"), caption="Dashboard concept", use_container_width=True)
    st.stop()

try:
    y, sr = load_audio(uploaded, AUDIO_CFG)
    quality = audio_quality(y, sr, QUALITY_CFG)
    db = logmel(y, sr, AUDIO_CFG)
    x_norm, norm_label = normalize_logmel(db, norm_path or None)
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()

q1,q2,q3,q4 = st.columns(4)
q1.metric("Duration", f"{quality['duration_s']:.1f} s")
q2.metric("Silence fraction", f"{quality['silence_fraction']*100:.1f}%")
q3.metric("Clipping fraction", f"{quality['clipping_fraction']*100:.2f}%")
q4.metric("Audio quality", quality["status"].upper())
if quality["flags"]:
    st.warning("Audio quality flags: " + "; ".join(quality["flags"]))

c1,c2 = st.columns(2)
with c1: st.pyplot(waveform_figure(y, sr), clear_figure=True, use_container_width=True)
with c2: st.pyplot(logmel_figure(db, sr, AUDIO_CFG.hop_length), clear_figure=True, use_container_width=True)
st.caption(f"Front end: {sr} Hz, {AUDIO_CFG.n_mels} Mel bands, n_fft={AUDIO_CFG.n_fft}, hop={AUDIO_CFG.hop_length}; normalization: {norm_label}.")
if "fallback" in norm_label:
    st.warning("Training-derived normalization statistics were not supplied. Visualization remains valid, but inference may not exactly match the validated training pipeline.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_torchscript(checkpoint, DEVICE)
if model is None:
    st.warning("No compatible TorchScript model was found. Prediction and XAI are disabled; audio-quality and feature review remain available.")
    st.stop()

x = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
with torch.no_grad():
    p = float(predict_probability(model, x)[0].cpu())
label, class_prob = predicted_class(p, THRESHOLD)

m1,m2,m3 = st.columns(3)
m1.metric("Predicted class", label)
m2.metric("COPD probability", f"{p*100:.1f}%")
m3.metric("Predicted-class confidence", f"{class_prob*100:.1f}%")

try:
    sal = smoothgrad_saliency(model, x, SAL_CFG, label == "COPD")
    audit = deletion_curve(model, x, sal, SAL_CFG, label == "COPD")
except Exception as exc:
    st.error(f"Explainability computation failed: {exc}")
    st.stop()

left,right = st.columns([1.35,1.0])
with left:
    st.pyplot(saliency_overlay_figure(db, sal, sr, AUDIO_CFG.hop_length), clear_figure=True, use_container_width=True)
with right:
    st.pyplot(deletion_curve_figure(audit), clear_figure=True, use_container_width=True)
    st.metric("Faithfulness ΔAUC", f"{audit['faithfulness_delta_auc']:.4f}")
    st.caption("Positive ΔAUC means top-saliency deletion degrades predicted-class probability more strongly than random deletion over the tested fractions.")

st.subheader("Review summary")
summary = {
    "predicted_class": label,
    "copd_probability": p,
    "predicted_class_probability": class_prob,
    "audio_quality": quality,
    "normalization": norm_label,
    "saliency_audit": {
        "salient_drop_auc": audit["salient_drop_auc"],
        "random_drop_auc": audit["random_drop_auc"],
        "faithfulness_delta_auc": audit["faithfulness_delta_auc"],
        "fractions": audit["fractions"],
    },
    "context": {"age": age, "sex": sex, "chest_location": chest, "device": device_name},
}
st.json(summary)
st.download_button("Download audit JSON", data=json.dumps(summary, indent=2), file_name="aerocopdnet_audit.json", mime="application/json")
st.markdown("**Transparent review path:** Upload → Quality check → Transform → Predict → Explain → Faithfulness audit → Human review")
st.caption("Research prototype — supports review, not autonomous diagnosis.")
