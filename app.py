"""Local Streamlit interface for the video detector."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from deepfake_detector.model import select_device
from deepfake_detector.predict import load_checkpoint, score_video


st.set_page_config(page_title="Deepfake Video Detector", page_icon="🎞️", layout="centered")
st.title("Deepfake Video Detector")
st.caption("A research demo that estimates whether sampled video frames resemble DFDC fakes.")

st.info(
    "This model can make mistakes, especially on new manipulation methods or compressed footage. "
    "Treat its score as a research signal, not proof of authenticity."
)

default_checkpoint = os.environ.get("DEEPFAKE_CHECKPOINT", "artifacts/best_model.pt")
checkpoint_text = st.text_input("Model checkpoint", value=default_checkpoint)
uploaded_video = st.file_uploader("Choose an MP4 video", type=["mp4"])
frame_count = st.slider("Frames to analyze", min_value=4, max_value=32, value=16, step=4)

if st.button("Analyze video", type="primary", disabled=not uploaded_video):
    checkpoint_path = Path(checkpoint_text).expanduser()
    if not checkpoint_path.is_file():
        st.error(f"Checkpoint not found: {checkpoint_path}. Train a model first.")
    else:
        with st.spinner("Analyzing sampled frames locally…"):
            temporary_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary_file:
                    temporary_file.write(uploaded_video.getvalue())
                    temporary_path = temporary_file.name
                device = select_device()
                model = load_checkpoint(checkpoint_path, device)
                result = score_video(model, temporary_path, device, frame_count)
                probability = float(result["fake_probability"])
                st.metric("Estimated fake probability", f"{probability:.1%}")
                st.progress(probability)
                st.write(f"Model label: **{result['prediction'].replace('_', ' ')}**")
                st.caption(
                    f"Scored {result['frame_count']} frames. Frame score standard deviation: "
                    f"{result['frame_score_std']:.3f}. This variation is not a confidence interval."
                )
            except Exception as error:
                st.error(f"Could not analyze this video: {error}")
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)
