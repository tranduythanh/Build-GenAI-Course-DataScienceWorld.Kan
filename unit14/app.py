from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from agents.toeic_crew import run_audio_pipeline, run_pipeline
from db import init_db

init_db()

st.title("🎯 TOEIC Speaking Tutor - PoC")

image_dir = Path(__file__).resolve().parent / "data" / "images"
image_files = [
    p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
]
selected_image = None
if image_files:
    selected_image = st.selectbox("Select TOEIC picture", image_files)
else:
    st.info("Add image files to data/images to begin.")
user_text = st.text_area("Describe the picture in English")
audio_file = st.file_uploader(
    "Or upload your spoken answer", type=["wav", "mp3"]
)  # noqa: E501

if st.button("Evaluate") and selected_image and (user_text or audio_file):
    tmp_dir = tempfile.mkdtemp()
    img_path = str(image_dir / selected_image)

    if audio_file:
        audio_path = os.path.join(tmp_dir, audio_file.name)
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        transcript, sample, score, feedback = run_audio_pipeline(
            img_path, audio_path
        )  # noqa: E501
        st.markdown("### Transcript")
        st.write(transcript)
    else:
        sample, score, feedback = run_pipeline(img_path, user_text)

    st.image(Image.open(img_path), caption="Selected Picture")
    st.markdown("### Sample Description")
    st.write(sample)
    st.markdown("### Assessment")
    st.write(score)
    st.markdown("### Feedback")
    st.write(feedback)
else:
    st.info("Please select a picture and provide your description.")
