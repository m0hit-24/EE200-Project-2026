"""
app.py

Streamlit app for the audio-fingerprinting song identifier.
Two modes:
  1. Single-clip mode — upload one query, see spectrogram, constellation,
     offset histogram, and the predicted song.
  2. Batch mode — upload many queries, get results.csv with columns
     filename, prediction.
"""

import os
import tempfile

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from fingerprint.engine import identify, load_database

st.set_page_config(page_title="Song Identifier", layout="wide")


@st.cache_resource
def get_database():
    return load_database("database.pkl")


database = get_database()

st.title("Audio fingerprinting song identifier")
mode = st.radio("Choose mode", ["Single-clip mode", "Batch mode"], horizontal=True)


# ──────────────────────────────────────────────────────────────────
# SINGLE-CLIP MODE
# ──────────────────────────────────────────────────────────────────
if mode == "Single-clip mode":
    uploaded = st.file_uploader("Upload a query clip (.wav or .mp3)", type=["wav", "mp3"])

    if uploaded is not None:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Fingerprinting and matching..."):
            result = identify(tmp_path, database)

        st.subheader("Result")
        if result['prediction']:
            st.success(f"Identified song: **{result['prediction']}**")
        else:
            st.error("No confident match found.")

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Spectrogram")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pcolormesh(result['t'], result['f'], result['S_dB'],
                          shading='gouraud', cmap='magma', vmin=-80, vmax=0)
            ax.set_ylim(0, 4000)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            st.pyplot(fig)

        with col2:
            st.caption("Constellation map")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            pf = [result['f'][fi] for fi, ti in result['peaks']]
            pt = [result['t'][ti] for fi, ti in result['peaks']]
            ax2.scatter(pt, pf, s=6, c='red')
            ax2.set_ylim(0, 4000)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Frequency (Hz)")
            st.pyplot(fig2)

        st.caption("Offset histogram (top candidates)")
        top_songs = sorted(result['scores'], key=result['scores'].get, reverse=True)[:3]
        if top_songs:
            fig3, axes = plt.subplots(1, len(top_songs), figsize=(5 * len(top_songs), 3))
            if len(top_songs) == 1:
                axes = [axes]
            for ax, song in zip(axes, top_songs):
                counter = result['offsets'][song]
                xs = sorted(counter.keys())
                ys = [counter[x] for x in xs]
                ax.bar(xs, ys, width=0.08)
                ax.set_title(f"{song} (votes={result['scores'][song]})")
                ax.set_xlabel("Offset (s)")
            st.pyplot(fig3)
        else:
            st.info("No candidate songs received any matching hashes.")

        os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────────
# BATCH MODE
# ──────────────────────────────────────────────────────────────────
else:
    uploaded_files = st.file_uploader(
        "Upload one or more query clips", type=["wav", "mp3"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Run batch identification"):
        rows = []
        progress = st.progress(0.0)

        for i, uploaded in enumerate(uploaded_files):
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            result = identify(tmp_path, database)
            prediction = result['prediction'] if result['prediction'] else ""
            rows.append({"filename": uploaded.name, "prediction": prediction})

            os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded_files))

        results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(results_df)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", csv_bytes, file_name="results.csv", mime="text/csv")

        results_df.to_csv("results.csv", index=False)
        st.success("results.csv written.")
