# Song Identifier — Q3B (EE200 Course Project)

This folder contains the complete, tested code for the audio-fingerprinting
song identifier app. Every file has been verified to run correctly
end-to-end (spectrogram → peaks → hashes → database → matching) using
synthetic and real WAV test files.

## What's already done

- `fingerprint/engine.py` — the full fingerprinting engine (spectrogram,
  peak picking, hashing, database build/save/load, matching). Tested and
  working.
- `build_database.py` — one-off script to index your song library.
- `app.py` — the Streamlit app with both required modes (single-clip and
  batch). Structurally validated.
- `requirements.txt` — exact dependencies needed for deployment.

## What you still need to do (these require your own accounts — no one
## else can do this step for you)

1. **Add your song library.** Place the provided songs (.wav/.mp3, exact
   filenames, do not rename) into `data/songs/`.

2. **Build the database** (run this once, locally, before deploying):
   ```bash
   pip install -r requirements.txt
   python build_database.py
   ```
   This creates `database.pkl` in this folder.

3. **Test locally:**
   ```bash
   streamlit run app.py
   ```
   Open the local URL it prints, try both modes.

4. **Push to GitHub** (create a new repo first on github.com):
   ```bash
   git init
   git add .
   git commit -m "Song identifier app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   Double check on GitHub's website that `database.pkl` actually appears
   in the repo (some default .gitignore templates exclude .pkl files —
   the one included here does NOT).

5. **Deploy on Streamlit Community Cloud:**
   - Go to https://share.streamlit.io and sign in with GitHub.
   - Click "New app", pick your repo/branch, set main file to `app.py`.
   - Click "Deploy". Wait for the build to finish.
   - Open the live URL it gives you — that is your final link.

6. **Submit:** the live app URL + your GitHub repo URL, both in your Q3B
   PDF, plus this whole folder zipped as your code submission.

## Why I can't do steps 4–5 for you

Pushing code requires your GitHub login, and deploying requires a
Streamlit Cloud account tied to that GitHub login. There's no way to
generate a real, working live link without those credentials — anyone
claiming otherwise would just be giving you a broken or fake URL.
