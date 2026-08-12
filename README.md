# Deploying the BRISC2025 classifier on Streamlit Community Cloud

## 1. Files you need in one folder/repo
```
your-repo/
├── app.py
├── requirements.txt
└── brisc2025_efficientnetb3.keras   ← your trained model (from Colab)
```
Download the model from Google Drive first — in your notebook it saved to
`/content/drive/MyDrive/brisc2025_models/brisc2025_efficientnetb3.keras` —
and place a copy next to `app.py`. In the Drive web UI: right-click the file
→ **Download**.

> Model size check: EfficientNetB3 `.keras` files are typically 40–50 MB.
> GitHub blocks files >100 MB by default. If yours is close to or over that,
> use [Git LFS](https://git-lfs.com/):
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> ```
> then continue with the steps below as normal.

## 2. Push to GitHub
```bash
git init
git add app.py requirements.txt brisc2025_efficientnetb3.keras
git commit -m "BRISC2025 Streamlit app"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## 3. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"New app"** → pick your repo/branch → set **Main file path** to `app.py`.
3. Click **Deploy**. First build takes a few minutes (installing TensorFlow).

That's it — you'll get a public `*.streamlit.app` URL.

## 4. Notes on what the app replicates from your notebook
- **Resize** to 300×300 (EfficientNetB3 input size).
- **Grayscale→RGB round-trip** before preprocessing, same as your `preprocess()` function.
- **`efficientnet.preprocess_input`** for normalization (matches training/inference).
- **Optional TTA**: horizontal-flip averaging, same as your final test evaluation cell.
- **Class order**: `glioma, meningioma, no_tumor, pituitary` — this is the
  alphabetical order Keras's `image_dataset_from_directory` uses. **Double-check
  this against the `class_names` printout in your own notebook run** before
  trusting predictions — if your folder names differed even slightly, the order
  (and therefore the labels) could shift.

## 5. Common gotchas
- **Out-of-memory on free tier**: Streamlit Community Cloud free apps have ~1GB
  RAM. TensorFlow + EfficientNetB3 usually fits, but if it crashes on load,
  switch `requirements.txt` to `tensorflow-cpu` (already set) and avoid loading
  the model more than once (the app already uses `@st.cache_resource`).
- **Slow cold start**: first prediction after the app wakes up will be slower
  while TensorFlow initializes — this is normal.
- **Local testing** before deploying:
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```
