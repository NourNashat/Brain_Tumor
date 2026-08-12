import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_PATH = "brisc2025_efficientnetb3.keras"
IMG_SIZE = 300
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

st.set_page_config(
    page_title="BRISC2025 Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)


# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Mirror the exact preprocessing used at training time."""
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(pil_img)
    arr = tf.expand_dims(arr, axis=0)  # batch dim

    # training used: grayscale -> rgb (to strip any color cast), then
    # efficientnet's own preprocess_input
    arr = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(arr))
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def predict(model, pil_img: Image.Image, use_tta: bool = True):
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]

    if use_tta:
        # match the notebook's test-time augmentation (horizontal flip average)
        x_flip = tf.image.flip_left_right(x)
        probs_flip = model.predict(x_flip, verbose=0)[0]
        probs = (probs + probs_flip) / 2.0

    return probs


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🧠 BRISC2025 Brain Tumor MRI Classifier")
st.caption(
    "EfficientNetB3 model fine-tuned on the BRISC2025 dataset "
    "(glioma / meningioma / pituitary / no tumor)."
)

st.warning(
    "⚠️ Research / educational demo only. This tool is **not** a medical "
    "device and must not be used for real diagnosis or clinical decisions. "
    "Always consult a qualified radiologist or physician.",
    icon="⚠️",
)

with st.spinner("Loading model..."):
    model = load_model()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image (T1-weighted, axial/sagittal/coronal)",
    type=["jpg", "jpeg", "png"],
)

use_tta = st.checkbox(
    "Use test-time augmentation (flip averaging) — matches notebook eval",
    value=True,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference..."):
        probs = predict(model, image, use_tta=use_tta)

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted class", pred_class.replace("_", " ").title())
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.subheader("Class probabilities")
    prob_dict = {c.replace("_", " ").title(): float(p) for c, p in zip(CLASS_NAMES, probs)}
    st.bar_chart(prob_dict)

    with st.expander("Raw probabilities"):
        for c, p in sorted(prob_dict.items(), key=lambda kv: -kv[1]):
            st.write(f"**{c}**: {p:.4f}")
else:
    st.info("Upload an MRI image above to get a prediction.")

st.divider()
st.caption(
    "Model: `brisc2025_efficientnetb3.keras` · Input size: "
    f"{IMG_SIZE}×{IMG_SIZE} · Classes: {', '.join(CLASS_NAMES)}"
)
