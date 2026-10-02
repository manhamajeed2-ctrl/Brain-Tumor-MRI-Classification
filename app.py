"""
Streamlit Web App – Brain Tumor MRI Image Classification
Upload an MRI scan and get real-time tumor type prediction.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.models import get_model
from src.dataset import get_transforms, CLASS_NAMES
from src.utils import load_model_for_inference

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------
MODELS_DIR = Path("models")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_MODELS = {
    "Custom CNN": "custom_cnn",
    "ResNet (from scratch)": "resnet_scratch",
}


@st.cache_resource
def load_checkpoint(model_key: str):
    """Load model + class names from best checkpoint."""
    ckpt_path = MODELS_DIR / f"{model_key}_best.pth"
    if not ckpt_path.exists():
        return None, None, f"Model file not found: {ckpt_path}"

    try:
        model = get_model(model_key, num_classes=len(CLASS_NAMES))
        model, class_names = load_model_for_inference(model, str(ckpt_path), DEVICE)
        return model, class_names, None
    except Exception as e:
        return None, None, str(e)


def predict(model, image: Image.Image, class_names):
    """Run inference on a single PIL image."""
    transform = get_transforms(IMG_SIZE, is_train=False)
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], probs, pred_idx


def main():
    st.title("🧠 Brain Tumor MRI Image Classification")
    st.markdown(
        """
        Upload a brain MRI scan and the model will classify it into one of four categories:
        **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.
        """
    )

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    model_display = st.sidebar.selectbox(
        "Select Model",
        list(AVAILABLE_MODELS.keys()),
        index=0,
    )
    model_key = AVAILABLE_MODELS[model_display]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **How to use**
        1. Choose a trained model
        2. Upload a clear axial / coronal MRI image (JPG / PNG)
        3. Click **Predict**
        """
    )
    st.sidebar.info(
        "This tool is for educational / research purposes only "
        "and is **not** a substitute for professional medical diagnosis."
    )

    # Load model
    model, class_names, err = load_checkpoint(model_key)
    if err:
        st.error(
            f"⚠️ Could not load model `{model_key}`.\n\n"
            f"{err}\n\n"
            "Please run `python train.py` first to generate the weights."
        )
        st.stop()

    st.success(f"✅ Model **{model_display}** loaded successfully on `{DEVICE}`.")

    # File uploader
    uploaded = st.file_uploader(
        "Choose an MRI image…",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan (preferably 2D axial or coronal view).",
    )

    col1, col2 = st.columns(2)

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        if st.button("🔍 Predict", type="primary", use_container_width=True):
            with st.spinner("Running inference…"):
                pred_class, probs, pred_idx = predict(model, image, class_names)

            with col2:
                st.subheader("Prediction Result")
                st.markdown(f"### 🩺 **{pred_class.replace('_', ' ').title()}**")
                st.metric("Confidence", f"{probs[pred_idx]*100:.1f}%")

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(6, 3.5))
                colors = ["#2ecc71" if i == pred_idx else "#3498db" for i in range(len(class_names))]
                bars = ax.barh(
                    [c.replace("_", " ").title() for c in class_names],
                    probs * 100,
                    color=colors,
                )
                ax.set_xlabel("Probability (%)")
                ax.set_xlim(0, 100)
                ax.set_title("Class Probabilities")
                for bar, p in zip(bars, probs):
                    ax.text(
                        bar.get_width() + 1,
                        bar.get_y() + bar.get_height() / 2,
                        f"{p*100:.1f}%",
                        va="center",
                        fontsize=9,
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                if pred_class == "no_tumor":
                    st.info("The model did not detect a tumor in this scan.")
                else:
                    st.warning(
                        f"The model suggests a **{pred_class.replace('_', ' ')}** tumor. "
                        "Please consult a qualified radiologist for definitive diagnosis."
                    )
    else:
        st.info("👆 Upload an MRI image to get started.")

    st.markdown("---")
    st.caption(
        "Built with PyTorch + Streamlit | Brain Tumor MRI Classification Capstone Project"
    )


if __name__ == "__main__":
    main()
