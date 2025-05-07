import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from collections import defaultdict
import os

# Page config
st.set_page_config(
    page_title="NIT Warangal | Steel Surface Defect Detection",
    page_icon="🛠️",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .reportview-container .main {
        overflow: hidden;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2 {
        color: #0a3d62;
        text-align: center;
    }
    .stButton>button {
        background-color: #0a3d62;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    .stFileUploader label {
        color: #0a3d62;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("National Institute of Technology, Warangal")
st.subheader("🛠️ AI-Based Steel Surface Defect Detection System")
st.markdown("Upload an image of a **hot rolled steel strip** to classify surface defects using a **MobileNetV2 deep learning model**.")

# Load MobileNetV2 model structure and weights
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights=None, classes=6)
    model.load_weights("weight.weights.h5")
    return model

model = load_model()

# Class labels (adjust according to your actual class order)
class_labels = ["Crazing", "Patches", "Pitted_surface", "Rolled-in_scale", "Scratches", "Inclusion"]

# Defect Knowledge Base
defect_knowledge = {
    "Crazing": {
        "Cause": "Tensile stress beyond material limit due to cooling issues or high rolling speed.",
        "Prevention": "Optimize rolling speed and ensure uniform cooling."
    },
    "Patches": {
        "Cause": "Local oxidation or improper cleaning before rolling.",
        "Prevention": "Maintain surface cleanliness and control mill scale formation."
    },
    "Pitted_surface": {
        "Cause": "Localized corrosion or trapped air bubbles during rolling.",
        "Prevention": "Improve descaling processes and surface inspection."
    },
    "Rolled-in_scale": {
        "Cause": "Oxide scales not removed properly before rolling.",
        "Prevention": "Enhance descaling efficiency and pre-cleaning."
    },
    "Scratches": {
        "Cause": "Abrasive particles or improper handling.",
        "Prevention": "Maintain clean rollers and handling equipment."
    },
    "Inclusion": {
        "Cause": "Non-metallic particles embedded during manufacturing.",
        "Prevention": "Use high-quality raw materials and controlled processing."
    }
}

# Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image for MobileNetV2
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]
        top_idx = np.argmax(predictions)
        top_class = class_labels[top_idx]
        confidence = predictions[top_idx]

        st.subheader("📋 Predicted Defect")
        st.markdown(f"### 🔹 {top_class}")
        st.markdown(f"- **Confidence:** {confidence:.2f}")

        if top_class in defect_knowledge:
            st.markdown(f"**🛠 Cause:** {defect_knowledge[top_class]['Cause']}")
            st.markdown(f"**✅ Prevention:** {defect_knowledge[top_class]['Prevention']}")
        else:
            st.warning("⚠️ No information found for this defect.")

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("Please upload a valid image file (jpg, jpeg, png, bmp, tiff, webp).")
