import streamlit as st
import numpy as np
import pickle
import random
import os
import gdown
from PIL import Image
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(page_title="YasOrna Visual Search", layout="wide")

# ----------------------------------------
# Download model if not exists
# ----------------------------------------
MODEL_PATH = "feature_extractor.h5"
GOOGLE_DRIVE_ID = "18ALhVqJ0We9MMs3wzqytbdvWHCDw2Bc0"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model from Google Drive..."):
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)

# ----------------------------------------
# Load model and feature files
# ----------------------------------------
feat_extractor = load_model(MODEL_PATH)

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

pca_features = np.load("pca_features.npy")

with open("valid_images.pkl", "rb") as f:
    valid_images = pickle.load(f)

# ----------------------------------------
# Fix paths for Linux
# ----------------------------------------
IMAGE_BASE_DIR = os.path.join(os.getcwd(), "Images")

def fix_path(path):
    fixed = os.path.normpath(path.replace("\\", "/"))
    subpath = os.path.join(*fixed.split(os.sep)[-2:])
    return os.path.join(IMAGE_BASE_DIR, subpath)

valid_images = [fix_path(p) for p in valid_images]

# ----------------------------------------
# Helper Functions
# ----------------------------------------
def process_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_closest_images(query_features, num_results=10):
    query_pca_features = pca.transform(query_features)[0]
    distances = [distance.cosine(query_pca_features, feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[:num_results]
    return idx_closest

def get_random_description():
    descriptions = [
        "Elegant diamond–studded ring.",
        "Premium handcrafted design.",
        "Perfect for every occasion.",
        "Stylish and timeless beauty.",
        "Modern design with a classic feel.",
        "Crafted with precision and love.",
        "A true statement piece.",
        "Graceful charm for daily wear."
    ]
    return random.choice(descriptions)

def store_details_and_open(img_path, price, description):
    st.session_state["selected_product"] = {
        "img_path": img_path,
        "price": price,
        "description": description
    }
    st.session_state["page"] = "details"

# ----------------------------------------
# UI Logic
# ----------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state["page"] == "home":
    st.title("🔍 Visual Jewelry Search")
    st.markdown("## 📤 Upload or 📸 Capture an Image")

    upload_tab, camera_tab = st.tabs(["Upload", "Capture"])
    uploaded_file = None

    with upload_tab:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    with camera_tab:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            uploaded_file = camera_image

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Selected Image", use_container_width=True)

        img, x = process_image(uploaded_file)
        features = feat_extractor.predict(x)

        similar_images = get_closest_images(features)

        st.subheader("🔗 Most Similar Products")

        cols = st.columns(3)
        for i, idx in enumerate(similar_images):
            img_path = valid_images[idx]
            price = f"₹{random.randint(5000, 100000):,}"
            description = get_random_description()

            with cols[i % 3]:
                st.image(img_path, use_container_width=True)
                if st.button(f"View → {i}", key=f"view_{i}"):
                    store_details_and_open(img_path, price, description)

elif st.session_state["page"] == "details":
    product = st.session_state.get("selected_product", {})
    if product:
        st.markdown("### 🛍️ Product Details")
        st.image(product["img_path"], width=500)
        st.markdown(f"### {product['price']}")
        st.markdown(f"**{product['description']}**")

        if st.button("🔙 Back to Results"):
            st.session_state["page"] = "home"
    else:
        st.warning("No product selected. Go back and choose one.")
