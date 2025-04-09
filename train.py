# Import required libraries
import numpy as np
import os
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.decomposition import PCA
from PIL import Image

# Set directory and parameters
base_dir = './Images'
img_size = (224, 224)
max_images = 2000  # Limit the number of images processed

# Collect image paths
image_extensions = ['.jpg', '.png', '.jpeg']
images = [
    os.path.join(dp, f) for dp, dn, filenames in os.walk(base_dir)
    for f in filenames if os.path.splitext(f)[1].lower() in image_extensions
]

# Limit images if necessary
if len(images) > max_images:
    images = random.sample(images, max_images)

print(f"Using {len(images)} images for feature extraction.")

# Load VGG16 model pre-trained on ImageNet
base_model = VGG16(weights="imagenet", include_top=True)
feat_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

# Function to process an image
def load_and_process_image(path):
    img = image.load_img(path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Extract features for all images
features = []
valid_images = []  # Store paths of successfully processed images

for img_path in tqdm(images, desc="Extracting Features"):
    try:
        # Verify image
        with Image.open(img_path) as img:
            img.verify()

        # Reopen image and extract features
        x = load_and_process_image(img_path)
        feat = feat_extractor.predict(x)[0]
        features.append(feat)
        valid_images.append(img_path)

    except Exception as e:
        print(f"Skipping corrupted image: {img_path} -> {e}")

features = np.array(features)
print(f"Extracted features for {len(features)} images.")

# Apply PCA for dimensionality reduction
pca = PCA(n_components=min(500, len(features), features.shape[1]))
pca_features = pca.fit_transform(features)

# Save models and features
feat_extractor.save("feature_extractor.keras")
with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
np.save("pca_features.npy", pca_features)
with open("valid_images.pkl", "wb") as f:
    pickle.dump(valid_images, f)

print("Model and features saved successfully!")
