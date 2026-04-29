import numpy as np
import cv2
import joblib
import os
from tensorflow.keras.models import load_model
from src.feature_extractor import extract_features

# ==============================
# 1️⃣ Load Saved Models
# ==============================

print("Loading saved models...")

autoencoder = load_model("models/advanced_autoencoder.keras")
svm = joblib.load("models/svm_model.pkl")
pca = joblib.load("models/pca_model.pkl")

print("Models loaded successfully!")

# IMPORTANT: Class mapping (must match training order)
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ==============================
# 2️⃣ Image Preprocessing
# ==============================

def preprocess_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image could not be loaded. Check the path.")

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 1))

    return img

# ==============================
# 3️⃣ Predict Function
# ==============================

def predict_image(image_path):

    img = preprocess_image(image_path)

    # Feature extraction
    features = extract_features(autoencoder, img)

    # Apply PCA
    features_pca = pca.transform(features)

    # Prediction
    prediction = svm.predict(features_pca)
    probabilities = svm.predict_proba(features_pca)

    predicted_class = class_names[prediction[0]]
    confidence = np.max(probabilities) * 100

    print("\nPredicted Class:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")

    if predicted_class == "no_tumor":
        print("✅ No Tumor Detected")
    else:
        print(f"🧠 Tumor Type Detected: {predicted_class}")

# ==============================
# 4️⃣ Run Prediction
# ==============================

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    if os.path.exists(image_path):
        predict_image(image_path)
    else:
        print("Invalid image path!")