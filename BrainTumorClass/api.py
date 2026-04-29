"""
Flask REST API for Brain Tumor Classification.
Wraps the existing prediction pipeline (Autoencoder + PCA + SVM).
"""

import os
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
import joblib
import base64
from src.xai_gradcam import build_gradcam_model, generate_heatmap, create_overlay
# ============================================================
# App Setup
# ============================================================
app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = 128

# ============================================================
# Load Models Once at Startup
# ============================================================
print("Loading models...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
autoencoder = load_model(os.path.join(BASE_DIR, "models", "advanced_autoencoder.keras"))
svm = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "models", "pca_model.pkl"))

# Build encoder (bottleneck extractor) once
encoder = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.get_layer("bottleneck").output
)

# Build Grad-CAM model once
grad_model = build_gradcam_model(autoencoder, conv_layer_name="conv2d_2", bottleneck_layer_name="bottleneck")

print("All models loaded successfully!")

# ============================================================
# Helper Functions
# ============================================================

def preprocess_image(image_path):
    """Read, resize, normalize an image for the model."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image could not be loaded. Check the file format.")
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 1))
    return img


def predict(image_path):
    """Run the full prediction pipeline on an image."""
    img = preprocess_image(image_path)

    # Feature extraction via encoder bottleneck
    features = encoder.predict(img, verbose=0)

    # Dimensionality reduction
    features_pca = pca.transform(features)

    # SVM prediction
    prediction = svm.predict(features_pca)
    probabilities = svm.predict_proba(features_pca)[0]

    predicted_class = CLASS_NAMES[prediction[0]]
    confidence = float(np.max(probabilities) * 100)

    prob_dict = {
        CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    
    # --- Grad-CAM Generation ---
    try:
        heatmap_array = generate_heatmap(img, grad_model)
        overlay = create_overlay(image_path, heatmap_array)
        _, buffer = cv2.imencode('.jpg', overlay)
        gradcam_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"GradCAM Error: {e}")
        gradcam_base64 = None

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": prob_dict,
        "is_tumor": predicted_class != "no_tumor",
        "gradcam_image": gradcam_base64
    }

# ============================================================
# API Routes
# ============================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "models_loaded": True})


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """Accept an MRI image and return tumor classification."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send a file with key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Validate extension
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Invalid file type '{ext}'. Allowed: {', '.join(allowed)}"}), 400

    # Save to temp
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        file.save(save_path)
        result = predict(save_path)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return the list of supported tumor classes."""
    return jsonify({
        "classes": CLASS_NAMES,
        "descriptions": {
            "glioma": "A tumor that starts in the glial cells of the brain or spine.",
            "meningioma": "A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.",
            "no_tumor": "No tumor detected in the MRI scan.",
            "pituitary": "A tumor that forms in the pituitary gland near the base of the brain."
        }
    })


# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
