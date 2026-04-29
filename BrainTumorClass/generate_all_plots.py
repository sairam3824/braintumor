import os
import random
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import load_model

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.data_loader import load_data
from src.feature_extractor import extract_features
from src.xai_gradcam import build_gradcam_model, generate_heatmap

# Create diagrams folder
DIAGRAMS_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(DIAGRAMS_DIR, exist_ok=True)

print("Loading Models and Data for Diagrams...")
autoencoder = load_model("models/advanced_autoencoder.keras")
svm = joblib.load("models/svm_model.pkl")
pca_50 = joblib.load("models/pca_model.pkl")

X_test, y_test = load_data("dataset/testing", augment=False)
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ==========================================
# 1. CONFUSION MATRIX
# ==========================================
print("1/3 Generating Confusion Matrix...")
features_test = extract_features(autoencoder, X_test)
features_test_pca = pca_50.transform(features_test)
predictions = svm.predict(features_test_pca)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig(os.path.join(DIAGRAMS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ==========================================
# 2. PCA 2D SCATTER PLOT
# ==========================================
print("2/3 Generating PCA 2D Feature Scatter Plot...")
pca_2d = PCA(n_components=2)
features_2d = pca_2d.fit_transform(features_test)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_test, cmap="viridis", alpha=0.7)
plt.title("2D PCA Visualization of Autoencoder Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

def get_class_name(val, loc):
    idx = int(val)
    if idx >= 0 and idx < len(class_names):
        return class_names[idx]
    return ""

formatter = plt.FuncFormatter(get_class_name)
plt.colorbar(scatter, ticks=[0, 1, 2, 3], format=formatter)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIAGRAMS_DIR, "pca_scatter.png"), dpi=150)
plt.close()

# ==========================================
# 3. GRAD-CAM EXPLAINABILITY VISUALIZATIONS
# ==========================================
print("3/3 Generating Grad-CAM Heatmaps...")
grad_model = build_gradcam_model(autoencoder, conv_layer_name="conv2d_2", bottleneck_layer_name="bottleneck")

num_samples = 4
indices = random.sample(range(len(X_test)), num_samples)

plt.figure(figsize=(16, 8))
for i, idx in enumerate(indices):
    img_array = X_test[idx:idx+1]
    actual_class = class_names[y_test[idx]]
    
    heatmap = generate_heatmap(img_array, grad_model)
    
    img_display = np.squeeze(img_array)
    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(img_display, cmap='gray')
    plt.title(f"Original: {actual_class}")
    plt.axis('off')
    
    plt.subplot(2, num_samples, i + 1 + num_samples)
    plt.imshow(img_display, cmap='gray')
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(DIAGRAMS_DIR, "gradcam_heatmaps.png"), dpi=150)
plt.close()

print("All diagrams generated successfully in the 'diagrams' folder!")
