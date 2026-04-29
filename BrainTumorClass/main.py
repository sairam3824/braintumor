import os
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.autoencoder import build_autoencoder
from src.feature_extractor import extract_features
from src.classifier import train_svm, baseline_linear_svm

# ===============================
# 1️⃣ Load Dataset
# ===============================
print("Loading Dataset...")
X_train, y_train = load_data("dataset/training", augment=True)
X_test, y_test = load_data("dataset/testing", augment=False)

# ===============================
# 2️⃣ Build Autoencoder
# ===============================
autoencoder = build_autoencoder()

model_path = "models/advanced_autoencoder.keras"

if not os.path.exists(model_path):
    print("Training Autoencoder...")
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=25,
        batch_size=16,
        validation_data=(X_test, X_test)
    )
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    autoencoder.save(model_path)

    # Plot Training Curve (ONLY when training)
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

else:
    print("Loading Saved Model...")
    from tensorflow.keras.models import load_model
    autoencoder = load_model(model_path)

# ===============================
# 3️⃣ Feature Extraction
# ===============================
print("\nExtracting Features...")
features_train = extract_features(autoencoder, X_train)
features_test = extract_features(autoencoder, X_test)

# ===============================
# 4️⃣ Model Comparison
# ===============================
print("\nRunning Baseline Model...")
baseline_acc = baseline_linear_svm(features_train, features_test, y_train, y_test)

print("\nRunning Advanced Model...")
train_svm(features_train, features_test, y_train, y_test)