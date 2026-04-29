import joblib
from tensorflow.keras.models import load_model

def analyze_models():
    # 1. Keras model
    try:
        model = load_model("models/advanced_autoencoder.keras")
        print("--- Keras Autoencoder Layers ---")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name} | Type: {type(layer).__name__} | Output: {layer.output_shape}")
    except Exception as e:
        print(f"Error loading Keras model: {e}")

    # 2. PCA model
    try:
        pca = joblib.load("models/pca_model.pkl")
        print("\n--- PCA Properties ---")
        print(f"Components: {pca.n_components_}")
    except Exception as e:
        print(f"Error loading PCA: {e}")

    # 3. SVM model
    try:
        svm = joblib.load("models/svm_model.pkl")
        print("\n--- SVM Properties ---")
        print(f"Type: {type(svm).__name__}")
        if hasattr(svm, "kernel"):
            print(f"Kernel: {svm.kernel}")
        if hasattr(svm, "classes_"):
            print(f"Classes: {svm.classes_}")
    except Exception as e:
        print(f"Error loading SVM: {e}")

if __name__ == "__main__":
    analyze_models()
