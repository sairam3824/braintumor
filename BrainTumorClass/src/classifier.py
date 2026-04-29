from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

def train_svm(features_train, features_test, y_train, y_test):

    print("\nApplying PCA...")
    pca = PCA(n_components=50)
    features_train_pca = pca.fit_transform(features_train)
    features_test_pca = pca.transform(features_test)

    print("Training Advanced RBF SVM...")
    svm = SVC(kernel='rbf', C=300, gamma=0.01, probability=True)
    svm.fit(features_train_pca, y_train)

    predictions = svm.predict(features_test_pca)

    acc = accuracy_score(y_test, predictions)

    print("\nFinal Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 🔥 Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # 🔥 Save models
    joblib.dump(svm, "models/svm_model.pkl")
    joblib.dump(pca, "models/pca_model.pkl")

    print("\nModels saved successfully in models/ folder!")

    return acc


def baseline_linear_svm(features_train, features_test, y_train, y_test):    
    svm = SVC(kernel='linear')
    svm.fit(features_train, y_train)
    predictions = svm.predict(features_test)
    acc = accuracy_score(y_test, predictions)
    print("\nBaseline Linear SVM Accuracy:", acc)
    return acc