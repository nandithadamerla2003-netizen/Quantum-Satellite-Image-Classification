import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


# =====================================================
# 1. PATHS
# =====================================================

BACKEND_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BACKEND_FOLDER, "models")

tf.get_logger().setLevel("ERROR")


# =====================================================
# 2. LOAD DATASET (ALL IMAGES)
# =====================================================

import importlib.util

preprocessing_file = os.path.join(
    BACKEND_FOLDER,
    "preprocessing",
    "image_preprocessing.py"
)

spec = importlib.util.spec_from_file_location(
    "image_preprocessing",
    preprocessing_file
)

preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)

X_train, X_test, y_train, y_test, encoder = (
    preprocessing.prepare_dataset()
)

y_train = np.asarray(y_train, dtype=int)
y_test = np.asarray(y_test, dtype=int)

if y_train.min() == 1 and y_train.max() == 5:
    y_train -= 1
    y_test -= 1

X_train = X_train.reshape(-1, 64, 64, 3)
X_test = X_test.reshape(-1, 64, 64, 3)

print("Train images:", len(X_train))
print("Test images :", len(X_test))


# =====================================================
# 3. CLASSICAL SVM
# =====================================================

print("\nLoading Classical SVM...")

classical_svm = joblib.load(
    os.path.join(MODEL_FOLDER, "classical_svm.pkl")
)

X_test_flat = X_test.reshape(len(X_test), -1)

classical_predictions = classical_svm.predict(
    X_test_flat
)

print("Classical prediction completed!")


# =====================================================
# 4. LOAD SAVED CNN FEATURE EXTRACTOR
# =====================================================

print("\nLoading CNN feature extractor...")

cnn = tf.keras.models.load_model(
    os.path.join(MODEL_FOLDER, "cnn_feature_extractor.keras"),
    compile=False
)


# =====================================================
# 5. FAST CNN FEATURE EXTRACTION
# =====================================================

print("\nExtracting CNN features...")

train_features = cnn.predict(
    X_train,
    batch_size=128,
    verbose=0
)

test_features = cnn.predict(
    X_test,
    batch_size=128,
    verbose=0
)

print("CNN features extracted!")


# =====================================================
# 6. PCA AND SCALER
# =====================================================

pca = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_pca.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_scaler.pkl")
)

train_features = scaler.transform(
    pca.transform(train_features)
)

test_features = scaler.transform(
    pca.transform(test_features)
)

print("PCA and scaling completed!")


# =====================================================
# 7. QUANTUM FEATURE MAP
# =====================================================

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear"
)

parameters = list(feature_map.parameters)


# =====================================================
# 8. FAST STATEVECTOR CREATION
# =====================================================

def create_states(data):

    states = np.empty(
        (len(data), 16),
        dtype=np.complex128
    )

    for i, row in enumerate(data):

        circuit = feature_map.assign_parameters(
            dict(zip(parameters, row))
        )

        states[i] = Statevector.from_instruction(
            circuit
        ).data

    return states


print("\nCreating quantum states...")

train_states = create_states(train_features)
test_states = create_states(test_features)

print("Quantum states created!")


# =====================================================
# 9. FAST QUANTUM KERNEL (NUMPY)
# =====================================================

print("\nCalculating quantum kernel...")

# Training kernel: 1200 x 1200
train_kernel = np.abs(
    train_states @ train_states.conj().T
) ** 2

# Testing kernel: 300 x 1200
test_kernel = np.abs(
    test_states @ train_states.conj().T
) ** 2

print("Quantum kernel completed!")


# =====================================================
# 10. LOAD QUANTUM SVM
# =====================================================

print("\nLoading Quantum SVM...")

quantum_svm = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_svm.pkl")
)


# =====================================================
# 11. QUANTUM PREDICTION
# =====================================================

print("\nPredicting Quantum SVM...")

quantum_predictions = quantum_svm.predict(
    test_kernel
)

print("Prediction completed!")


# =====================================================
# 12. EVALUATION
# =====================================================

models = {
    "Classical SVM": classical_predictions,
    "Quantum SVM": quantum_predictions
}

results = {}

for name, predictions in models.items():

    results[name] = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )
    }


# =====================================================
# 13. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 50)
print("CLASSICAL SVM vs QUANTUM SVM")
print("=" * 50)

for name, metrics in results.items():

    print(f"\n{name}")

    for metric, value in metrics.items():

        print(f"{metric}: {value * 100:.2f}%")


# =====================================================
# 14. ACCURACY GRAPH
# =====================================================

classical_accuracy = results["Classical SVM"]["Accuracy"] * 100
quantum_accuracy = results["Quantum SVM"]["Accuracy"] * 100

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Classical SVM", "Quantum SVM"],
    [classical_accuracy, quantum_accuracy]
)

plt.ylabel("Accuracy (%)")
plt.title("Classical SVM vs Quantum SVM")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(
    bars,
    [classical_accuracy, quantum_accuracy]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(MODEL_FOLDER, "accuracy_comparison.png"),
    dpi=200
)

plt.show()


# =====================================================
# 15. CONFUSION MATRICES
# =====================================================

class_names = [
    "Forest",
    "Highway",
    "Industrial",
    "PermanentCrop",
    "River"
]

classical_cm = confusion_matrix(
    y_test,
    classical_predictions,
    labels=[0, 1, 2, 3, 4]
)

quantum_cm = confusion_matrix(
    y_test,
    quantum_predictions,
    labels=[0, 1, 2, 3, 4]
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, title in zip(
    axes,
    [classical_cm, quantum_cm],
    ["Classical SVM", "Quantum SVM"]
):

    ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

plt.savefig(
    os.path.join(MODEL_FOLDER, "confusion_matrix_comparison.png"),
    dpi=200
)

plt.show()


# =====================================================
# 16. FINAL
# =====================================================

print("\n====================================")
print("EVALUATION COMPLETED")
print("====================================")

print(f"Classical SVM Accuracy: {classical_accuracy:.2f}%")
print(f"Quantum SVM Accuracy  : {quantum_accuracy:.2f}%")

print("\nGraphs saved in:", MODEL_FOLDER)