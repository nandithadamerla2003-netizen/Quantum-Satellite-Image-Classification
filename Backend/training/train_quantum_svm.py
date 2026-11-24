
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras import layers, Model

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


# =====================================================
# 1. PATHS
# =====================================================

BACKEND_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_FOLDER = os.path.join(BACKEND_FOLDER, "models")

os.makedirs(MODEL_FOLDER, exist_ok=True)

tf.get_logger().setLevel("ERROR")


# =====================================================
# 2. LOAD DATASET
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

prepare_dataset = preprocessing.prepare_dataset

X_train, X_test, y_train, y_test, encoder = prepare_dataset()

y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

if y_train.min() == 1 and y_train.max() == 5:
    y_train = y_train - 1
    y_test = y_test - 1

X_train = X_train.reshape(-1, 64, 64, 3)
X_test = X_test.reshape(-1, 64, 64, 3)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)


# =====================================================
# 3. BUILD CNN
# =====================================================

cnn_input = layers.Input(shape=(64, 64, 3))

x = layers.Conv2D(32, (3, 3), activation="relu")(cnn_input)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)

features = layers.Dense(128, activation="relu", name="features")(x)

output = layers.Dense(5, activation="softmax")(features)

cnn = Model(inputs=cnn_input, outputs=output)

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining CNN...")

cnn.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# =====================================================
# 4. EXTRACT CNN FEATURES
# =====================================================

feature_model = Model(
    inputs=cnn_input,
    outputs=features
)

print("\nExtracting CNN features...")

train_features = feature_model.predict(X_train, verbose=0)
test_features = feature_model.predict(X_test, verbose=0)

print("Train features:", train_features.shape)
print("Test features:", test_features.shape)


# =====================================================
# 5. REDUCE FEATURES TO 4
# =====================================================

pca = PCA(n_components=4)

train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)

scaler = MinMaxScaler(feature_range=(-1, 1))

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

print("Features reduced to 4")


# =====================================================
# 6. CREATE QUANTUM CIRCUIT
# =====================================================

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear"
)

parameters = list(feature_map.parameters)

print("\nQuantum circuit created")


# =====================================================
# 7. CREATE QUANTUM STATES
# =====================================================

def create_states(data):

    states = []

    for row in data:

        circuit = feature_map.assign_parameters(
            dict(zip(parameters, row))
        )

        state = Statevector.from_instruction(circuit)

        states.append(state.data)

    return np.array(states)


print("Creating quantum states...")

train_states = create_states(train_features)
test_states = create_states(test_features)


# =====================================================
# 8. CALCULATE QUANTUM KERNEL
# =====================================================

def calculate_kernel(a, b):

    overlap = a @ b.conj().T

    kernel = np.abs(overlap) ** 2

    return kernel


print("\nCalculating quantum kernel...")

train_kernel = calculate_kernel(train_states, train_states)
test_kernel = calculate_kernel(test_states, train_states)


# =====================================================
# 9. TRAIN QUANTUM SVM
# =====================================================

print("\nTraining Quantum SVM...")

qsvm = SVC(kernel="precomputed")

qsvm.fit(train_kernel, y_train)

y_pred = qsvm.predict(test_kernel)


# =====================================================
# 10. EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

class_names = [
    "Forest",
    "Highway",
    "Industrial",
    "PermanentCrop",
    "River"
]

print("\n===================================")
print("QUANTUM SVM RESULTS")
print("===================================")

print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=class_names,
    zero_division=0
))

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3, 4]
))


# =====================================================
# 11. SAVE MODELS
# =====================================================

joblib.dump(qsvm, os.path.join(MODEL_FOLDER, "quantum_svm.pkl"))
joblib.dump(pca, os.path.join(MODEL_FOLDER, "quantum_pca.pkl"))
joblib.dump(scaler, os.path.join(MODEL_FOLDER, "quantum_scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_FOLDER, "quantum_label_encoder.pkl"))

cnn.save(os.path.join(MODEL_FOLDER, "eurosat_cnn.keras"))
feature_model.save(os.path.join(MODEL_FOLDER, "cnn_feature_extractor.keras"))

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")
print("Models saved successfully!")
print("Model folder:", MODEL_FOLDER)
