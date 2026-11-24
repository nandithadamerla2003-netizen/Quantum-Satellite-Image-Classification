
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import tensorflow as tf

from PIL import Image

from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


# =====================================================
# 1. PATHS
# =====================================================

BACKEND_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BACKEND_FOLDER, "models")

tf.get_logger().setLevel("ERROR")

CLASS_NAMES = [
    "Forest",
    "Highway",
    "Industrial",
    "PermanentCrop",
    "River"
]


# =====================================================
# 2. LOAD TRAINED MODELS
# =====================================================

print("Loading trained models...")

classical_svm = joblib.load(
    os.path.join(MODEL_FOLDER, "classical_svm.pkl")
)

quantum_svm = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_svm.pkl")
)

pca = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_pca.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_scaler.pkl")
)

cnn = tf.keras.models.load_model(
    os.path.join(MODEL_FOLDER, "cnn_feature_extractor.keras"),
    compile=False
)

# Saved states of all training images
train_states = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_train_states.pkl")
)

print("All models loaded successfully!")


# =====================================================
# 3. CREATE QUANTUM FEATURE MAP
# =====================================================

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear"
)

parameters = list(feature_map.parameters)


# =====================================================
# 4. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((64, 64))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # CNN input
    cnn_input = np.expand_dims(image_array, axis=0)

    # Classical SVM input
    classical_input = image_array.reshape(1, -1)

    return cnn_input, classical_input


# =====================================================
# 5. CREATE QUANTUM STATE
# =====================================================

def create_quantum_state(features):

    circuit = feature_map.assign_parameters(
        dict(zip(parameters, features[0]))
    )

    state = Statevector.from_instruction(circuit)

    return state.data


# =====================================================
# 6. PREDICTION FUNCTION
# =====================================================

def predict_satellite_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\nProcessing satellite image...")

    # Image preprocessing
    cnn_input, classical_input = preprocess_image(
        image_path
    )

    # ---------------------------------------------
    # CLASSICAL SVM PREDICTION
    # ---------------------------------------------

    classical_prediction = classical_svm.predict(
        classical_input
    )[0]

    # ---------------------------------------------
    # QUANTUM SVM PREDICTION
    # ---------------------------------------------

    cnn_features = cnn.predict(
        cnn_input,
        verbose=0
    )

    reduced_features = pca.transform(cnn_features)

    scaled_features = scaler.transform(reduced_features)

    test_state = create_quantum_state(
        scaled_features
    )

    # Calculate kernel against all training states
    overlap = test_state @ train_states.conj().T

    test_kernel = np.abs(overlap) ** 2

    test_kernel = test_kernel.reshape(1, -1)

    quantum_prediction = quantum_svm.predict(
        test_kernel
    )[0]

    # ---------------------------------------------
    # CLASS NAME
    # ---------------------------------------------

    classical_name = CLASS_NAMES[int(classical_prediction)]

    quantum_name = CLASS_NAMES[int(quantum_prediction)]

    # ---------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------

    print("\n====================================")
    print("SATELLITE IMAGE CLASSIFICATION")
    print("====================================")

    print("Image:", os.path.basename(image_path))

    print("\nClassical SVM Prediction:")
    print(classical_name)

    print("\nQuantum SVM Prediction:")
    print(quantum_name)

    print("\n====================================")

    return {
        "classical_prediction": classical_name,
        "quantum_prediction": quantum_name
    }


# =====================================================
# 7. TEST WITH ONE IMAGE
# =====================================================

if __name__ == "__main__":

    image_path = input(
        "\nEnter satellite image path: "
    ).strip().strip('"')

    result = predict_satellite_image(image_path)

    print("\nFinal Result:")
    print(result)
