
import os
import warnings
import csv

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import tensorflow as tf

from PIL import Image
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


# ==========================================================
# 1. FOLDER PATHS
# ==========================================================

BACKEND_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(BACKEND_FOLDER)

MODEL_FOLDER = os.path.join(BACKEND_FOLDER, "models")
RESULT_FOLDER = os.path.join(BACKEND_FOLDER, "results")

os.makedirs(RESULT_FOLDER, exist_ok=True)

CLASS_NAMES = [
    "Forest",
    "Highway",
    "Industrial",
    "PermanentCrop",
    "River"
]

# Check possible dataset locations
POSSIBLE_DATA_FOLDERS = [
    os.path.join(
        PROJECT_FOLDER,
        "RawData",
        "Quantum Satellite Image Classifier"
    ),
    os.path.join(
        BACKEND_FOLDER,
        "RawData",
        "Quantum Satellite Image Classifier"
    ),
    os.path.join(PROJECT_FOLDER, "RawData")
]

DATA_FOLDER = None

# Select the folder containing the five class folders
for folder in POSSIBLE_DATA_FOLDERS:
    if not os.path.isdir(folder):
        continue

    # Check whether class folders are directly inside this folder
    if all(
        os.path.isdir(os.path.join(folder, class_name))
        for class_name in CLASS_NAMES
    ):
        DATA_FOLDER = folder
        break

    # Check for the dataset folder inside RawData
    nested_folder = os.path.join(
        folder,
        "Quantum Satellite Image Classifier"
    )

    if all(
        os.path.isdir(os.path.join(nested_folder, class_name))
        for class_name in CLASS_NAMES
    ):
        DATA_FOLDER = nested_folder
        break


# ==========================================================
# 2. LOAD TRAINED MODELS
# ==========================================================

print("Loading trained models...")

classical_svm = joblib.load(
    os.path.join(MODEL_FOLDER, "classical_svm.pkl")
)

quantum_svm = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_svm.pkl")
)

quantum_pca = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_pca.pkl")
)

quantum_scaler = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_scaler.pkl")
)

quantum_train_states = joblib.load(
    os.path.join(MODEL_FOLDER, "quantum_train_states.pkl")
)

cnn_model = tf.keras.models.load_model(
    os.path.join(MODEL_FOLDER, "cnn_feature_extractor.keras"),
    compile=False
)

quantum_train_states = np.asarray(quantum_train_states)

print("All models loaded successfully!")


# ==========================================================
# 3. QUANTUM FEATURE MAP
# ==========================================================

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1,
    entanglement="linear"
)


# ==========================================================
# 4. FIND ONLY 2 IMAGES PER CLASS
# ==========================================================

def find_images(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff"
    )

    image_paths = []

    if not folder or not os.path.isdir(folder):
        raise FileNotFoundError(
            "Dataset folder not found. Check your RawData location."
        )

    print("\nSelecting only 2 images from each class:\n")

    for class_name in CLASS_NAMES:

        class_folder = os.path.join(folder, class_name)

        if not os.path.isdir(class_folder):
            print(f"{class_name}: Folder not found")
            continue

        class_images = []

        # Search inside the class folder and its subfolders
        for root, dirs, files in os.walk(class_folder):

            dirs.sort()

            for file in sorted(files):

                if file.lower().endswith(extensions):

                    class_images.append(
                        os.path.join(root, file)
                    )

        # IMPORTANT: Select only 2 images
        selected_images = class_images[:2]

        image_paths.extend(selected_images)

        print(
            f"{class_name}: {len(selected_images)} images selected"
        )

    print("\nTotal selected images:", len(image_paths))

    return image_paths


# ==========================================================
# 5. PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((64, 64))

    image_array = np.array(image, dtype=np.float32) / 255.0

    # Input for CNN
    cnn_input = np.expand_dims(image_array, axis=0)

    # Input for Classical SVM
    classical_input = image_array.reshape(1, -1)

    return cnn_input, classical_input


# ==========================================================
# 6. CREATE QUANTUM STATE
# ==========================================================

def create_quantum_state(features):

    circuit = feature_map.assign_parameters(
        features,
        inplace=False
    )

    state = Statevector.from_instruction(circuit)

    return state.data


# ==========================================================
# 7. PREDICT ONE IMAGE
# ==========================================================

def predict_satellite_image(image_path):

    cnn_input, classical_input = preprocess_image(image_path)

    # ---------------- Classical SVM ----------------

    classical_prediction = classical_svm.predict(
        classical_input
    )[0]

    # ---------------- CNN Feature Extraction ----------------

    cnn_features = cnn_model.predict(
        cnn_input,
        verbose=0
    )

    # ---------------- PCA ----------------

    pca_features = quantum_pca.transform(
        cnn_features
    )

    # ---------------- Scaling ----------------

    scaled_features = quantum_scaler.transform(
        pca_features
    )[0]

    # ---------------- Quantum State ----------------

    test_state = create_quantum_state(
        scaled_features
    )

    # ---------------- Quantum Fidelity Kernel ----------------

    quantum_kernel = np.abs(
        test_state @ quantum_train_states.conj().T
    ) ** 2

    quantum_kernel = quantum_kernel.reshape(1, -1)

    # ---------------- Quantum SVM ----------------

    quantum_prediction = quantum_svm.predict(
        quantum_kernel
    )[0]

    # Convert numeric predictions to class names
    if isinstance(classical_prediction, (int, np.integer)):
        classical_prediction = CLASS_NAMES[int(classical_prediction)]

    if isinstance(quantum_prediction, (int, np.integer)):
        quantum_prediction = CLASS_NAMES[int(quantum_prediction)]

    return classical_prediction, quantum_prediction


# ==========================================================
# 8. GET ACTUAL CLASS NAME
# ==========================================================

def get_actual_class(image_path):

    relative_path = os.path.relpath(
        image_path,
        DATA_FOLDER
    )

    path_parts = relative_path.split(os.sep)

    for part in path_parts:

        if part in CLASS_NAMES:
            return part

    return "Unknown"


# ==========================================================
# 9. PREDICT ONLY SELECTED 10 IMAGES
# ==========================================================

def predict_all_images():

    if DATA_FOLDER is None:
        print("\nDataset folder not found!")
        print("Please check your RawData folder location.")
        return

    print("\nDataset folder:", DATA_FOLDER)

    # This function selects maximum 2 images per class
    image_paths = find_images(DATA_FOLDER)

    if not image_paths:
        print("No images found.")
        return

    csv_file = os.path.join(
        RESULT_FOLDER,
        "predictions.csv"
    )

    successful = 0

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Image",
            "Actual_Class",
            "Classical_SVM",
            "Quantum_SVM",
            "Status"
        ])

        # Predict ONLY the selected images
        for image_path in image_paths:

            try:

                actual_class = get_actual_class(
                    image_path
                )

                classical_prediction, quantum_prediction = (
                    predict_satellite_image(image_path)
                )

                status = (
                    "Correct"
                    if quantum_prediction == actual_class
                    else "Incorrect"
                )

                print("\n----------------------------------")
                print("Image:", os.path.basename(image_path))
                print("Actual Class:", actual_class)
                print("Classical SVM:", classical_prediction)
                print("Quantum SVM:", quantum_prediction)
                print("Status:", status)

                writer.writerow([
                    os.path.basename(image_path),
                    actual_class,
                    classical_prediction,
                    quantum_prediction,
                    status
                ])

                successful += 1

            except Exception as e:

                print("\nError processing:", image_path)
                print("Error:", e)

    print("\n==================================")
    print("PREDICTION COMPLETED")
    print("==================================")

    print("Total selected images:", len(image_paths))
    print("Successfully predicted:", successful)

    print("\nResults saved at:")
    print(csv_file)


# ==========================================================
# 10. MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    predict_all_images()
