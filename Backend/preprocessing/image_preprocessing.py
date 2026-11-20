
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Dataset path
DATASET_PATH = "RawData/Quantum Satellite Image Classifier"

# Image size
IMAGE_SIZE = (64, 64)


def load_dataset():

    X = []
    y = []

    classes = sorted(os.listdir(DATASET_PATH))

    print("Loading EuroSAT dataset...")

    for class_name in classes:

        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(class_path):
            continue

        print("Loading class:", class_name)

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            try:

                # Load image
                image = Image.open(image_path)

                # Convert to RGB
                image = image.convert("RGB")

                # Resize image
                image = image.resize(IMAGE_SIZE)

                # Convert image to NumPy array
                image_array = np.array(image, dtype=np.float32)

                # Normalize pixel values
                image_array = image_array / 255.0

                # Store image
                X.append(image_array)

                # Store class label
                y.append(class_name)

            except Exception as e:

                print("Error loading:", image_path, e)


    # Convert lists into arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("\nDataset Loaded Successfully")
    print("Images Shape:", X.shape)
    print("Labels Shape:", y.shape)

    return X, y


def prepare_dataset():

    # Load images
    X, y = load_dataset()

    # Encode class labels
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    # Flatten images
    X_flat = X.reshape(X.shape[0], -1)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining Data:", X_train.shape)
    print("Testing Data:", X_test.shape)

    print("\nClass Labels:")
    for i, class_name in enumerate(encoder.classes_):
        print(i, ":", class_name)

    return X_train, X_test, y_train, y_test, encoder


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, encoder = prepare_dataset()
