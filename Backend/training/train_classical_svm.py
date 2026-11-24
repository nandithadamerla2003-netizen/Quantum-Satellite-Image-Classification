
import os
import sys
import joblib

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. BACKEND PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add Backend directory to Python path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# --------------------------------------------------
# 2. IMPORT PREPROCESSING
# --------------------------------------------------

from preprocessing.image_preprocessing import prepare_dataset


# --------------------------------------------------
# 3. MODEL SAVE DIRECTORY
# --------------------------------------------------

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

print("=" * 50)
print("CLASSICAL SVM - SATELLITE IMAGE CLASSIFICATION")
print("=" * 50)

print("\nLoading Dataset...")

X_train, X_test, y_train, y_test, encoder = prepare_dataset()

print("\nDataset Loaded Successfully!")

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))


# --------------------------------------------------
# 5. CREATE CLASSICAL SVM MODEL
# --------------------------------------------------

print("\nCreating Classical SVM Model...")

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)


# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Classical SVM...")

model.fit(X_train, y_train)

print("\nTraining Completed Successfully!")


# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

print("\nPredicting Test Images...")

y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("CLASSICAL SVM RESULTS")
print("=" * 50)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 9. SAVE TRAINED MODEL
# --------------------------------------------------

model_path = os.path.join(MODEL_DIR, "classical_svm.pkl")

encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

print("\nSaving Model...")

joblib.dump(model, model_path)

joblib.dump(encoder, encoder_path)

print("\nModel Saved Successfully!")

print("Model Path:", model_path)

print("Encoder Path:", encoder_path)


# --------------------------------------------------
# 10. FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 50)
print("CLASSICAL SVM TRAINING FINISHED")
print("=" * 50)
