import os
import sys
import time
import joblib

from sklearn.metrics import accuracy_score, classification_report

# Add Backend folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.image_preprocessing import prepare_dataset
from classical_svm import train_classical_svm


# Start time
start_time = time.time()

print("Loading Preprocessed Dataset...")

X_train, X_test, y_train, y_test, encoder = prepare_dataset()

# Train model
model = train_classical_svm(X_train, y_train)

# Predict test images
print("\nPredicting Test Images...")

y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nClassical SVM Results")
print("-------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# Create models folder
model_path = "../models"

os.makedirs(model_path, exist_ok=True)

# Save model and encoder
joblib.dump(model, os.path.join(model_path, "classical_svm.pkl"))

joblib.dump(encoder, os.path.join(model_path, "label_encoder.pkl"))

# Execution time
end_time = time.time()

print("\nExecution Time:", round(end_time - start_time, 2), "seconds")

print("\nClassical SVM Model Saved Successfully!")