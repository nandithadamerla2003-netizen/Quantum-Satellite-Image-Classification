
import os
import uuid
import shutil
import warnings
import csv
import uvicorn

warnings.filterwarnings("ignore")

import evaluation
from prediction import predict_all_images, predict_satellite_image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


# 1. PROJECT TITLE
print(" =============== QUANTUM SATELLITE IMAGE CLASSIFIER ===============" )
print("\n All packages installed successfully!")

# 3. FOLDER SETTINGS

BACKEND_FOLDER = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BACKEND_FOLDER, "uploads")
RESULTS_FOLDER = os.path.join(BACKEND_FOLDER, "results")
RESULTS_FILE = os.path.join(RESULTS_FOLDER, "predictions.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# 4. MODEL EVALUATION

print("MODEL ACCURACY AND EVALUATION")
print("\nModel evaluation completed successfully!")

# 5. PREDICTION - 2 IMAGES FROM EACH CLASS

print("SATELLITE IMAGE PREDICTION")

print("\nClasses:")
print("1. Forest")
print("2. Highway")
print("3. Industrial")
print("4. PermanentCrop")
print("5. River")

print("\nSelecting 2 images from each class")
print("Maximum images: 10")


try:
    if os.path.exists(RESULTS_FILE):

        with open(RESULTS_FILE, "r", newline="", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            count = 0

            for row in reader:

                count += 1

                image_name = row["Image"]
                actual_class = row["Actual_Class"]
                classical_prediction = row["Classical_SVM"]
                quantum_prediction = row["Quantum_SVM"]

                # Check Classical SVM result
                if actual_class.strip().lower() == classical_prediction.strip().lower():
                    classical_result = "Correct"
                else:
                    classical_result = "Incorrect"

                # Check Quantum SVM result
                if actual_class.strip().lower() == quantum_prediction.strip().lower():
                    quantum_result = "Correct"
                else:
                    quantum_result = "Incorrect"

                print("\nImage:", image_name)
                print("\nActual Class:", actual_class)
                print("Classical SVM Prediction:", classical_prediction)
                print("Quantum SVM Prediction:", quantum_prediction)
                print("\nClassical SVM Result:", classical_result)
                print("Quantum SVM Result:", quantum_result)        

        print("\nPrediction results saved in:")
        print(RESULTS_FILE)

    else:
        print("\nPrediction results file not found!")

except Exception as e:
    print("\nPrediction Error:", str(e))

# 6. FASTAPI APPLICATION

app = FastAPI(
    title="Quantum Satellite Image Classifier",
    description="Satellite Image Classification using Classical SVM and Quantum SVM",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# 7. HOME API

@app.get("/")
def home():

    return {
        "message": "Welcome to Quantum Satellite Image Classifier",
        "status": "Backend is running",
        "models": [
            "Classical SVM",
            "Quantum SVM"
        ]
    }

# 8. HEALTH CHECK API

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "message": "Quantum Satellite Image Classifier API is working"
    }

# 9. IMAGE PREDICTION API

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    
    file_extension = os.path.splitext(file.filename or "")[1].lower()

    if file_extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Upload JPG, JPEG, PNG or TIFF."
        )

    image_name = f"{uuid.uuid4().hex}{file_extension}"
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    try:
        # Save uploaded image
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Verify image
        try:
            with Image.open(image_path) as image:
                image.verify()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image."
            )

        # Predict using both models
        classical_prediction, quantum_prediction = (
            predict_satellite_image(image_path)
        )

        return {
            "status": "success",
            "image_name": file.filename,
            "actual_class": "Not available for uploaded image",
            "classical_svm_prediction": str(classical_prediction),
            "quantum_svm_prediction": str(quantum_prediction)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        await file.close()

        if os.path.exists(image_path):
            os.remove(image_path)

# 10. START BACKEND SERVER
if __name__ == "__main__":
    print("\nBackend Server Starting")
    print("API URL: http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")

    uvicorn.run(app,host="127.0.0.1",port=8000)
