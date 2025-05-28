from fastapi import FastAPI                  # Import FastAPI framework to create API
from pydantic import BaseModel              # Import BaseModel for data validation
import joblib                               # Import joblib to load the saved ML model
import pandas as pd                         # Import pandas to create DataFrame

# Load trained model from the file 'diabetes_model.pkl'
model = joblib.load("diabetes_model.pkl")

# Define input data schema using Pydantic BaseModel
# This ensures incoming JSON has the required fields with correct types
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Initialize FastAPI app instance
app = FastAPI()

# Define POST API endpoint '/predict'
# This endpoint receives patient data and returns diabetes prediction
@app.post("/predict")
def predict(data: PatientData):
    # Convert incoming data to dictionary, then wrap into a list to create a DataFrame with one row
    # This DataFrame has the same feature names as used in model training
    input_df = pd.DataFrame([data.dict()])

    # Use the loaded model to predict probabilities for each class (0 or 1)
    # [0][1] means: first sample's probability of class '1' (diabetes positive)
    probability = model.predict_proba(input_df)[0][1]

    # Decide predicted class based on threshold 0.5:
    # If probability >= 0.5, classify as 1 (diabetes positive), else 0 (negative)
    prediction = int(probability >= 0.5)

    # Print input data, predicted probability, and predicted label for debugging/logging
    print(f"\nInput: {input_df.to_dict(orient='records')[0]}")
    print(f"Probability of diabetes: {probability:.4f}")
    print(f"Predicted label: {prediction}")

    # Return prediction result as JSON with boolean Diabetes field
    # True if prediction == 1, False if 0
    return {"Diabetes": bool(prediction)}
