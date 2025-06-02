# ML Diabetes Predictor

This project predicts diabetes using a machine learning model trained on the Pima Indians Diabetes Dataset. It includes data preprocessing, model training using RandomForestClassifier, and deployment using FastAPI for real-time predictions.

## Features

- End-to-end machine learning pipeline
- Cleans missing or zero data values
- Trains a model with Random Forest Classifier
- FastAPI-based REST API to serve predictions
- Swagger UI for testing the API interactively

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/prashantsde11/ml-diabetes-predictor.git
cd ml-diabetes-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
