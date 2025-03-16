from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
import gdown
model_path="Deployment/yield_prediction.pkl"
drive_file_id="1Nt18E84oojN5DwyAKn3BovzcFNVByHXH"
def download_model():
    if not os.path.exists(model_path):
        gdown.download(drive_file_id, model_path, quiet=False)
def load_model():
    return joblib.load(model_path)
download_model()



app = Flask(__name__)

# Load the trained model and scaler
model = load_model()
scaler = joblib.load("scaler.pkl")

# Define route for the home page
@app.route('/')
def index():
    return render_template("index.html", prediction=None, production=None, message=None)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Check if any value is missing
        if not all(data.values()):
            return render_template("index.html", prediction=None, production=None, message="⚠ Fill all values!")

        # Extract Area separately
        area = float(data.pop("Area", 1))  # Default area to 1 if missing

        # Convert input data to DataFrame
        features = pd.DataFrame([data])

        # Predict yield
        yield_pred = model.predict(scaler.transform(features))[0]

        # Calculate production
        production = yield_pred * area

        return render_template("index.html", prediction=yield_pred, production=production, message=None)
    
    except Exception as e:
        return render_template("index.html", prediction=None, production=None, message="⚠ Error: Please check input values!")

if __name__ == '__main__':
    app.run(debug=True)

