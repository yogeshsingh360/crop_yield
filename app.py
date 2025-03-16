from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("yield_prediction.pkl")
scaler = joblib.load("scaler.pkl")

# Define route for the home page
@app.route('/')
def index():
    return render_template("index.html", prediction=None, production=None, message=None)

# Define route for prediction
@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='GET':
        return redirect(url_for('index'))
    # Get form data
    data = request.form.to_dict()
    # Extract Area separately
    #area = float(data.pop("Area", 1))  # Default area to 1 if missing

    # Convert input data to DataFrame
    features = pd.DataFrame([data])

    # Predict yield
    yield_pred = model.predict(scaler.transform(features))[0]

    # Calculate production
    #production = yield_pred * area

    return render_template("index.html", prediction=yield_pred, message=None)
    
    
if __name__ == '__main__':
    app.run(debug=True)

