from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename
from model import train_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Global variables to store model information
model = None
scaler = None
feature_columns = None
target_column = 'Class'  # Default target column name

# Try to load pre-trained model if available
try:
    model = joblib.load('fraud_detector.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    print("Pre-trained model loaded successfully")
except Exception as e:
    print(f"No pre-trained model found: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "Empty filename.", 400

    # Get target column from form
    global target_column
    target_column = request.form.get('target_column', 'Class')

    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        # Train the model
        global model, scaler, feature_columns
        model, scaler, feature_columns = train_model(
            filepath, 
            target_column=target_column,
            n_estimators=int(request.form.get('n_estimators', 100))
        )

        return render_template('train_success.html')

    except Exception as e:
        return f"Error training model: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "Empty filename.", 400

    try:
        # Check if model is loaded
        global model, scaler, feature_columns
        if model is None or scaler is None or feature_columns is None:
            return "Model not trained yet. Please train a model first.", 400

        # Read uploaded CSV file
        data = pd.read_csv(file)
        original_data = data.copy()

        # Check for required columns
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            return f"Error: Missing required columns: {', '.join(missing_columns)}", 400

        # Prepare features using the exact columns from training
        X = data[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Add predictions to original data
        original_data['Prediction'] = predictions
        
        # Extract fraudulent transactions
        fraud_cases = original_data[original_data['Prediction'] == 1]

        return render_template(
            'result.html',
            total=len(predictions),
            fraud_count=len(fraud_cases),
            fraud_cases=fraud_cases.to_html(classes='table table-striped table-bordered', index=False)
        )

    except Exception as e:
        return f"Error processing file: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
