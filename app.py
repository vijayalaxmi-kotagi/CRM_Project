from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from fpdf import FPDF
import traceback  # For better error handling

app = Flask(__name__)

# File Paths
MODEL_PATH = "models/isolation_forest.pkl"
OUTPUT_CSV = "data/cleaned_data.csv"
REPORT_PDF = "data/report.pdf"

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Function to train and save the Isolation Forest model
def train_model():
    df = pd.read_csv("data/sample_data.csv")
    df.fillna(df.mean(), inplace=True)  # Handle missing values
    df.drop_duplicates(inplace=True)  # Remove duplicates

    # Select only numerical features
    num_df = df.select_dtypes(include=["number"])

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(num_df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    return model

# Load the model if it exists, otherwise train a new one
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Convert all columns to numeric where possible
        df = df.apply(pd.to_numeric, errors='coerce')

        df.fillna(df.mean(), inplace=True)
        df.drop_duplicates(inplace=True)

        # Ensure model feature consistency
        model_features = model.feature_names_in_
        num_df = df[model_features]

        # Predict anomalies
        df["Anomaly"] = model.predict(num_df)

        # ✅ Convert NaN values to None for JSON response
        df.replace({np.nan: None}, inplace=True)

        # ✅ Use new accuracy function
        accuracy = calculate_accuracy(df)

        # Save cleaned data
        df.to_csv(OUTPUT_CSV, index=False)

        return jsonify({
            "data": df.to_dict(orient="records"),
            "accuracy": accuracy
        })

    except Exception as e:
        print("❌ Error processing file:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



@app.route('/download_csv')
def download_csv():
    if not os.path.exists(OUTPUT_CSV):
        return jsonify({"error": "CSV file not found. Please analyze data first."}), 400
    return send_file(OUTPUT_CSV, as_attachment=True)

@app.route('/generate_report')
def generate_report():
    if not os.path.exists(OUTPUT_CSV):
        return jsonify({"error": "CSV file not found. Please analyze data first."}), 400

    df = pd.read_csv(OUTPUT_CSV)
    anomalies = df[df["Anomaly"] == -1].shape[0]
    normal = df[df["Anomaly"] == 1].shape[0]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Data Cleaning & Anomaly Detection Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Total Records Processed: {len(df)}", ln=True)
    pdf.cell(200, 10, f"Anomalies Detected: {anomalies}", ln=True)
    pdf.cell(200, 10, f"Normal Entries: {normal}", ln=True)

    pdf.output(REPORT_PDF)
    return send_file(REPORT_PDF, as_attachment=True)

# Function to calculate accuracy
def calculate_accuracy(df):
    # ✅ Estimate expected anomalies based on contamination rate
    expected_anomalies = int(len(df) * 0.05)  # Assuming 5% contamination

    detected_anomalies = (df["Anomaly"] == -1).sum()
    total_records = len(df)

    # ✅ Compute accuracy based on expected anomalies
    accuracy = (1 - abs(expected_anomalies - detected_anomalies) / total_records) * 100
    return round(accuracy, 2)



if __name__ == '__main__':
    app.run(debug=True)
