# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return jsonify({'predicted_tuition': prediction})

if __name__ == '__main__':
    app.run(debug=True)