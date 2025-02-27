from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('student_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("CSv.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request is JSON (single prediction)
        if request.is_json:
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)  # Ensure 2D array for model
            predictions = model.predict(features)
            return jsonify({'predictions': predictions.tolist()})

        # Otherwise, check for file upload (CSV batch prediction)
        elif 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
            if df.shape[1] != 54:  # Ensure it has exactly 54 columns
                return jsonify({'error': 'CSV file must contain exactly 54 features'}), 400
            
            predictions = model.predict(df.values)
            return jsonify({'predictions': predictions.tolist()})

        # If neither JSON nor CSV, return error
        return jsonify({'error': 'Invalid request format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))
