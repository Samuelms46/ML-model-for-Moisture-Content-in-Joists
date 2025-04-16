from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
        # Handle single prediction (JSON input)
        if request.is_json:
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
            logging.debug(f"Single prediction features: {features.shape}")
            predictions = model.predict(features)
            return jsonify({'predictions': predictions.tolist()})

        # Handle batch prediction (CSV file input)
        elif 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify({'error': 'No file uploaded or invalid file'}), 400
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'File must be a CSV'}), 400

            # Read and validate CSV
            try:
                df = pd.read_csv(file, encoding='utf-8')
                logging.debug(f"CSV shape: {df.shape}")
            except pd.errors.ParserError:
                return jsonify({'error': 'Failed to parse CSV file'}), 400

            # Drop unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            if df.empty:
                return jsonify({'error': 'CSV file is empty'}), 400

            # Convert to numeric and drop invalid rows
            df = df.apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)
            if df.empty:
                return jsonify({'error': 'No valid numeric data after processing'}), 400

            # Validate feature count
            if df.shape[1] != 54:
                return jsonify({'error': f'CSV must contain exactly 54 numeric features. Found {df.shape[1]}'}), 400

            # Batch predictions
            batch_size = 1000
            results = []
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                if batch.empty:
                    continue
                preds = model.predict(batch.values)
                results.extend(preds.tolist())  # Convert ndarray to list
                logging.debug(f"Batch {i//batch_size + 1}: {len(preds)} predictions")

            logging.debug(f"Total predictions: {len(results)}")
            return jsonify({'predictions': results})

        # Invalid request
        return jsonify({'error': 'Invalid request format — must be JSON or file upload'}), 400

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# Run the Flask app
if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))