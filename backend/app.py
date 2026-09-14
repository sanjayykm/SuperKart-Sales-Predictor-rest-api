
import pandas as pd
import joblib
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
model = joblib.load("super_kart_model_v1_0.joblib")

# Initialize Flask app
superkart_api = Flask("Superkart Predictor")

@superkart_api.route('/')
def home():
    return 'Flask server is running!'

@superkart_api.post('/v1/predict')
def predict():
    """
      This function handles POST requests to the '/v1/rental' endpoint.
      It expects a JSON payload containing property details and returns
      the predicted rental price as a JSON response.
    """
    if not request.is_json:
        logging.info("Error: Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)
    logging.info(f"Received data: {data}") # Log incoming data

    # Expected order of columns for the model
    expected_columns = [
        "Product_Weight",
        "Product_Sugar_Content",
        "Product_Type",
        "Product_MRP",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type",
        "Store_Age"
    ]

    try:
        # Convert incoming JSON data to a pandas DataFrame
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([data])

        # Reorder columns to match the expected order during training
        # This is crucial because ColumnTransformer expects columns in a specific order
        input_df = input_df[expected_columns]

        # Make prediction
        prediction = model.predict(input_df)
        logging.info(f"Prediction: {prediction[0]}") # Log prediction

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        logging.info(f"Prediction error: {str(e)}") # Log error
        return jsonify({"error": str(e)}), 500

# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_batch():
    """
    Handles POST requests to '/v1/predictbatch'.
    Expects a CSV file (multipart form field 'file') containing product +
    store details, and returns predicted sales totals as JSON.
    """
    EXPECTED_COLUMNS = [
        "Product_Weight", "Product_Sugar_Content", "Product_Type",
        "Product_MRP", "Store_Size", "Store_Location_City_Type",
        "Store_Type", "Store_Age",
    ]

    if 'file' not in request.files:
        return jsonify({"error": "No file part named 'file' in the request"}), 400

    try:
        input_data = pd.read_csv(request.files['file'])
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    if input_data.empty:
        return jsonify({"error": "Uploaded CSV contains no rows"}), 400

    missing = [c for c in EXPECTED_COLUMNS if c not in input_data.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    try:
        features = input_data[EXPECTED_COLUMNS].copy()

        # match the normalization applied during training
        features["Product_Sugar_Content"] = (
            features["Product_Sugar_Content"].str.strip().str.lower()
        )

        predictions = model.predict(features).tolist()
        logging.info(f"Batch prediction: {len(predictions)} rows")

        return jsonify({
            "n_rows": len(predictions),
            "predictions": [
                {"row": i, "predicted_sales_total": round(p, 2)}
                for i, p in enumerate(predictions)
            ],
        }), 200

    except Exception as e:
        logging.exception("Batch prediction failed")
        return jsonify({"error": str(e)}), 500

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_api.run(debug=True)
