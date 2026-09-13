
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
def predict_rental_price_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_prices = model.predict(input_data).tolist()

    # Create a dictionary of predictions with property IDs as keys
    property_ids = input_data['Product_Id_char'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(property_ids, predicted_prices))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_api.run(debug=True)
