from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables to hold the model and column information
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    """
    Predicts the price based on the loaded model.
    """
    try:
        # Find the index for the location column
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        # If location is not found, its index will be -1
        loc_index = -1

    # Create the input array for the model
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict and round the result
    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    """
    Loads the saved model and column information from files.
    """
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Load column information
    try:
        with open("./columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]  # First 3 are sqft, bath, bhk
    except FileNotFoundError:
        print("Error: 'columns.json' not found.")
        return

    # Load the trained model
    try:
        # Ensure this file is named correctly and is in the same directory
        with open("./bangalore_home_prices_model.pickle", 'rb') as f:
            __model = pickle.load(f)
    except FileNotFoundError:
        print("Error: 'bangalore_home_prices_model.pickle' not found.")
        return
        
    print("Loading saved artifacts...done")

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """
    Returns the list of available locations.
    """
    if __locations is None:
        return jsonify({'error': 'Locations not loaded'}), 500
        
    response = jsonify({
        'locations': __locations
    })
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """
    Handles the prediction request from the front-end.
    """
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    total_sqft = float(data['total_sqft'])
    location = data['location']
    bhk = int(data['bhk'])
    bath = int(data['bath'])

    estimated_price = get_estimated_price(location, total_sqft, bhk, bath)

    response = jsonify({
        'estimated_price': estimated_price
    })
    
    return response

if __name__ == '__main__':
    print("Starting Python Flask Server For Home Price Prediction...")
    load_saved_artifacts()
    # To run this server, use the command: flask run
    # Make sure you have Flask and Flask-Cors installed:
    # pip install Flask flask-cors scikit-learn numpy
    app.run()
