from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import json
import pickle
import numpy as np
from data_fetcher import fetch_upcoming_race_data, fetch_current_season_standings
from model_trainer import load_model, preprocess_prediction_data

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    # Get the next race information
    try:
        upcoming_race = fetch_upcoming_race_data()
        current_standings = fetch_current_season_standings()
        return render_template('index.html', 
                               upcoming_race=upcoming_race, 
                               current_standings=current_standings,
                               prediction=None,
                               error=None)
    except Exception as e:
        return render_template('index.html', 
                               upcoming_race=None, 
                               current_standings=None,
                               prediction=None,
                               error=f"Error fetching race data: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model
        model = load_model()
        
        # Get upcoming race data
        upcoming_race = fetch_upcoming_race_data()
        current_standings = fetch_current_season_standings()
        
        # Prepare data for prediction
        X_pred = preprocess_prediction_data(upcoming_race, current_standings)
        
        # Make predictions
        predictions = model.predict_proba(X_pred)
        driver_probabilities = []
        
        # Get driver names from the model
        with open('models/driver_labels.pkl', 'rb') as f:
            driver_labels = pickle.load(f)
        
        # Convert prediction probabilities to a list of driver probabilities
        for i, driver in enumerate(driver_labels):
            driver_probabilities.append({
                'driver': driver,
                'probability': float(np.max(predictions[predictions[:, i] > 0, i]))
            })
        
        # Sort by probability descending
        driver_probabilities = sorted(driver_probabilities, 
                                     key=lambda x: x['probability'], 
                                     reverse=True)
        
        return render_template('index.html', 
                              upcoming_race=upcoming_race, 
                              current_standings=current_standings,
                              prediction=driver_probabilities,
                              error=None)
    
    except Exception as e:
        return render_template('index.html', 
                              upcoming_race=None, 
                              current_standings=None,
                              prediction=None,
                              error=f"Error making prediction: {str(e)}")

@app.route('/api/predict', methods=['GET'])
def api_predict():
    try:
        # Load model
        model = load_model()
        
        # Get upcoming race data
        upcoming_race = fetch_upcoming_race_data()
        current_standings = fetch_current_season_standings()
        
        # Prepare data for prediction
        X_pred = preprocess_prediction_data(upcoming_race, current_standings)
        
        # Make predictions
        predictions = model.predict_proba(X_pred)
        driver_probabilities = []
        
        # Get driver names from the model
        with open('models/driver_labels.pkl', 'rb') as f:
            driver_labels = pickle.load(f)
        
        # Convert prediction probabilities to a list of driver probabilities
        for i, driver in enumerate(driver_labels):
            driver_probabilities.append({
                'driver': driver,
                'probability': float(np.max(predictions[predictions[:, i] > 0, i]))
            })
        
        # Sort by probability descending
        driver_probabilities = sorted(driver_probabilities, 
                                     key=lambda x: x['probability'], 
                                     reverse=True)
        
        return jsonify({
            'upcoming_race': upcoming_race,
            'predictions': driver_probabilities
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
