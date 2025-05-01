from data_fetcher import fetch_upcoming_race_data, fetch_current_season_standings
from model_trainer import load_model, preprocess_prediction_data
import numpy as np
import pandas as pd
import pickle

def predict_race():
    # Fetch race and standings data
    print("Fetching race data...")
    upcoming_race = fetch_upcoming_race_data()
    current_standings = fetch_current_season_standings()
    
    # Debug: Print the structure of the data
    print("\nUpcoming Race Data:")
    print(upcoming_race)
    print("\nCurrent Standings Data:")
    print(current_standings)
    
    # Load model and make predictions
    print("Loading model and making predictions...")
    model = load_model()
    X_pred = preprocess_prediction_data(upcoming_race, current_standings)
    predictions = model.predict_proba(X_pred)
    
    # Load driver mappings
    with open('models/driver_mapping.pkl', 'rb') as f:
        driver_mapping = pickle.load(f)
    with open('models/driver_labels.pkl', 'rb') as f:
        driver_labels = pickle.load(f)
    
    # Get top 3 predictions
    top_3_indices = np.argsort(-predictions, axis=1)[0][:3]
    
    print(f"\nPredicted Top 3 for {upcoming_race['circuit']} GP:")
    for i, idx in enumerate(top_3_indices, 1):
        driver_id = driver_labels[idx]
        probability = predictions[0][idx] * 100
        print(f"{i}. {driver_mapping[driver_id]} ({probability:.1f}%)")

if __name__ == "__main__":
    predict_race()

