import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime
import joblib

from data_fetcher import (
    fetch_race_results,
    fetch_qualifying_results,
    fetch_driver_standings,
    fetch_upcoming_race_data,
    fetch_current_season_standings
)

# Directory to save model files
MODEL_DIR = 'models'

def ensure_model_dir():
    """Ensure model directory exists"""
    os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_historical_data(years=None, circuits=None):
    """
    Fetch historical F1 data for model training.
    
    Parameters:
    - years: List of years to fetch data for. If None, uses last 5 years.
    - circuits: List of circuit IDs to filter for. If None, fetches all circuits.
    
    Returns:
    - Dictionary containing dataframes for race results, qualifying, and standings
    """
    if years is None:
        current_year = datetime.now().year
        years = list(range(current_year - 5, current_year))
    
    # Initialize empty dataframes
    all_race_results = pd.DataFrame()
    all_qualifying_results = pd.DataFrame()
    all_driver_standings = pd.DataFrame()
    
    for year in years:
        print(f"Fetching data for {year}...")
        
        # Race results
        race_results = fetch_race_results(year)
        all_race_results = pd.concat([all_race_results, race_results])
        
        # Qualifying results
        qualifying_results = fetch_qualifying_results(year)
        all_qualifying_results = pd.concat([all_qualifying_results, qualifying_results])
        
        # Driver standings (get final standings)
        driver_standings = fetch_driver_standings(year)
        all_driver_standings = pd.concat([all_driver_standings, driver_standings])
    
    # Filter by circuit if necessary
    if circuits:
        circuit_filter = all_race_results['circuit_id'].isin(circuits)
        all_race_results = all_race_results[circuit_filter]
        
        # Get the corresponding rounds and filter qualifying
        rounds = all_race_results[['year', 'round']].drop_duplicates()
        qualifying_filter = pd.merge(all_qualifying_results, rounds, on=['year', 'round'])
        all_qualifying_results = qualifying_filter
    
    return {
        'race_results': all_race_results,
        'qualifying_results': all_qualifying_results,
        'driver_standings': all_driver_standings
    }

def engineer_features(data):
    """
    Engineer features from the raw data for model training.
    
    Parameters:
    - data: Dictionary containing dataframes from fetch_historical_data()
    
    Returns:
    - X: Feature matrix
    - y: Target vector (race winner)
    - feature_names: List of feature names
    - driver_names: List of driver names (for the target labels)
    """
    # First fetch current drivers to filter data
    try:
        current_standings = fetch_current_season_standings()
        if 'drivers' in current_standings:
            current_driver_ids = [driver['driver_id'] for driver in current_standings['drivers']]
            print(f"Using {len(current_driver_ids)} current drivers from {current_standings['year']} season")
        else:
            # If we couldn't get current standings, we'll use recent drivers
            print("Couldn't get current driver list, will use recent drivers only")
            current_driver_ids = None
    except Exception as e:
        print(f"Error fetching current drivers: {e}")
        current_driver_ids = None
    race_results = data['race_results']
    qualifying_results = data['qualifying_results']
    driver_standings = data['driver_standings']
    
    # Merge qualifying data with race results
    race_qual = pd.merge(
        race_results,
        qualifying_results,
        on=['year', 'round', 'driver_id'],
        suffixes=('', '_qual')
    )
    
    # Merge with driver standings before each race
    # We need to adjust the round to get standings before the race
    race_qual['prev_round'] = (race_qual['round'].astype(int) - 1).astype(str)
    race_qual['prev_round'] = race_qual['prev_round'].str.replace('-1', '1')  # Handle first race
    
    # Now merge with driver standings
    race_qual['year'] = race_qual['year'].astype(int)
    driver_standings['year'] = driver_standings['year'].astype(int)
    
    race_data = pd.merge(
        race_qual,
        driver_standings,
        left_on=['year', 'prev_round', 'driver_id'],
        right_on=['year', 'round', 'driver_id'],
        suffixes=('', '_standings'),
        how='left'
    )
    
    # Fill missing standings data (e.g., first race of season)
    race_data['position_standings'] = race_data['position_standings'].fillna(20)
    race_data['points_standings'] = race_data['points_standings'].fillna(0)
    race_data['wins'] = race_data['wins'].fillna(0)
    
    # Calculate driver historical performance at each circuit
    driver_circuit_stats = race_data.groupby(['driver_id', 'circuit_name']).agg({
        'position': ['mean', 'min']
    }).reset_index()
    driver_circuit_stats.columns = ['driver_id', 'circuit_name', 'avg_position_at_circuit', 'best_position_at_circuit']
    
    # Calculate team historical performance at each circuit
    team_circuit_stats = race_data.groupby(['constructor', 'circuit_name']).agg({
        'position': ['mean', 'min']
    }).reset_index()
    team_circuit_stats.columns = ['constructor', 'circuit_name', 'team_avg_position_at_circuit', 'team_best_position_at_circuit']
    
    # Calculate overall grid-to-finish correlation
    grid_finish_data = race_data.groupby('driver_id').apply(
        lambda x: x['position'].corr(x['qualifying_position'])
    ).reset_index()
    grid_finish_data.columns = ['driver_id', 'grid_finish_correlation']
    
    # Add driver's recent form (average position in last 3 races)
    race_data = race_data.sort_values(['driver_id', 'year', 'round'])
    race_data['recent_form'] = race_data.groupby('driver_id')['position'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Merge back the new features
    race_data = race_data.merge(driver_circuit_stats, on=['driver_id', 'circuit_name'], how='left')
    race_data = race_data.merge(team_circuit_stats, on=['constructor', 'circuit_name'], how='left')
    race_data = race_data.merge(grid_finish_data, on='driver_id', how='left')
    
    # Fill NA values for new columns
    race_data['avg_position_at_circuit'] = race_data['avg_position_at_circuit'].fillna(race_data['position_standings'])
    race_data['best_position_at_circuit'] = race_data['best_position_at_circuit'].fillna(20)
    race_data['team_avg_position_at_circuit'] = race_data['team_avg_position_at_circuit'].fillna(10)
    race_data['team_best_position_at_circuit'] = race_data['team_best_position_at_circuit'].fillna(10)
    race_data['grid_finish_correlation'] = race_data['grid_finish_correlation'].fillna(0)
    
    # Create target: 1 if driver won, 0 otherwise
    race_data['won_race'] = (race_data['position'] == 1).astype(int)
    
    # Features to use
    feature_cols = [
        'qualifying_position',  # Starting position
        'position_standings',   # Driver championship position before race
        'points_standings',     # Driver points before race
        'wins',       # Driver wins before race
        'avg_position_at_circuit',  # Driver's average position at this circuit
        'best_position_at_circuit',  # Driver's best position at this circuit
        'team_avg_position_at_circuit',  # Team's average position at this circuit
        'team_best_position_at_circuit',  # Team's best position at this circuit
        'grid_finish_correlation',  # Correlation between qualifying and finishing positions
        'recent_form',  # Driver's recent form (average position in last 3 races)
    ]
    
    # Add constructor as a categorical feature
    constructor_dummies = pd.get_dummies(race_data['constructor'], prefix='constructor')
    race_data = pd.concat([race_data, constructor_dummies], axis=1)
    
    # Add circuit as a categorical feature
    circuit_dummies = pd.get_dummies(race_data['circuit_name'], prefix='circuit')
    race_data = pd.concat([race_data, circuit_dummies], axis=1)
    
    # Get all feature columns including the dummies
    all_feature_cols = feature_cols + list(constructor_dummies.columns) + list(circuit_dummies.columns)
    
    # Prepare final dataset for modeling
    X = race_data[all_feature_cols].copy()
    y = race_data['driver_id']  # Target is driver ID (who won)
    
    # Filter to include only current drivers if we have that list
    if current_driver_ids:
        print(f"Filtering data to include only current drivers...")
        
        # Keep only races with current drivers
        race_data_filtered = race_data[race_data['driver_id'].isin(current_driver_ids)]
        
        # If we have enough data, use the filtered dataset
        if len(race_data_filtered) > 100:  # Minimum threshold for sufficient data
            print(f"Using {len(race_data_filtered)} races with current drivers out of {len(race_data)} total races")
            X = race_data_filtered[all_feature_cols].copy()
            y = race_data_filtered['driver_id']
        else:
            print(f"Not enough data for current drivers only ({len(race_data_filtered)} races). Using all recent data.")
    
    # Get unique driver IDs for multi-class classification
    driver_ids = y.unique()
    print(f"Model will be trained on {len(driver_ids)} unique drivers")
    driver_names = race_data[['driver_id', 'driver_name']].drop_duplicates()
    
    return X, y, all_feature_cols, driver_names

def train_model(X, y, feature_names=None, cv=3):
    """
    Train a machine learning model to predict F1 race winners.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - feature_names: List of feature names
    - cv: Number of cross-validation folds
    
    Returns:
    - Trained model
    - Training metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define numeric features (qualifying position, standings)
    numeric_features = [
        'qualifying_position', 
        'position_standings', 
        'points_standings', 
        'wins',
        'avg_position_at_circuit',
        'best_position_at_circuit', 
        'team_avg_position_at_circuit', 
        'team_best_position_at_circuit',
        'grid_finish_correlation',
        'recent_form'
    ]
    numeric_features = [f for f in numeric_features if f in X.columns]
    
    # All other features are assumed to be categorical (one-hot encoded)
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Column transformer to apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('pass', 'passthrough', categorical_features)
        ])
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])
    
    # Define parameter grid for tuning
    # Define parameter grid for tuning
    param_grid = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [None, 15, 30, 45],
        'classifier__min_samples_split': [2, 5, 10]
    }
    # Use GridSearchCV to find best parameters
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 score: {metrics['f1_score']:.4f}")
    
    return best_model, metrics

def save_model(model, feature_names, driver_names, metrics=None):
    """
    Save the trained model and related artifacts.
    
    Parameters:
    - model: Trained model
    - feature_names: List of feature names
    - driver_names: DataFrame with driver_id and driver_name
    - metrics: Training metrics (optional)
    """
    ensure_model_dir()
    
    # Save the model
    joblib.dump(model, os.path.join(MODEL_DIR, 'f1_predictor_model.pkl'))
    
    # Save the feature names
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Filter driver labels to only include current drivers if possible
    try:
        current_standings = fetch_current_season_standings()
        if 'drivers' in current_standings:
            current_driver_ids = [driver['driver_id'] for driver in current_standings['drivers']]
            driver_labels = [d_id for d_id in driver_names['driver_id'].tolist() if d_id in current_driver_ids]
            print(f"Saved {len(driver_labels)} current driver labels for prediction")
        else:
            driver_labels = driver_names['driver_id'].tolist()
    except Exception:
        driver_labels = driver_names['driver_id'].tolist()
        
    with open(os.path.join(MODEL_DIR, 'driver_labels.pkl'), 'wb') as f:
        pickle.dump(driver_labels, f)
    
    # Create a mapping from driver_id to driver_name (filtered to saved drivers)
    driver_mapping = driver_names[driver_names['driver_id'].isin(driver_labels)].set_index('driver_id')['driver_name'].to_dict()
    print(f"Saving mapping for {len(driver_mapping)} drivers")
    with open(os.path.join(MODEL_DIR, 'driver_mapping.pkl'), 'wb') as f:
        pickle.dump(driver_mapping, f)
    
    # Save metrics if provided
    if metrics:
        with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'w') as f:
            json.dump(metrics, f)
    
    print(f"Model and artifacts saved to {MODEL_DIR}/")

def load_model():
    """
    Load the trained model and related artifacts.
    
    Returns:
    - model: Trained model
    - feature_names: List of feature names
    - driver_labels: List of driver labels
    """
    model_path = os.path.join(MODEL_DIR, 'f1_predictor_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    
    # Load model
    model = joblib.load(model_path)
    
    # Load feature names
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load driver labels
    with open(os.path.join(MODEL_DIR, 'driver_labels.pkl'), 'rb') as f:
        driver_labels = pickle.load(f)
    
    return model

def preprocess_prediction_data(upcoming_race, current_standings):
    """
    Preprocess data for an upcoming race to make predictions.
    
    Parameters:
    - upcoming_race: Data about the upcoming race
    - current_standings: Current season driver standings
    
    Returns:
    - X_pred: Processed features for prediction
    """
    # Load feature names to ensure we create the same format
    try:
        with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Feature names file not found. Please train the model first.")
    
    # Create a dataframe with one row per driver in current standings
    drivers = current_standings['drivers']
    rows = []
    
    # Add rows for each driver
    for driver in drivers:
        # Basic driver info
        row = {
            'driver_id': driver['driver_id'],
            'driver_name': driver['driver_name'],
            'constructor': driver['constructor'],
            'position_standings': driver['position'],
            'points_standings': driver['points'],
            'wins': driver['wins'],
            'circuit_name': upcoming_race['circuit']
        }
        
        # For simplicity, use current qualifying positions based on championship standings
        # In a real app, you would update this after qualifying
        row['qualifying_position'] = driver['position']
        
        rows.append(row)
    
    # Create dataframe
    pred_df = pd.DataFrame(rows)
    
    # Create one-hot encoded features for constructor and circuit
    constructor_dummies = pd.get_dummies(pred_df['constructor'], prefix='constructor')
    circuit_dummies = pd.get_dummies(pred_df['circuit_name'], prefix='circuit')
    
    # Combine all features
    pred_df = pd.concat([pred_df, constructor_dummies, circuit_dummies], axis=1)
    
    # Create feature matrix matching the training data columns
    X_pred = pd.DataFrame(index=pred_df.index)
    
    # Add numeric features
    numeric_features = [
        'qualifying_position', 
        'position_standings', 
        'points_standings', 
        'wins',
        'avg_position_at_circuit',
        'best_position_at_circuit', 
        'team_avg_position_at_circuit', 
        'team_best_position_at_circuit',
        'grid_finish_correlation',
        'recent_form'
    ]
    for col in numeric_features:
        if col in pred_df.columns:
            X_pred[col] = pred_df[col]
        else:
            X_pred[col] = 0
    
    # Add one-hot encoded features (ensure all expected features are present)
    for col in feature_names:
        if col not in numeric_features:
            # If the column is in our prediction data, use it
            if col in pred_df.columns:
                X_pred[col] = pred_df[col]
            # Otherwise create a zero column
            else:
                X_pred[col] = 0
    
    # Ensure we have all required columns from training
    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    # Keep only the columns used in training
    X_pred = X_pred[feature_names]
    
    return X_pred

def train_and_save_model():
    """
    Complete workflow to fetch data, train model and save it.
    """
    print("Fetching historical F1 data...")
    data = fetch_historical_data()
    
    print("Engineering features...")
    X, y, feature_names, driver_names = engineer_features(data)
    
    print("Training model...")
    model, metrics = train_model(X, y, feature_names)
    
    print("Saving model...")
    save_model(model, feature_names, driver_names, metrics)
    
    return model, metrics


if __name__ == "__main__":
    # Run the training workflow when script is executed directly
    train_and_save_model()

