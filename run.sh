#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# First install base packages
pip install --upgrade pip setuptools wheel

# Then install other dependencies
pip install -r requirements.txt

# Train the model if it doesn't exist
if [ ! -d "models" ]; then
    echo "Training model..."
    python3 model_trainer.py
fi

# Start the Flask application
python3 app.py

