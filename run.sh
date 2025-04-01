#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Check if the models directory exists and has any models
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "No models found. Downloading a model..."
    python3 download_model.py
fi

# Check if an engine was specified
if [ "$1" == "--engine" ] || [ "$1" == "-e" ]; then
    ENGINE=$2
    shift 2
    
    # Run with specified engine
    if [ "$ENGINE" == "vosk" ] || [ "$ENGINE" == "whisper" ]; then
        echo "Running with $ENGINE engine..."
        python3 app.py
    else
        echo "Error: Unknown engine '$ENGINE'"
        echo "Supported engines: 'vosk', 'whisper'"
        exit 1
    fi
else
    # Run with default engine from config
    python3 app.py
fi