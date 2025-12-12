# Interactive Dark Matter Detection Guide

## Overview

The interactive feature allows you to upload an image of a galaxy grid, manually mark regions where you believe dark matter is present, and then use our quantum ML model to predict dark matter locations.

## Setup

### 1. Install Dependencies

Make sure you have Flask and Flask-CORS installed:

```bash
pip install flask flask-cors
```

Or install all requirements:

```bash
pip install -r python-requirements.txt
```

### 2. Train the Model (if not already done)

```bash
python3 src/generate_dark_matter_dataset.py
python3 src/train_model.py
```

### 3. Start the API Server

In one terminal, start the Flask API server:

```bash
python3 src/api_server.py
```

The server will run on `http://localhost:5000`

### 4. Start the Website

In another terminal, start the web server:

```bash
cd website
python3 -m http.server 8000
```

Then open `http://localhost:8000/interactive.html` in your browser.

## How to Use

1. **Upload Image**: Click the upload area or drag and drop a galaxy grid image
2. **Mark Dark Matter**: Click on galaxies where you believe dark matter is present (red markers)
3. **Mark Background**: Switch to "Mark Background" mode and click on galaxies that are background/no dark matter (blue markers)
4. **Generate Grid**: Click "Generate Galaxy Grid" to extract features from your marked regions
5. **Run Prediction**: Click "Run Prediction" to let the quantum ML model predict dark matter locations
6. **View Results**: Compare your markings (ground truth) with the model predictions

## Features

- **Interactive Marking**: Click on galaxies to mark them as dark matter or background
- **Real-time Visualization**: See your markings and predictions overlaid on the image
- **Accuracy Calculation**: Compare your markings with model predictions
- **Quantum ML Predictions**: Uses the trained VQC model for accurate predictions

## API Endpoints

The API server provides the following endpoints:

- `GET /health` - Check server status and model loading
- `POST /predict` - Predict from features array
  ```json
  {
    "features": [[eps1, eps2], [eps1, eps2], ...]
  }
  ```
- `POST /predict_batch` - Predict from galaxy objects
  ```json
  {
    "galaxies": [
      {"x": 100, "y": 200, "eps1": 0.05, "eps2": 0.03},
      ...
    ]
  }
  ```

## Troubleshooting

- **API not connecting**: Make sure the API server is running on port 5000
- **Model not loaded**: Ensure you've trained the model and it's in the `models/` directory
- **Predictions not accurate**: The model needs sufficient training data. Try retraining with more samples.

## Notes

- The current implementation extracts synthetic ellipticity features from image brightness
- For production use, you would need proper galaxy shape analysis algorithms
- The model uses the same VQC architecture as the main project

