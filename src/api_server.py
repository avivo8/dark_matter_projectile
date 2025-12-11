#!/usr/bin/env python3
"""
Flask API server for interactive dark matter detection.
Loads the trained VQC model and provides prediction endpoints.
"""

import os
import sys
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Global variables for model and scaler
vqc_model = None
scaler = None

def load_model():
    """Load the trained VQC model and scaler."""
    global vqc_model, scaler
    
    models_dir = os.path.join(project_root, 'models')
    
    try:
        # Load scaler
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully!")
        
        # Try to load the full model
        try:
            with open(os.path.join(models_dir, 'vqc_model.pkl'), 'rb') as f:
                data = f.read()
                if len(data) > 0:
                    vqc_model = pickle.loads(data)
                    print("Full model loaded successfully!")
                    return True
        except (FileNotFoundError, ValueError, EOFError):
            print("Full model file not found or empty. Model will need to be retrained.")
            return False
            
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': vqc_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict dark matter presence from ellipticity features."""
    try:
        data = request.json
        features = data.get('features', [])
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Convert to numpy array
        X = np.array(features)
        
        if X.shape[1] != 2:
            return jsonify({'error': 'Features must have 2 dimensions (eps1, eps2)'}), 400
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        if vqc_model is None:
            # Fallback to rule-based prediction if model not loaded
            predictions = []
            probabilities = []
            for x in X_scaled:
                eps1, eps2 = x[0], x[1]
                gamma_tot = np.sqrt(eps1**2 + eps2**2)
                pred = 1 if gamma_tot > 0.5 else 0  # Scaled threshold
                prob = min(1.0, max(0.0, gamma_tot))
                predictions.append(int(pred))
                probabilities.append([1 - prob, prob])
        else:
            predictions = vqc_model.predict(X_scaled).tolist()
            probabilities = vqc_model.predict_proba(X_scaled).tolist()
        
        return jsonify({
            'predictions': predictions,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict dark matter for multiple galaxies."""
    try:
        data = request.json
        galaxies = data.get('galaxies', [])
        
        if not galaxies:
            return jsonify({'error': 'No galaxies provided'}), 400
        
        # Extract features
        features = [[g['eps1'], g['eps2']] for g in galaxies]
        
        # Convert to numpy array
        X = np.array(features)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        if vqc_model is None:
            # Fallback prediction
            predictions = []
            probabilities = []
            for x in X_scaled:
                eps1, eps2 = x[0], x[1]
                gamma_tot = np.sqrt(eps1**2 + eps2**2)
                pred = 1 if gamma_tot > 0.5 else 0
                prob = min(1.0, max(0.0, gamma_tot))
                predictions.append(int(pred))
                probabilities.append([1 - prob, prob])
        else:
            predictions = vqc_model.predict(X_scaled).tolist()
            probabilities = vqc_model.predict_proba(X_scaled).tolist()
        
        # Format response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': i,
                'prediction': pred,
                'probability': prob[1] if len(prob) > 1 else prob,
                'x': galaxies[i].get('x', 0),
                'y': galaxies[i].get('y', 0)
            })
        
        return jsonify({
            'results': results,
            'total': len(results),
            'dark_matter_count': sum(1 for r in results if r['prediction'] == 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    model_loaded = load_model()
    
    if not model_loaded:
        print("Warning: Model not fully loaded. Using fallback predictions.")
    
    print("Starting API server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Predict from features array")
    print("  POST /predict_batch - Predict from galaxy objects")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

