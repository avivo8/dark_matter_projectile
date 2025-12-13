#!/usr/bin/env python3
"""
Apply the improved VQR model to SDSS data for concentration predictions.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')

print("=" * 70)
print("Applying Improved VQR Model to SDSS Data")
print("=" * 70)

# Load improved model configuration
print("\n1. Loading improved VQR model...")
try:
    with open(os.path.join(models_dir, 'improved_vqr_config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(models_dir, 'target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    
    print("   ✓ Model configuration loaded")
except FileNotFoundError as e:
    print(f"   ✗ Error: {e}")
    print("   Please run train_improved_vqr.py first")
    exit(1)

# Reconstruct the model
num_qubits = config['num_qubits']
reps_feature = config['reps_feature']
reps_ansatz = config['reps_ansatz']
weights = config['weights']
combination_weights = config['combination_weights']

feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=reps_feature,
    entanglement='linear'
)

ansatz = RealAmplitudes(
    num_qubits=num_qubits,
    reps=reps_ansatz,
    entanglement='linear'
)

# Create observables
observable_z0 = SparsePauliOp(['Z' + 'I' * (num_qubits - 1)])
if num_qubits >= 2:
    observable_z1 = SparsePauliOp(['I' + 'Z' + 'I' * (num_qubits - 2)])
    observable_z0z1 = SparsePauliOp(['ZZ' + 'I' * (num_qubits - 2)])
else:
    observable_z1 = observable_z0
    observable_z0z1 = observable_z0

observables = [observable_z0, observable_z1, observable_z0z1]

# Create quantum circuit
qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

# Create EstimatorQNNs
estimator = Estimator()
qnns = []
for obs in observables:
    qnn = EstimatorQNN(
        estimator=estimator,
        circuit=qc,
        observables=obs,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )
    qnns.append(qnn)

print(f"   ✓ Model reconstructed with {len(observables)} observables")

# Load SDSS data
print("\n2. Loading SDSS data...")
df = pd.read_csv(os.path.join(data_dir, 'sdss_predictions_with_mass.csv'))
X = df[['Observed_Eps1', 'Observed_Eps2']].values
X_scaled = scaler.transform(X)

print(f"   ✓ Loaded {len(df)} galaxies")

# Make predictions
print("\n3. Making concentration predictions...")
print("   This may take a few minutes...")

# Forward pass: combine multiple expectation values
outputs = []
for qnn in qnns:
    exp_val = qnn.forward(X_scaled, weights)
    if len(exp_val.shape) > 1:
        exp_val = exp_val.flatten()
    outputs.append(exp_val)

# Combine: α⟨Z₀⟩ + β⟨Z₁⟩ + γ⟨Z₀Z₁⟩
combined = np.zeros_like(outputs[0])
for i, out in enumerate(outputs):
    combined += combination_weights[i] * out

# Convert back to original scale
log10_c_pred_improved = target_scaler.inverse_transform(combined.reshape(-1, 1)).flatten()

print(f"   ✓ Predictions complete!")
print(f"   Concentration range: {log10_c_pred_improved.min():.3f} - {log10_c_pred_improved.max():.3f}")

# Add predictions to dataframe
df['log10_c_pred_improved'] = log10_c_pred_improved

# Save updated dataframe
output_file = os.path.join(data_dir, 'sdss_predictions_with_improved.csv')
df.to_csv(output_file, index=False)
print(f"\n4. Saved predictions to: {output_file}")

print("\n" + "=" * 70)
print("Improved VQR Predictions Complete!")
print("=" * 70)

