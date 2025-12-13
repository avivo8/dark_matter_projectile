#!/usr/bin/env python3
"""
Test the multi-redshift VQR model on z=1 test data
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
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt

# Get directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')
vis_dir = os.path.join(project_root, 'visualizations')

print("=" * 70)
print("Testing Multi-Redshift VQR Model on z=1 Data")
print("=" * 70)

# Load model
print("\n1. Loading multi-redshift model...")
try:
    with open(os.path.join(models_dir, 'multi_redshift_vqr_config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    with open(os.path.join(models_dir, 'scaler_multi_z.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(models_dir, 'target_scaler_multi_z.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    
    print("   ✓ Model loaded")
except FileNotFoundError as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Reconstruct model
num_qubits = config['num_qubits']
reps_feature = config['reps_feature']
reps_ansatz = config['reps_ansatz']
weights = config['weights']
combination_weights = config['combination_weights']
bias_correction_a = config['bias_correction_a']
bias_correction_b = config['bias_correction_b']

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
observable_z0 = SparsePauliOp(['ZII'])
observable_z1 = SparsePauliOp(['IZI'])
observable_z2 = SparsePauliOp(['IIZ'])
observables = [observable_z0, observable_z1, observable_z2]

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

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

print(f"   ✓ Model reconstructed")

# Load z=1 test data
print("\n2. Loading z=1 test data...")
df_z1 = pd.read_csv(os.path.join(data_dir, 'synthetic_z1_test_data.csv'))
X_z1 = df_z1[['Observed_Eps1', 'Observed_Eps2', 'z']].values
X_z1_scaled = scaler.transform(X_z1)
log10_c_ref_z1 = df_z1['log10_c_ref'].values

print(f"   ✓ Loaded {len(df_z1)} test samples")

# Make predictions
print("\n3. Making predictions...")
outputs = []
for qnn in qnns:
    exp_val = qnn.forward(X_z1_scaled, weights)
    if len(exp_val.shape) > 1:
        exp_val = exp_val.flatten()
    outputs.append(exp_val)

combined = np.zeros_like(outputs[0])
for i, out in enumerate(outputs):
    combined += combination_weights[i] * out

# Apply bias correction
z_normalized = X_z1_scaled[:, 2]
correction = bias_correction_a * z_normalized + bias_correction_b
predictions_scaled = combined - correction

# Convert to original scale
log10_c_pred_z1 = target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()

print(f"   ✓ Predictions complete")

# Compute metrics
mse_z1 = np.mean((log10_c_pred_z1 - log10_c_ref_z1)**2)
bias_z1 = np.mean(log10_c_pred_z1 - log10_c_ref_z1)
rmse_z1 = np.sqrt(mse_z1)

print(f"\n4. Results:")
print(f"   MSE:  {mse_z1:.6f}")
print(f"   RMSE: {rmse_z1:.6f}")
print(f"   Bias: {bias_z1:+.6f}")

# Compare with previous z=1 test (without redshift feature)
try:
    df_prev = pd.read_csv(os.path.join(data_dir, 'cross_redshift_z1_results.csv'))
    mse_prev = np.mean((df_prev['log10_c_pred'] - df_prev['log10_c_ref'])**2)
    bias_prev = np.mean(df_prev['log10_c_pred'] - df_prev['log10_c_ref'])
    
    print(f"\n5. Comparison with previous model (no redshift feature):")
    print(f"   Previous MSE:  {mse_prev:.6f}")
    print(f"   Current MSE:   {mse_z1:.6f}")
    print(f"   Improvement:   {(mse_prev - mse_z1) / mse_prev * 100:.1f}%")
    print(f"\n   Previous Bias: {bias_prev:+.6f}")
    print(f"   Current Bias:  {bias_z1:+.6f}")
    print(f"   Bias improvement: {abs(bias_prev) - abs(bias_z1):+.6f}")
except FileNotFoundError:
    print("\n5. Previous results not found for comparison")

# Create visualization
print("\n6. Creating visualization...")
fig, ax = plt.subplots(figsize=(12, 7))

residuals = log10_c_pred_z1 - log10_c_ref_z1
log10_M = df_z1['log10_M_halo'].values

ax.scatter(
    log10_M,
    residuals,
    s=50,
    alpha=0.6,
    c='blue',
    edgecolors='darkblue',
    linewidths=1,
    label=f'Multi-Redshift Model (MSE={mse_z1:.4f}, Bias={bias_z1:+.4f})'
)

ax.axhline(0.0, color='black', linestyle='-', linewidth=2, alpha=0.5)

ax.set_xlabel(r'$\log_{10} M_{\mathrm{halo}}$', fontsize=14, fontweight='bold')
ax.set_ylabel(
    r'$\log_{10} c_{\mathrm{pred}} - \log_{10} c_{\mathrm{ref}}$',
    fontsize=14,
    fontweight='bold'
)
ax.set_title(
    'Multi-Redshift VQR: z=1 Test Results\n(with Redshift Feature & Bias Correction)',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.9)

plt.tight_layout()
output_plot = os.path.join(vis_dir, 'multi_redshift_z1_test.png')
fig.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved plot to: {output_plot}")
plt.close(fig)

# Save results
df_z1['log10_c_pred_multi_z'] = log10_c_pred_z1
df_z1['residual_multi_z'] = residuals
results_file = os.path.join(data_dir, 'multi_redshift_z1_results.csv')
df_z1.to_csv(results_file, index=False)
print(f"   ✓ Saved results to: {results_file}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)

