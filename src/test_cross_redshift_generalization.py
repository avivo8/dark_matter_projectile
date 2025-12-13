#!/usr/bin/env python3
"""
Cross-Redshift Generalization Test for VQR at z=1

Goal: Evaluate the ability of the optimized VQR to generalize its prediction
of halo concentration to an out-of-distribution redshift (z=1), using synthetic
data that mimics physical evolution.

Phase 1: Generate synthetic z=1 test data
Phase 2: Run inference and compute metrics
Phase 3: Visualize and interpret results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp

# Get directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')
vis_dir = os.path.join(project_root, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

print("=" * 70)
print("Cross-Redshift Generalization Test: z=1")
print("=" * 70)

# ============================================================================
# Phase 1: Synthetic z=1 Data Generation
# ============================================================================
print("\nPhase 1: Generating synthetic z=1 test data...")

# Action 1.1: Generate synthetic z=1 dataset
N_samples = 200
z_test = 1.0  # Target redshift

# Use different random seed to ensure no overlap with z=0 training data
np.random.seed(999)  # Different from training seed (42)

# Generate halo masses in the same range as training
# Sample from log-normal distribution similar to training
log10_M_halo_z1 = np.random.uniform(3.5, 5.5, N_samples)

# Generate galaxy shape parameters (ellipticity proxy)
# Consistent with training assumptions
eps1_z1 = np.random.normal(0, 0.1, N_samples)
eps2_z1 = np.random.normal(0, 0.1, N_samples)

# Action 1.2: Generate reference concentrations using Duffy+2008 with z=1
# Duffy+2008: c(M,z) = A * (M/M_pivot)^B * (1+z)^C
# For z=1: c is lower than z=0 due to (1+z)^(-0.47) factor
A = 5.71
B = -0.084
C = -0.47
M_pivot = 2e12  # Msun/h

M_halo_z1 = 10**log10_M_halo_z1
c_ref_z1 = A * (M_halo_z1 / M_pivot)**B * (1 + z_test)**C
log10_c_ref_z1 = np.log10(c_ref_z1)

print(f"   ✓ Generated {N_samples} synthetic halos at z={z_test}")
print(f"   Halo mass range: {log10_M_halo_z1.min():.2f} - {log10_M_halo_z1.max():.2f}")
print(f"   Reference concentration range: {log10_c_ref_z1.min():.3f} - {log10_c_ref_z1.max():.3f}")
print(f"   Mean concentration: {log10_c_ref_z1.mean():.3f}")

# Create dataframe
df_z1 = pd.DataFrame({
    'z': z_test,
    'log10_M_halo': log10_M_halo_z1,
    'Observed_Eps1': eps1_z1,
    'Observed_Eps2': eps2_z1,
    'log10_c_ref': log10_c_ref_z1
})

# Action 1.2: Apply training-time normalization (CRITICAL)
# Load the scalers from z=0 training
print("\n   Applying z=0 training normalization...")
try:
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler_z0 = pickle.load(f)
    
    with open(os.path.join(models_dir, 'target_scaler.pkl'), 'rb') as f:
        target_scaler_z0 = pickle.load(f)
    
    print("   ✓ Loaded z=0 normalization parameters")
except FileNotFoundError as e:
    print(f"   ✗ Error: {e}")
    print("   Please train the improved VQR model first")
    exit(1)

# Normalize features using z=0 scaler (DO NOT recompute!)
X_z1 = df_z1[['Observed_Eps1', 'Observed_Eps2']].values
X_z1_scaled = scaler_z0.transform(X_z1)  # Use z=0 normalization

print(f"   ✓ Applied z=0 feature normalization")
print(f"   Normalized feature range: [{X_z1_scaled.min():.3f}, {X_z1_scaled.max():.3f}]")

# Save z=1 test data
output_file = os.path.join(data_dir, 'synthetic_z1_test_data.csv')
df_z1.to_csv(output_file, index=False)
print(f"   ✓ Saved z=1 test data to: {output_file}")

# ============================================================================
# Phase 2: Inference and Metric Evaluation
# ============================================================================
print("\nPhase 2: Running cross-redshift inference...")

# Action 2.1: Load the optimized VQR model
print("\n   Loading optimized VQR model...")
try:
    with open(os.path.join(models_dir, 'improved_vqr_config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    num_qubits = config['num_qubits']
    reps_feature = config['reps_feature']
    reps_ansatz = config['reps_ansatz']
    weights = config['weights']
    combination_weights = config['combination_weights']
    
    print("   ✓ Model configuration loaded")
except FileNotFoundError as e:
    print(f"   ✗ Error: {e}")
    print("   Please train the improved VQR model first")
    exit(1)

# Reconstruct the model
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

# Run inference on z=1 data
print("\n   Running inference on z=1 test data...")
print("   This may take a few minutes...")

# Forward pass: combine multiple expectation values
outputs = []
for qnn in qnns:
    exp_val = qnn.forward(X_z1_scaled, weights)
    if len(exp_val.shape) > 1:
        exp_val = exp_val.flatten()
    outputs.append(exp_val)

# Combine: α⟨Z₀⟩ + β⟨Z₁⟩ + γ⟨Z₀Z₁⟩
combined = np.zeros_like(outputs[0])
for i, out in enumerate(outputs):
    combined += combination_weights[i] * out

# Convert back to original scale using z=0 target scaler
log10_c_pred_z1_scaled = combined
log10_c_pred_z1 = target_scaler_z0.inverse_transform(
    log10_c_pred_z1_scaled.reshape(-1, 1)
).flatten()

print(f"   ✓ Inference complete!")
print(f"   Predicted concentration range: {log10_c_pred_z1.min():.3f} - {log10_c_pred_z1.max():.3f}")

# Add predictions to dataframe
df_z1['log10_c_pred'] = log10_c_pred_z1
df_z1['residual'] = log10_c_pred_z1 - log10_c_ref_z1

# Action 2.2: Compute performance metrics
print("\n   Computing performance metrics...")
mse_z1 = np.mean((log10_c_pred_z1 - log10_c_ref_z1)**2)
bias_z1 = np.mean(log10_c_pred_z1 - log10_c_ref_z1)
rmse_z1 = np.sqrt(mse_z1)
mae_z1 = np.mean(np.abs(log10_c_pred_z1 - log10_c_ref_z1))

print(f"\n   --- z=1 Test Results ---")
print(f"   MSE:  {mse_z1:.6f}")
print(f"   RMSE: {rmse_z1:.6f}")
print(f"   MAE:  {mae_z1:.6f}")
print(f"   Bias: {bias_z1:+.6f}")

# Compare with z=0 performance (from training)
# Load z=0 validation results for comparison
try:
    df_z0_val = pd.read_csv(os.path.join(data_dir, 'concentration_validation.csv'))
    mse_z0 = np.mean((df_z0_val['log10_c_pred'] - df_z0_val['log10_c_ref'])**2)
    bias_z0 = np.mean(df_z0_val['log10_c_pred'] - df_z0_val['log10_c_ref'])
    
    print(f"\n   --- Comparison with z=0 ---")
    print(f"   z=0 MSE:  {mse_z0:.6f}")
    print(f"   z=1 MSE:  {mse_z1:.6f}")
    print(f"   MSE increase: {(mse_z1 - mse_z0) / mse_z0 * 100:+.1f}%")
    print(f"\n   z=0 Bias: {bias_z0:+.6f}")
    print(f"   z=1 Bias: {bias_z1:+.6f}")
    print(f"   Bias change: {bias_z1 - bias_z0:+.6f}")
except FileNotFoundError:
    print("   ⚠ z=0 validation data not found for comparison")

# Save results
results_file = os.path.join(data_dir, 'cross_redshift_z1_results.csv')
df_z1.to_csv(results_file, index=False)
print(f"\n   ✓ Saved results to: {results_file}")

# ============================================================================
# Phase 3: Visualization and Interpretation
# ============================================================================
print("\nPhase 3: Generating diagnostic plots...")

# Action 3.1: Residuals vs Halo Mass plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot z=1 residuals
scatter_z1 = ax.scatter(
    log10_M_halo_z1,
    df_z1['residual'].values,
    s=50,
    alpha=0.6,
    c='red',
    edgecolors='darkred',
    linewidths=1,
    label=f'z=1 Test Data (MSE={mse_z1:.4f}, Bias={bias_z1:+.4f})',
    zorder=2
)

# Overlay z=0 residual trend if available
try:
    df_z0_val = pd.read_csv(os.path.join(data_dir, 'concentration_validation.csv'))
    resid_z0 = df_z0_val['log10_c_pred'] - df_z0_val['log10_c_ref']
    log10_M_z0 = df_z0_val['log10_M_halo'].values
    
    # Plot z=0 residuals as reference
    ax.scatter(
        log10_M_z0,
        resid_z0,
        s=30,
        alpha=0.4,
        c='green',
        edgecolors='darkgreen',
        linewidths=0.5,
        label=f'z=0 Training (MSE={mse_z0:.4f}, Bias={bias_z0:+.4f})',
        zorder=1
    )
    
    # Add trend line for z=0
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(log10_M_z0, resid_z0)
    M_range = np.linspace(log10_M_z0.min(), log10_M_z0.max(), 100)
    trend_z0 = slope * M_range + intercept
    ax.plot(
        M_range,
        trend_z0,
        'g--',
        linewidth=2,
        alpha=0.7,
        label='z=0 Trend Line',
        zorder=0
    )
except Exception as e:
    print(f"   ⚠ Could not overlay z=0 reference: {e}")

# Reference line at zero
ax.axhline(0.0, color='black', linestyle='-', linewidth=2, alpha=0.5, zorder=0)

# Labels and formatting
ax.set_xlabel(r'$\log_{10} M_{\mathrm{halo}}$', fontsize=14, fontweight='bold')
ax.set_ylabel(
    r'$\log_{10} c_{\mathrm{pred}} - \log_{10} c_{\mathrm{ref}}$',
    fontsize=14,
    fontweight='bold'
)
ax.set_title(
    'Cross-Redshift Generalization Test\nResiduals vs Halo Mass (z=1 vs z=0)',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.9)

# Add metrics text box
metrics_text = f'z=1 Test Results:\n'
metrics_text += f'  MSE: {mse_z1:.6f}\n'
metrics_text += f'  RMSE: {rmse_z1:.6f}\n'
metrics_text += f'  Bias: {bias_z1:+.6f}\n'
metrics_text += f'\nHypothesis Check:\n'
if mse_z1 < 0.022:  # Original baseline MSE
    metrics_text += f'  ✓ MSE < baseline (0.022)\n'
else:
    metrics_text += f'  ✗ MSE > baseline (0.022)\n'

if abs(bias_z1) < abs(bias_z0) if 'bias_z0' in locals() else abs(bias_z1) < 0.05:
    metrics_text += f'  ✓ Bias acceptable\n'
else:
    metrics_text += f'  ⚠ Bias may be problematic\n'

ax.text(
    0.02,
    0.98,
    metrics_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
)

plt.tight_layout()

# Save plot
output_plot = os.path.join(vis_dir, 'cross_redshift_generalization_z1.png')
fig.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved plot to: {output_plot}")
plt.close(fig)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("CROSS-REDSHIFT GENERALIZATION TEST COMPLETE")
print("=" * 70)
print(f"\nTest Configuration:")
print(f"  Redshift: z={z_test}")
print(f"  Test samples: {N_samples}")
print(f"  Used z=0 normalization: ✓")

print(f"\nResults:")
print(f"  MSE:  {mse_z1:.6f}")
print(f"  RMSE: {rmse_z1:.6f}")
print(f"  MAE:  {mae_z1:.6f}")
print(f"  Bias: {bias_z1:+.6f}")

if 'mse_z0' in locals():
    print(f"\nComparison with z=0:")
    print(f"  MSE increase: {(mse_z1 - mse_z0) / mse_z0 * 100:+.1f}%")
    print(f"  Bias change: {bias_z1 - bias_z0:+.6f}")

print(f"\nInterpretation:")
if mse_z1 < 0.022:
    print(f"  ✓ Model generalizes well: MSE ({mse_z1:.6f}) < baseline (0.022)")
else:
    print(f"  ⚠ Model may not generalize: MSE ({mse_z1:.6f}) > baseline (0.022)")

if abs(bias_z1) < 0.05:
    print(f"  ✓ Bias is acceptable: {bias_z1:+.6f}")
else:
    print(f"  ⚠ Bias may indicate systematic error: {bias_z1:+.6f}")

print(f"\nFiles created:")
print(f"  Test data: {output_file}")
print(f"  Results: {results_file}")
print(f"  Plot: {output_plot}")
print("=" * 70)

