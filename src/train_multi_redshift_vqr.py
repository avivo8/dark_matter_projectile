#!/usr/bin/env python3
"""
Multi-Redshift VQR Training with Redshift as Input Feature

This script:
1. Generates multi-redshift training data (z=0, 0.3, 0.6, 1.0)
2. Adds redshift as an input feature (3 features: eps1, eps2, z)
3. Trains improved VQR with redshift-dependent bias correction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
import pickle
import os

# Get directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')

print("=" * 70)
print("Multi-Redshift VQR Training with Redshift Feature")
print("=" * 70)

# ============================================================================
# Generate Multi-Redshift Training Data
# ============================================================================
print("\n1. Generating multi-redshift training data...")

# Define redshifts for training
redshifts = [0.0, 0.3, 0.6, 1.0]
n_samples_per_z = 30  # Reduced to 30 samples per redshift = 120 total (to prevent overfitting)
np.random.seed(42)

all_data = []

for z in redshifts:
    # Generate halo masses
    log10_M_halo = np.random.uniform(3.5, 5.5, n_samples_per_z)
    
    # Generate ellipticity (galaxy shape)
    eps1 = np.random.normal(0, 0.1, n_samples_per_z)
    eps2 = np.random.normal(0, 0.1, n_samples_per_z)
    
    # Generate reference concentrations using Duffy+2008
    A = 5.71
    B = -0.084
    C = -0.47
    M_pivot = 2e12
    
    M_halo = 10**log10_M_halo
    c_ref = A * (M_halo / M_pivot)**B * (1 + z)**C
    log10_c_ref = np.log10(c_ref)
    
    # Store data
    for i in range(n_samples_per_z):
        all_data.append({
            'z': z,
            'log10_M_halo': log10_M_halo[i],
            'Observed_Eps1': eps1[i],
            'Observed_Eps2': eps2[i],
            'log10_c_ref': log10_c_ref[i]
        })

df_multi_z = pd.DataFrame(all_data)
print(f"   ✓ Generated {len(df_multi_z)} samples across {len(redshifts)} redshifts")
print(f"   Redshifts: {redshifts}")
print(f"   Samples per redshift: {n_samples_per_z}")

# Save multi-redshift data
multi_z_file = os.path.join(data_dir, 'multi_redshift_training_data.csv')
df_multi_z.to_csv(multi_z_file, index=False)
print(f"   ✓ Saved to: {multi_z_file}")

# ============================================================================
# Prepare Features (including redshift)
# ============================================================================
print("\n2. Preparing features with redshift...")

# Features: [eps1, eps2, z]
X = df_multi_z[['Observed_Eps1', 'Observed_Eps2', 'z']].values
Y = df_multi_z['log10_c_ref'].values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Normalize targets
target_scaler = MinMaxScaler()
Y_scaled = target_scaler.fit_transform(Y.reshape(-1, 1)).flatten()

print(f"   ✓ Features shape: {X_scaled.shape}")
print(f"   ✓ Feature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
print(f"   ✓ Target range: [{Y_scaled.min():.3f}, {Y_scaled.max():.3f}]")

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.3, random_state=42
)

print(f"\n   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# ============================================================================
# Build Quantum Circuit (3 qubits for 3 features)
# ============================================================================
print("\n3. Building quantum circuit...")

num_qubits = X_train.shape[1]  # 3 qubits: eps1, eps2, z
reps_feature = 1  # Reduced feature map repetitions
reps_ansatz = 3  # Further reduced to prevent overfitting

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

print(f"   ✓ Feature map: {reps_feature} repetitions")
print(f"   ✓ Ansatz: {reps_ansatz} repetitions")
print(f"   ✓ Total parameters: {ansatz.num_parameters}")
print(f"   ✓ Qubits: {num_qubits} (eps1, eps2, z)")

# ============================================================================
# Multi-Observable Measurement
# ============================================================================
print("\n4. Setting up multi-observable measurement...")

# Create observables for 3 qubits (minimal set to prevent overfitting)
observable_z0 = SparsePauliOp(['ZII'])
observable_z1 = SparsePauliOp(['IZI'])
observable_z2 = SparsePauliOp(['IIZ'])

# Reduced to 3 observables to prevent overfitting
observables = [observable_z0, observable_z1, observable_z2]

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

print(f"   ✓ Using {len(observables)} observables")

# ============================================================================
# Custom VQR with Redshift-Dependent Bias Correction
# ============================================================================
class MultiRedshiftVQR:
    """VQR with redshift-dependent bias correction"""
    
    def __init__(self, feature_map, ansatz, optimizer, estimator, 
                 observables, bias_weight=2.0):
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.estimator = estimator
        self.observables = observables
        self.bias_weight = bias_weight
        
        self.qc = QuantumCircuit(num_qubits)
        self.qc.compose(feature_map, inplace=True)
        self.qc.compose(ansatz, inplace=True)
        
        self.qnns = []
        for obs in observables:
            qnn = EstimatorQNN(
                estimator=estimator,
                circuit=self.qc,
                observables=obs,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters
            )
            self.qnns.append(qnn)
        
        # Initialize combination weights
        self.combination_weights = np.ones(len(observables)) / len(observables)
        self.weights = None
        
        # Redshift-dependent bias correction parameters
        # bias_correction(z) = a * z + b
        self.bias_correction_a = 0.0
        self.bias_correction_b = 0.0
    
    def _forward(self, X, weights):
        """Forward pass with multiple observables"""
        outputs = []
        for qnn in self.qnns:
            exp_val = qnn.forward(X, weights)
            if len(exp_val.shape) > 1:
                exp_val = exp_val.flatten()
            outputs.append(exp_val)
        
        combined = np.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            combined += self.combination_weights[i] * out
        
        return combined
    
    def _apply_bias_correction(self, predictions, X):
        """Apply redshift-dependent bias correction"""
        # Extract redshift (3rd feature, already normalized)
        z_normalized = X[:, 2]  # z is the 3rd feature
        
        # Apply correction: pred_corrected = pred - (a*z + b)
        correction = self.bias_correction_a * z_normalized + self.bias_correction_b
        return predictions - correction
    
    def _loss(self, X, y, weights):
        """Loss with bias penalty and L2 regularization"""
        predictions = self._forward(X, weights)
        
        # Apply bias correction
        predictions_corrected = self._apply_bias_correction(predictions, X)
        
        # MSE component
        mse = np.mean((predictions_corrected - y)**2)
        
        # Bias penalty (per redshift bin)
        bias_penalty = 0.0
        unique_z = np.unique(X[:, 2])
        for z_val in unique_z:
            mask = X[:, 2] == z_val
            if np.any(mask):
                bias = np.mean(predictions_corrected[mask] - y[mask])
                bias_penalty += self.bias_weight * bias**2
        
        # L2 regularization to prevent overfitting (increased)
        l2_reg = 0.05 * np.mean(weights**2)
        
        return mse + bias_penalty / len(unique_z) + l2_reg
    
    def fit(self, X, y, maxiter=200):
        """Train the model"""
        print(f"   Training with {maxiter} iterations...")
        
        # Initialize weights
        initial_weights = np.random.uniform(-np.pi, np.pi, self.ansatz.num_parameters)
        
        # Optimize quantum circuit weights
        def objective(weights):
            return self._loss(X, y, weights)
        
        result = self.optimizer.minimize(objective, initial_weights)
        self.weights = result.x
        
        # Optimize combination weights
        from scipy.optimize import minimize
        def combo_objective(combo_weights):
            self.combination_weights = combo_weights
            return self._loss(X, y, self.weights)
        
        combo_result = minimize(
            combo_objective, 
            self.combination_weights,
            method='L-BFGS-B',
            bounds=[(0, 1)] * len(self.observables)
        )
        self.combination_weights = combo_result.x
        
        # Fit redshift-dependent bias correction
        predictions_raw = self._forward(X, self.weights)
        residuals = predictions_raw - y
        
        # Fit linear correction: residual = a*z + b
        z_normalized = X[:, 2]
        A_fit = np.vstack([z_normalized, np.ones(len(z_normalized))]).T
        bias_params = np.linalg.lstsq(A_fit, residuals, rcond=None)[0]
        self.bias_correction_a = bias_params[0]
        self.bias_correction_b = bias_params[1]
        
        print(f"   ✓ Training complete!")
        print(f"   Combination weights: {self.combination_weights[:3]}...")
        print(f"   Bias correction: {self.bias_correction_a:.4f}*z + {self.bias_correction_b:.4f}")
    
    def predict(self, X):
        """Make predictions with bias correction"""
        predictions_raw = self._forward(X, self.weights)
        return self._apply_bias_correction(predictions_raw, X)
    
    def get_bias_by_redshift(self, X, y):
        """Get bias per redshift"""
        predictions = self.predict(X)
        unique_z = np.unique(X[:, 2])
        biases = {}
        for z_val in unique_z:
            mask = X[:, 2] == z_val
            if np.any(mask):
                biases[z_val] = np.mean(predictions[mask] - y[mask])
        return biases

# Create model with reduced complexity
optimizer = COBYLA(maxiter=100)  # Further reduced iterations
bias_weight = 1.0  # Reduced bias weight

vqr_model = MultiRedshiftVQR(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    estimator=estimator,
    observables=observables,
    bias_weight=bias_weight
)

print(f"   ✓ Model created with redshift-dependent bias correction")

# ============================================================================
# Training
# ============================================================================
print("\n5. Training multi-redshift VQR model...")
print("   Using regularization to prevent overfitting:")
print("   - Reduced ansatz layers: 3 (from 6)")
print("   - Reduced feature map: 1 repetition")
print("   - Reduced observables: 3 (from 6)")
print("   - L2 regularization: 0.05")
print("   - Reduced training samples: 120 total")
print("   - Reduced max iterations: 100")
vqr_model.fit(X_train, Y_train, maxiter=100)

# ============================================================================
# Evaluation
# ============================================================================
print("\n6. Evaluating model...")

# Predictions
Y_pred_train = vqr_model.predict(X_train)
Y_pred_test = vqr_model.predict(X_test)

# Convert back to original scale
Y_train_orig = target_scaler.inverse_transform(Y_train.reshape(-1, 1)).flatten()
Y_test_orig = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
Y_pred_train_orig = target_scaler.inverse_transform(Y_pred_train.reshape(-1, 1)).flatten()
Y_pred_test_orig = target_scaler.inverse_transform(Y_pred_test.reshape(-1, 1)).flatten()

# Overall metrics
train_mse = mean_squared_error(Y_train_orig, Y_pred_train_orig)
test_mse = mean_squared_error(Y_test_orig, Y_pred_test_orig)
train_bias = np.mean(Y_pred_train_orig - Y_train_orig)
test_bias = np.mean(Y_pred_test_orig - Y_test_orig)

print(f"\n   --- Overall Results ---")
print(f"   Train MSE:  {train_mse:.6f}")
print(f"   Test MSE:   {test_mse:.6f}")
print(f"   Train Bias: {train_bias:+.6f}")
print(f"   Test Bias:  {test_bias:+.6f}")

# Per-redshift metrics
print(f"\n   --- Per-Redshift Bias ---")
X_test_orig = scaler.inverse_transform(X_test)
z_test_orig = X_test_orig[:, 2]  # Extract original z values
Y_test_by_z = {}
Y_pred_by_z = {}

for z in redshifts:
    mask = np.abs(z_test_orig - z) < 0.1
    if np.any(mask):
        Y_test_by_z[z] = Y_test_orig[mask]
        Y_pred_by_z[z] = Y_pred_test_orig[mask]
        bias_z = np.mean(Y_pred_by_z[z] - Y_test_by_z[z])
        mse_z = np.mean((Y_pred_by_z[z] - Y_test_by_z[z])**2)
        print(f"   z={z:.1f}: MSE={mse_z:.6f}, Bias={bias_z:+.6f}")

# ============================================================================
# Save Model
# ============================================================================
print("\n7. Saving model...")

model_config = {
    'num_qubits': num_qubits,
    'reps_feature': reps_feature,
    'reps_ansatz': reps_ansatz,
    'weights': vqr_model.weights,
    'combination_weights': vqr_model.combination_weights,
    'bias_correction_a': vqr_model.bias_correction_a,
    'bias_correction_b': vqr_model.bias_correction_b,
    'bias_weight': bias_weight,
    'num_observables': len(observables),
    'includes_redshift': True
}

with open(os.path.join(models_dir, 'multi_redshift_vqr_config.pkl'), 'wb') as f:
    pickle.dump(model_config, f)

with open(os.path.join(models_dir, 'scaler_multi_z.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(models_dir, 'target_scaler_multi_z.pkl'), 'wb') as f:
    pickle.dump(target_scaler, f)

print("   ✓ Model saved successfully!")
print(f"\n{'='*70}")
print("Multi-Redshift VQR Training Complete!")
print(f"{'='*70}")
print(f"\nModel includes:")
print(f"  ✓ Redshift as input feature (3 qubits)")
print(f"  ✓ Multi-redshift training data (z={redshifts})")
print(f"  ✓ Redshift-dependent bias correction")
print(f"  ✓ {len(observables)} observables")
print(f"  ✓ {reps_ansatz} ansatz layers")

