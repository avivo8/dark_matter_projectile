#!/usr/bin/env python3
"""
Phase 2 & 3: Improved Quantum Circuit Architecture for Concentration Regression

Action 2.1: Increase circuit expressivity (reps_ansatz: 3 -> 6)
Action 2.2: Multi-expectation value measurement strategy
Action 3.1: Bias mitigation via cost function modification

This script trains an improved Variational Quantum Regressor (VQR) with:
- Increased ansatz depth (6 layers)
- Multiple expectation value measurements (⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩)
- Bias-aware cost function
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import VQR
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import Estimator
import pickle
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')

print("=" * 70)
print("Phase 2 & 3: Improved VQR Training")
print("=" * 70)

# Load concentration validation data (has halo masses and concentrations)
print("\n1. Loading concentration data...")
try:
    df = pd.read_csv(os.path.join(data_dir, 'sdss_predictions_with_mass.csv'))
    # Use log10_c_pred as target (or log10_c_ref if we want to train on reference)
    # For now, we'll use log10_c_ref as the target to train the model
    X = df[['Observed_Eps1', 'Observed_Eps2']].values
    Y = df['log10_c_ref'].values  # Target: reference concentration
    
    print(f"   ✓ Loaded {len(df)} galaxies")
    print(f"   Concentration range: {Y.min():.3f} - {Y.max():.3f}")
except FileNotFoundError:
    print("   ⚠ Concentration data not found. Using synthetic data...")
    # Fallback: generate synthetic data
    n_samples = 200
    X = np.random.uniform(-0.3, 0.3, (n_samples, 2))
    # Generate concentrations based on a simple relation
    Y = 1.3 + 0.1 * np.random.randn(n_samples)

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Normalize targets to [0, 1] for quantum regression
target_scaler = MinMaxScaler()
Y_scaled = target_scaler.fit_transform(Y.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled[:150], Y_scaled[:150], test_size=0.3, random_state=42
)

print(f"\n2. Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# ============================================================================
# Action 2.1: Increase Circuit Expressivity
# ============================================================================
print("\n3. Building improved quantum circuit...")

num_qubits = X_train.shape[1]  # 2 qubits for 2 features
reps_feature = 2
reps_ansatz = 6  # INCREASED from 3 to 6 (Action 2.1)

feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=reps_feature,
    entanglement='linear'
)

ansatz = RealAmplitudes(
    num_qubits=num_qubits,
    reps=reps_ansatz,  # Increased expressivity
    entanglement='linear'
)

print(f"   ✓ Feature map: {reps_feature} repetitions")
print(f"   ✓ Ansatz: {reps_ansatz} repetitions (increased from 3)")
print(f"   ✓ Total parameters: {ansatz.num_parameters}")

# ============================================================================
# Action 2.2: Multi-Expectation Value Measurement Strategy
# ============================================================================
print("\n4. Implementing multi-expectation value measurement...")

# Create quantum circuit combining feature map and ansatz
qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

# Define observables for multiple expectation values
# We'll measure ⟨Z₀⟩, ⟨Z₁⟩, and their combination
from qiskit.quantum_info import SparsePauliOp

# Observable 1: ⟨Z₀⟩
observable_z0 = SparsePauliOp(['Z' + 'I' * (num_qubits - 1)])

# Observable 2: ⟨Z₁⟩ (if num_qubits >= 2)
if num_qubits >= 2:
    observable_z1 = SparsePauliOp(['I' + 'Z' + 'I' * (num_qubits - 2)])
else:
    observable_z1 = observable_z0

# Observable 3: ⟨Z₀Z₁⟩ (correlation)
if num_qubits >= 2:
    observable_z0z1 = SparsePauliOp(['ZZ' + 'I' * (num_qubits - 2)])
else:
    observable_z0z1 = observable_z0

# Create Estimator for expectation value computation
estimator = Estimator()

# For VQR, we'll use a custom approach with multiple observables
# Qiskit's VQR uses a single observable, so we'll create a custom regressor
# that combines multiple expectation values

class MultiObservableVQR:
    """
    Custom VQR that uses multiple expectation values:
    output = α⟨Z₀⟩ + β⟨Z₁⟩ + γ⟨Z₀Z₁⟩
    """
    
    def __init__(self, feature_map, ansatz, optimizer, estimator, 
                 observables, bias_weight=1.0):
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.estimator = estimator
        self.observables = observables
        self.bias_weight = bias_weight
        
        # Create quantum circuit
        self.qc = QuantumCircuit(num_qubits)
        self.qc.compose(feature_map, inplace=True)
        self.qc.compose(ansatz, inplace=True)
        
        # Create EstimatorQNNs for each observable
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
        
        # Initialize combination weights (α, β, γ)
        self.combination_weights = np.ones(len(observables)) / len(observables)
        
        # Store parameters
        self.weights = None
        
    def _forward(self, X, weights):
        """Compute forward pass: combine multiple expectation values"""
        outputs = []
        for qnn in self.qnns:
            # Get expectation values directly from EstimatorQNN
            exp_val = qnn.forward(X, weights)
            # Ensure it's 1D
            if len(exp_val.shape) > 1:
                exp_val = exp_val.flatten()
            outputs.append(exp_val)
        
        # Combine: α⟨Z₀⟩ + β⟨Z₁⟩ + γ⟨Z₀Z₁⟩
        combined = np.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            combined += self.combination_weights[i] * out
        
        return combined
    
    def _loss(self, X, y, weights):
        """Custom loss with bias penalty (Action 3.1)"""
        predictions = self._forward(X, weights)
        
        # MSE component
        mse = np.mean((predictions - y) ** 2)
        
        # Bias penalty component (Action 3.1)
        bias = np.mean(predictions - y)
        bias_penalty = self.bias_weight * bias ** 2
        
        return mse + bias_penalty
    
    def fit(self, X, y, maxiter=150):
        """Train the model"""
        print(f"   Training with {maxiter} iterations...")
        
        # Initialize weights
        initial_weights = np.random.uniform(-np.pi, np.pi, 
                                           self.ansatz.num_parameters)
        
        # Optimize
        def objective(weights):
            return self._loss(X, y, weights)
        
        result = self.optimizer.minimize(objective, initial_weights)
        self.weights = result.x
        
        # Optimize combination weights too
        from scipy.optimize import minimize
        
        def combo_objective(combo_weights):
            self.combination_weights = combo_weights
            return self._loss(X, y, self.weights)
        
        combo_result = minimize(combo_objective, self.combination_weights, 
                               method='L-BFGS-B', bounds=[(0, 1)] * len(self.observables))
        self.combination_weights = combo_result.x
        
        print(f"   ✓ Training complete!")
        print(f"   Combination weights: {self.combination_weights}")
        
    def predict(self, X):
        """Make predictions"""
        return self._forward(X, self.weights)
    
    def get_bias(self, X, y):
        """Compute bias for evaluation"""
        predictions = self.predict(X)
        return np.mean(predictions - y)

# Create improved model
observables = [observable_z0, observable_z1, observable_z0z1]
optimizer = COBYLA(maxiter=150)

print(f"   ✓ Using {len(observables)} observables: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₀Z₁⟩")

# Action 3.1: Bias mitigation via increased bias_weight
bias_weight = 2.0  # Penalize bias more strongly

vqr_model = MultiObservableVQR(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    estimator=estimator,
    observables=observables,
    bias_weight=bias_weight
)

print(f"   ✓ Bias weight: {bias_weight} (Action 3.1)")

# ============================================================================
# Training
# ============================================================================
print("\n5. Training improved VQR model...")
vqr_model.fit(X_train, Y_train, maxiter=150)

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

# Metrics
train_mse = mean_squared_error(Y_train_orig, Y_pred_train_orig)
test_mse = mean_squared_error(Y_test_orig, Y_pred_test_orig)
train_mae = mean_absolute_error(Y_train_orig, Y_pred_train_orig)
test_mae = mean_absolute_error(Y_test_orig, Y_pred_test_orig)

train_bias = np.mean(Y_pred_train_orig - Y_train_orig)
test_bias = np.mean(Y_pred_test_orig - Y_test_orig)

print(f"\n--- Training Results ---")
print(f"MSE: {train_mse:.6f}")
print(f"MAE: {train_mae:.6f}")
print(f"Bias: {train_bias:+.6f}")

print(f"\n--- Test Results ---")
print(f"MSE: {test_mse:.6f}")
print(f"MAE: {test_mae:.6f}")
print(f"Bias: {test_bias:+.6f}")

# ============================================================================
# Save Model
# ============================================================================
print("\n7. Saving improved model...")

model_config = {
    'num_qubits': num_qubits,
    'reps_feature': reps_feature,
    'reps_ansatz': reps_ansatz,
    'weights': vqr_model.weights,
    'combination_weights': vqr_model.combination_weights,
    'bias_weight': bias_weight,
    'num_observables': len(observables)
}

with open(os.path.join(models_dir, 'improved_vqr_config.pkl'), 'wb') as f:
    pickle.dump(model_config, f)

# Save scalers
with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(models_dir, 'target_scaler.pkl'), 'wb') as f:
    pickle.dump(target_scaler, f)

print("   ✓ Model saved successfully!")
print(f"\n{'='*70}")
print("Improved VQR Training Complete!")
print(f"{'='*70}")
print(f"\nImprovements implemented:")
print(f"  ✓ Action 2.1: Ansatz depth increased to {reps_ansatz} layers")
print(f"  ✓ Action 2.2: Multi-observable measurement ({len(observables)} observables)")
print(f"  ✓ Action 3.1: Bias-aware cost function (weight={bias_weight})")
print(f"\nTest Bias: {test_bias:+.6f} (target: near zero)")

