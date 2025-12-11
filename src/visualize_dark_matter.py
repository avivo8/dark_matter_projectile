#!/usr/bin/env python3
"""
Visualize Dark Matter concentration using the trained VQC model.
Generates new data and creates a heatmap showing where dark matter is detected.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import os
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler

# Get the project root directory (parent of src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')
visualizations_dir = os.path.join(project_root, 'visualizations')

# Load the saved model and scaler
print("Loading model and scaler...")
vqc_model = None
scaler = None
model_config = None

# Load scaler first (needed for retraining if model not found)
try:
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Scaler file not found. Please run train_model.py first.")
    exit(1)

# Try to load the full model first
try:
    with open(os.path.join(models_dir, 'vqc_model.pkl'), 'rb') as f:
        data = f.read()
        if len(data) > 0:
            vqc_model = pickle.loads(data)
            print("Full model loaded successfully!")
        else:
            raise ValueError("Empty pickle file")
except (FileNotFoundError, ValueError, EOFError):
    print("Full model file not found or empty. Trying to reconstruct from configuration...")
    try:
        with open(os.path.join(models_dir, 'vqc_model_config.pkl'), 'rb') as f:
            data = f.read()
            if len(data) > 0:
                model_config = pickle.loads(data)
            else:
                raise ValueError("Empty config file")
        
        # Recreate the model structure
        num_qubits = model_config['num_qubits']
        reps_feature = model_config['reps_feature']
        reps_ansatz = model_config['reps_ansatz']
        
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
        
        # Use a quick optimizer for retraining if needed
        optimizer = COBYLA(maxiter=50)  # Quick training for visualization
        sampler = Sampler()
        
        vqc_model = VQC(
            feature_map=feature_map,
            ansatz=ansatz, 
            optimizer=optimizer, 
            sampler=sampler, 
            loss='cross_entropy'
        )
        
        # Try to load weights if available
        if 'weights' in model_config and model_config['weights'] is not None:
            try:
                vqc_model.neural_network.weights = model_config['weights']
                print("Model reconstructed with weights!")
            except:
                print("Could not set weights directly. Will retrain quickly...")
                # Load training data and retrain quickly
                df_train = pd.read_csv(os.path.join(data_dir, 'dark_matter_dataset.csv'))
                X_train = df_train[['Observed_Eps1', 'Observed_Eps2']].values[:100]
                Y_train = df_train['Label'].values[:100]
                X_train_scaled = scaler.transform(X_train)
                print("Retraining model with minimal iterations for visualization...")
                vqc_model.fit(X_train_scaled, Y_train)
                print("Quick retraining completed!")
        else:
            print("No weights found. Retraining model quickly for visualization...")
            # Load training data and retrain quickly
            df_train = pd.read_csv('dark_matter_dataset.csv')
            X_train = df_train[['Observed_Eps1', 'Observed_Eps2']].values[:100]
            Y_train = df_train['Label'].values[:100]
            X_train_scaled = scaler.transform(X_train)
            print("Training model (this may take a moment)...")
            vqc_model.fit(X_train_scaled, Y_train)
            print("Quick retraining completed!")
    except (FileNotFoundError, ValueError, EOFError) as e:
        print(f"Config file issue: {e}. Will reconstruct model and retrain...")
        model_config = None
        vqc_model = None
except Exception as e:
    print(f"Error loading model: {e}")
    print("Will reconstruct model and retrain...")
    model_config = None
    vqc_model = None

# If we don't have a model yet, reconstruct it
if vqc_model is None or not hasattr(vqc_model, 'neural_network') or not hasattr(vqc_model.neural_network, 'weights'):
    if model_config is None:
        # Default configuration
        model_config = {'num_qubits': 2, 'reps_feature': 2, 'reps_ansatz': 3}
        print("Using default model configuration")
    
    num_qubits = model_config.get('num_qubits', 2)
    reps_feature = model_config.get('reps_feature', 2)
    reps_ansatz = model_config.get('reps_ansatz', 3)
    
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
    
    optimizer = COBYLA(maxiter=50)
    sampler = Sampler()
    
    vqc_model = VQC(
        feature_map=feature_map,
        ansatz=ansatz, 
        optimizer=optimizer, 
        sampler=sampler, 
        loss='cross_entropy'
    )
    
    # Try to load weights if available
    if 'weights' in model_config and model_config['weights'] is not None:
        try:
            vqc_model.neural_network.weights = model_config['weights']
            print("Model reconstructed with weights!")
        except:
            print("Could not set weights. Will retrain...")
            # Load training data and retrain
            df_train = pd.read_csv('dark_matter_dataset.csv')
            X_train = df_train[['Observed_Eps1', 'Observed_Eps2']].values[:100]
            Y_train = df_train['Label'].values[:100]
            X_train_scaled = scaler.transform(X_train)
            print("Retraining model with minimal iterations for visualization...")
            vqc_model.fit(X_train_scaled, Y_train)
            print("Quick retraining completed!")
    else:
        print("No weights found. Retraining model for visualization...")
        # Load training data and retrain
        df_train = pd.read_csv('dark_matter_dataset.csv')
        X_train = df_train[['Observed_Eps1', 'Observed_Eps2']].values[:100]
        Y_train = df_train['Label'].values[:100]
        X_train_scaled = scaler.transform(X_train)
        print("Training model (this may take a moment)...")
        vqc_model.fit(X_train_scaled, Y_train)
        print("Training completed!")

# Scaler already loaded above

# Generate new data for visualization with ground truth labels
print("\nGenerating new data for visualization...")
# Use current time as seed for randomization (different each run)
np.random.seed(int(time.time()) % 2**32)  # Random seed based on current time

n_new_samples = 300
n_lensed = 150
n_background = 150

# Generate intrinsic ellipticity
epsilon_1_int = np.random.uniform(-0.05, 0.05, n_new_samples)
epsilon_2_int = np.random.uniform(-0.05, 0.05, n_new_samples)

# Generate lensed galaxies (stronger shear)
gamma_1_lensed = np.random.uniform(0.05, 0.08, n_lensed)
gamma_2_lensed = np.random.uniform(0.05, 0.08, n_lensed)

# Generate background galaxies (weaker/zero shear)
gamma_1_background = np.random.uniform(-0.03, 0.03, n_background)
gamma_2_background = np.random.uniform(-0.03, 0.03, n_background)

# Combine and shuffle
gamma_1 = np.concatenate([gamma_1_lensed, gamma_1_background])
gamma_2 = np.concatenate([gamma_2_lensed, gamma_2_background])
indices = np.random.permutation(n_new_samples)
epsilon_1_int = epsilon_1_int[indices]
epsilon_2_int = epsilon_2_int[indices]
gamma_1 = gamma_1[indices]
gamma_2 = gamma_2[indices]

# Compute total shear and ground truth labels
gamma_tot = np.sqrt(gamma_1**2 + gamma_2**2)
ground_truth_labels = (gamma_tot > 0.05).astype(int)  # Ground truth: 1 if gamma_tot > 0.05, else 0

# Compute observed ellipticity
observed_eps1 = epsilon_1_int + gamma_1
observed_eps2 = epsilon_2_int + gamma_2

# Generate spatial x-y coordinates for each galaxy (in arbitrary units, e.g., arcseconds or Mpc)
# Randomly distribute galaxies across a 2D sky plane
spatial_x = np.random.uniform(0, 100, n_new_samples)  # X coordinate in spatial units
spatial_y = np.random.uniform(0, 100, n_new_samples)  # Y coordinate in spatial units

# Create DataFrame with new data including ground truth and spatial coordinates
new_data = pd.DataFrame({
    'Observed_Eps1': observed_eps1,
    'Observed_Eps2': observed_eps2,
    'Ground_Truth_Label': ground_truth_labels,
    'Total_Shear_Gamma': gamma_tot,
    'Spatial_X': spatial_x,
    'Spatial_Y': spatial_y
})

print(f"Generated {len(new_data)} new samples")

# Scale the new data using the same scaler
X_new = new_data[['Observed_Eps1', 'Observed_Eps2']].values
X_new_scaled = scaler.transform(X_new)

# Make predictions
print("Making predictions on new data...")
predictions = vqc_model.predict(X_new_scaled)
probabilities = vqc_model.predict_proba(X_new_scaled)

# Add predictions to dataframe
new_data['Predicted_Label'] = predictions
new_data['Dark_Matter_Probability'] = probabilities[:, 1]  # Probability of dark matter (label=1)

# Calculate accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(new_data['Ground_Truth_Label'], new_data['Predicted_Label'])
cm = confusion_matrix(new_data['Ground_Truth_Label'], new_data['Predicted_Label'])

print(f"\nModel Performance on New Data:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Confusion Matrix:")
print(f"  True Negatives (TN): {cm[0,0]}, False Positives (FP): {cm[0,1]}")
print(f"  False Negatives (FN): {cm[1,0]}, True Positives (TP): {cm[1,1]}")

# Create a grid for smooth visualization
print("Creating visualization grid...")
eps1_min, eps1_max = new_data['Observed_Eps1'].min(), new_data['Observed_Eps1'].max()
eps2_min, eps2_max = new_data['Observed_Eps2'].min(), new_data['Observed_Eps2'].max()

# Extend the range slightly for better visualization
eps1_range = eps1_max - eps1_min
eps2_range = eps2_max - eps2_min
eps1_min -= 0.1 * eps1_range
eps1_max += 0.1 * eps1_range
eps2_min -= 0.1 * eps2_range
eps2_max += 0.1 * eps2_range

# Create a fine grid
grid_resolution = 50
eps1_grid = np.linspace(eps1_min, eps1_max, grid_resolution)
eps2_grid = np.linspace(eps2_min, eps2_max, grid_resolution)
Eps1_mesh, Eps2_mesh = np.meshgrid(eps1_grid, eps2_grid)

# Flatten grid for prediction
grid_points = np.column_stack([Eps1_mesh.ravel(), Eps2_mesh.ravel()])
grid_scaled = scaler.transform(grid_points)

# Make predictions on grid
print("Making predictions on visualization grid...")
grid_predictions = vqc_model.predict(grid_scaled)
grid_probabilities = vqc_model.predict_proba(grid_scaled)

# Reshape for plotting
dark_matter_prob_grid = grid_probabilities[:, 1].reshape(Eps1_mesh.shape)
dark_matter_label_grid = grid_predictions.reshape(Eps1_mesh.shape)

# Create separate figures for each visualization
print("Creating visualization figures...")
correct = new_data['Predicted_Label'] == new_data['Ground_Truth_Label']
incorrect = ~correct

# Figure 1: Model Predictions (Probability Heatmap)
print("Creating Figure 1: Model Predictions...")
fig1, ax1 = plt.subplots(figsize=(10, 8))
im1 = ax1.contourf(Eps1_mesh, Eps2_mesh, dark_matter_prob_grid, 
                   levels=20, cmap='RdYlBu_r', alpha=0.8)
ax1.scatter(new_data[new_data['Predicted_Label'] == 1]['Observed_Eps1'],
            new_data[new_data['Predicted_Label'] == 1]['Observed_Eps2'],
            c='red', s=30, alpha=0.6, label='Predicted: Dark Matter', edgecolors='black', linewidths=0.5)
ax1.scatter(new_data[new_data['Predicted_Label'] == 0]['Observed_Eps1'],
            new_data[new_data['Predicted_Label'] == 0]['Observed_Eps2'],
            c='blue', s=30, alpha=0.6, label='Predicted: Background', edgecolors='black', linewidths=0.5)
ax1.set_xlabel('Observed Epsilon 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('Observed Epsilon 2', fontsize=12, fontweight='bold')
ax1.set_title('Model Predictions\n(Probability Heatmap)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
plt.colorbar(im1, ax=ax1, label='Dark Matter Probability')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, '1_model_predictions.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 1_model_predictions.png")

# Figure 2: Ground Truth Labels
print("Creating Figure 2: Ground Truth Labels...")
fig2, ax2 = plt.subplots(figsize=(10, 8))
im2 = ax2.contourf(Eps1_mesh, Eps2_mesh, dark_matter_prob_grid, 
                   levels=20, cmap='RdYlBu_r', alpha=0.3)
ax2.scatter(new_data[new_data['Ground_Truth_Label'] == 1]['Observed_Eps1'],
            new_data[new_data['Ground_Truth_Label'] == 1]['Observed_Eps2'],
            c='darkred', s=50, alpha=0.8, label='Ground Truth: Dark Matter', 
            edgecolors='black', linewidths=1.5, marker='s')
ax2.scatter(new_data[new_data['Ground_Truth_Label'] == 0]['Observed_Eps1'],
            new_data[new_data['Ground_Truth_Label'] == 0]['Observed_Eps2'],
            c='darkblue', s=50, alpha=0.8, label='Ground Truth: Background', 
            edgecolors='black', linewidths=1.5, marker='s')
ax2.set_xlabel('Observed Epsilon 1', fontsize=12, fontweight='bold')
ax2.set_ylabel('Observed Epsilon 2', fontsize=12, fontweight='bold')
ax2.set_title('Ground Truth Labels\n(Actual Dark Matter Presence)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, '2_ground_truth_labels.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 2_ground_truth_labels.png")

# Figure 3: Prediction Accuracy (Correct vs Incorrect)
print("Creating Figure 3: Prediction Accuracy...")
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.contourf(Eps1_mesh, Eps2_mesh, dark_matter_label_grid, 
            levels=[-0.5, 0.5, 1.5], colors=['#4A90E2', '#E74C3C'], alpha=0.3)
# Correct predictions
ax3.scatter(new_data[correct & (new_data['Predicted_Label'] == 1)]['Observed_Eps1'],
            new_data[correct & (new_data['Predicted_Label'] == 1)]['Observed_Eps2'],
            c='green', s=60, alpha=0.8, marker='o', 
            edgecolors='darkgreen', linewidths=2, label='Correct: Dark Matter')
ax3.scatter(new_data[correct & (new_data['Predicted_Label'] == 0)]['Observed_Eps1'],
            new_data[correct & (new_data['Predicted_Label'] == 0)]['Observed_Eps2'],
            c='lightgreen', s=60, alpha=0.8, marker='o', 
            edgecolors='darkgreen', linewidths=2, label='Correct: Background')
# Incorrect predictions
ax3.scatter(new_data[incorrect]['Observed_Eps1'],
            new_data[incorrect]['Observed_Eps2'],
            c='red', s=80, alpha=0.9, marker='X', 
            edgecolors='darkred', linewidths=2, label='Incorrect Predictions')
ax3.set_xlabel('Observed Epsilon 1', fontsize=12, fontweight='bold')
ax3.set_ylabel('Observed Epsilon 2', fontsize=12, fontweight='bold')
ax3.set_title(f'Prediction Accuracy\n(Green=Correct, Red=Incorrect, Accuracy={accuracy*100:.1f}%)', 
             fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, '3_prediction_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 3_prediction_accuracy.png")

# Figure 4: Confusion Matrix
print("Creating Figure 4: Confusion Matrix...")
from matplotlib.colors import ListedColormap
fig4, ax4 = plt.subplots(figsize=(12, 10))
im4 = ax4.imshow(cm, interpolation='nearest', cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=cm.max())
ax4.set_title('Confusion Matrix\n(Model Performance Summary)', fontsize=18, fontweight='bold', pad=20)
ax4.set_ylabel('Ground Truth (Actual)', fontsize=16, fontweight='bold')
ax4.set_xlabel('Predicted (Model Output)', fontsize=16, fontweight='bold')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Background\n(No Dark Matter)', 'Dark Matter\n(Present)'], fontsize=14, fontweight='bold')
ax4.set_yticklabels(['Background\n(No Dark Matter)', 'Dark Matter\n(Present)'], fontsize=14, fontweight='bold')

# Add text annotations with percentages
total = cm.sum()
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        percentage = (count / total * 100) if total > 0 else 0
        ax4.text(j, i, f'{count}\n({percentage:.1f}%)',
               ha="center", va="center",
               color="white" if cm[i, j] > thresh else "black",
               fontsize=20, fontweight='bold')

# Add labels for TN, FP, FN, TP with explanations
ax4.text(0, -0.5, 'True Negative (TN)\nCorrectly identified background', 
        ha='center', fontsize=12, color='darkgreen', fontweight='bold')
ax4.text(1, -0.5, 'False Positive (FP)\nIncorrectly predicted dark matter', 
        ha='center', fontsize=12, color='orange', fontweight='bold')
ax4.text(-0.5, 0, 'True Negative (TN)\nCorrectly identified background', 
        ha='center', fontsize=12, color='darkgreen', fontweight='bold', rotation=90)
ax4.text(-0.5, 1, 'False Negative (FN)\nMissed dark matter', 
        ha='center', fontsize=12, color='red', fontweight='bold', rotation=90)

# Add True Positive label
ax4.text(1, 1, 'True Positive (TP)\nCorrectly identified dark matter', 
        ha='left', fontsize=12, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.colorbar(im4, ax=ax4, label='Number of Galaxies', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_dir, '4_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 4_confusion_matrix.png")

print("\nAll visualizations saved as separate figure files!")

# Print summary statistics
print("\n" + "="*60)
print("Visualization Summary")
print("="*60)
print(f"Total new samples: {len(new_data)}")
print(f"\nGround Truth:")
print(f"  Dark Matter (Label=1): {(new_data['Ground_Truth_Label'] == 1).sum()}")
print(f"  Background (Label=0): {(new_data['Ground_Truth_Label'] == 0).sum()}")
print(f"\nModel Predictions:")
print(f"  Dark Matter (Label=1): {(new_data['Predicted_Label'] == 1).sum()}")
print(f"  Background (Label=0): {(new_data['Predicted_Label'] == 0).sum()}")
print(f"\nPerformance Metrics:")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  True Positives (TP): {cm[1,1]}")
print(f"  True Negatives (TN): {cm[0,0]}")
print(f"  False Positives (FP): {cm[0,1]}")
print(f"  False Negatives (FN): {cm[1,0]}")
print(f"  Average Dark Matter Probability: {new_data['Dark_Matter_Probability'].mean():.3f}")
print("="*60)

