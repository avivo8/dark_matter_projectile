#!/usr/bin/env python3
"""
Generate a labeled dataset for training a Variational Quantum Classifier (VQC)
to detect Dark Matter halos using galaxy ellipticity measurements.
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 200
n_lensed = 100  # Galaxies with Dark Matter (stronger shear)
n_background = 100  # Background galaxies (weaker/zero shear)

# Initialize lists to store data
observed_eps1 = []
observed_eps2 = []
total_shear_gamma = []
labels = []

# Generate intrinsic ellipticity for all samples (same distribution)
epsilon_1_int = np.random.uniform(-0.05, 0.05, n_samples)
epsilon_2_int = np.random.uniform(-0.05, 0.05, n_samples)

# Generate lensed galaxies (stronger shear)
# Make gamma values larger so gamma_tot > 0.05
# Using [0.05, 0.08] ensures minimum gamma_tot = sqrt(0.05^2 + 0.05^2) ≈ 0.071 > 0.05
gamma_1_lensed = np.random.uniform(0.05, 0.08, n_lensed)
gamma_2_lensed = np.random.uniform(0.05, 0.08, n_lensed)

# Generate background galaxies (weaker/zero shear)
# Make gamma values smaller so gamma_tot <= 0.05
# Using [-0.03, 0.03] ensures maximum gamma_tot = sqrt(0.03^2 + 0.03^2) ≈ 0.042 < 0.05
gamma_1_background = np.random.uniform(-0.03, 0.03, n_background)
gamma_2_background = np.random.uniform(-0.03, 0.03, n_background)

# Combine all gamma values
gamma_1 = np.concatenate([gamma_1_lensed, gamma_1_background])
gamma_2 = np.concatenate([gamma_2_lensed, gamma_2_background])

# Shuffle the data to mix lensed and background galaxies
indices = np.random.permutation(n_samples)
epsilon_1_int = epsilon_1_int[indices]
epsilon_2_int = epsilon_2_int[indices]
gamma_1 = gamma_1[indices]
gamma_2 = gamma_2[indices]

# Process each sample
for i in range(n_samples):
    # Compute total shear
    gamma_tot = np.sqrt(gamma_1[i]**2 + gamma_2[i]**2)
    
    # Compute observed ellipticity (epsilon_obs = epsilon_int + gamma)
    eps_obs_1 = epsilon_1_int[i] + gamma_1[i]
    eps_obs_2 = epsilon_2_int[i] + gamma_2[i]
    
    # Create binary label: Y = 1 if gamma_tot > 0.05, else 0
    label = 1 if gamma_tot > 0.05 else 0
    
    # Store values
    observed_eps1.append(eps_obs_1)
    observed_eps2.append(eps_obs_2)
    total_shear_gamma.append(gamma_tot)
    labels.append(label)

# Create pandas DataFrame
df = pd.DataFrame({
    'Observed_Eps1': observed_eps1,
    'Observed_Eps2': observed_eps2,
    'Total_Shear_Gamma': total_shear_gamma,
    'Label': labels
})

import os
# Get the project root directory (parent of src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df.to_csv(os.path.join(project_root, 'data', 'dark_matter_dataset.csv'), index=False)

# Print first 5 rows
print("Dataset for Dark Matter Halo Detection (VQC Training)")
print("=" * 60)
print(f"\nTotal samples: {len(df)}")
print(f"Lensed galaxies (Label=1): {df['Label'].sum()}")
print(f"Background galaxies (Label=0): {(df['Label'] == 0).sum()}")
print("\nFirst 5 rows of the DataFrame:")
print("-" * 60)
print(df.head())
print("\n" + "=" * 60)
print("Dataset generation complete!")

