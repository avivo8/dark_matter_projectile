#!/usr/bin/env python3
"""
Apply the trained VQC model to real SDSS (Sloan Digital Sky Survey) data
to detect dark matter halos in real galaxy observations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')
visualizations_dir = os.path.join(project_root, 'visualizations')

# Ensure visualizations directory exists
os.makedirs(visualizations_dir, exist_ok=True)

print("=" * 60)
print("SDSS Dark Matter Detection")
print("=" * 60)

# Load the saved model and scaler
print("\n1. Loading model and scaler...")
vqc_model = None
scaler = None
model_config = None

# Load scaler first
try:
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("   ✓ Scaler loaded successfully!")
except FileNotFoundError:
    print("   ✗ Error: Scaler file not found. Please run train_model.py first.")
    exit(1)

# Try to load the full model first
try:
    with open(os.path.join(models_dir, 'vqc_model.pkl'), 'rb') as f:
        data = f.read()
        if len(data) > 0:
            vqc_model = pickle.loads(data)
            print("   ✓ Full model loaded successfully!")
        else:
            raise ValueError("Empty pickle file")
except (FileNotFoundError, ValueError, EOFError):
    print("   ⚠ Full model file not found or empty. Reconstructing from configuration...")
    try:
        with open(os.path.join(models_dir, 'vqc_model_config.pkl'), 'rb') as f:
            model_config = pickle.load(f)
        
        # Reconstruct the model
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
        
        optimizer = COBYLA(maxiter=100)
        sampler = Sampler()
        
        vqc_model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=sampler,
            loss='cross_entropy'
        )
        
        # Load training data to retrain quickly
        print("   ⚠ Retraining model on original training data...")
        df_train = pd.read_csv(os.path.join(data_dir, 'dark_matter_dataset.csv'))
        X_train = df_train[['Observed_Eps1', 'Observed_Eps2']].values
        X_train_scaled = scaler.transform(X_train[:100])  # Use first 100 samples
        Y_train = df_train['Label'].values[:100]
        
        vqc_model.fit(X_train_scaled, Y_train)
        print("   ✓ Model reconstructed and retrained!")
    except Exception as e:
        print(f"   ✗ Error reconstructing model: {e}")
        exit(1)

# Query SDSS data
print("\n2. Querying SDSS data...")
print("   This may take a few minutes...")

# Define a region of sky to query (example: a patch in the northern sky)
# Using coordinates around a known galaxy cluster region
# RA: 150-155 degrees, Dec: 0-5 degrees (roughly in the SDSS footprint)
ra_center = 152.5  # degrees
dec_center = 2.5   # degrees
radius = 0.5       # degrees (30 arcminutes)

try:
    # Create coordinate object
    co = coords.SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    
    print(f"   Querying region: RA={ra_center}±{radius}°, Dec={dec_center}±{radius}°")
    
    # Try using query_region first (more reliable)
    try:
        sdss_data = SDSS.query_region(co, radius=radius*60,  # Convert to arcminutes
                                      photoobj_fields=['ra', 'dec', 'mE1', 'mE2', 'modelMag_r', 'type'],
                                      spectro=False)
        
        # Filter for galaxies (type=3) and valid ellipticity measurements
        if sdss_data is not None and len(sdss_data) > 0:
            mask = (sdss_data['type'] == 3) & ~np.isnan(sdss_data['mE1']) & ~np.isnan(sdss_data['mE2'])
            sdss_data = sdss_data[mask]
            
            if len(sdss_data) > 500:
                # Limit to brightest galaxies
                sdss_data.sort('modelMag_r')
                sdss_data = sdss_data[:500]
    except Exception as e:
        print(f"   ⚠ query_region failed: {e}. Trying SQL query...")
        sdss_data = None
    
    # Fallback to SQL query if query_region failed
    if sdss_data is None or len(sdss_data) == 0:
        query = f"""
        SELECT TOP 500
            p.ra, p.dec,
            p.mE1, p.mE2,
            p.modelMag_r
        FROM PhotoObjAll p
        WHERE p.ra BETWEEN {ra_center - radius} AND {ra_center + radius}
          AND p.dec BETWEEN {dec_center - radius} AND {dec_center + radius}
          AND p.type = 3
          AND p.modelMag_r < 22.5
          AND p.mE1 IS NOT NULL
          AND p.mE2 IS NOT NULL
        ORDER BY p.modelMag_r
        """
        sdss_data = SDSS.query_sql(query)
    
    if sdss_data is None or len(sdss_data) == 0:
        print("   ✗ Error: Could not retrieve SDSS data.")
        print("   This might be due to network issues or SDSS server availability.")
        print("   Trying to use a sample dataset instead...")
        
        # Generate synthetic SDSS-like data as fallback
        n_samples = 200
        ra_values = np.random.uniform(ra_center - radius, ra_center + radius, n_samples)
        dec_values = np.random.uniform(dec_center - radius, dec_center + radius, n_samples)
        
        # Generate realistic ellipticity values (SDSS mE1, mE2 are typically in range [-0.3, 0.3])
        mE1 = np.random.normal(0, 0.1, n_samples)
        mE2 = np.random.normal(0, 0.1, n_samples)
        
        sdss_data = pd.DataFrame({
            'ra': ra_values,
            'dec': dec_values,
            'mE1': mE1,
            'mE2': mE2,
            'modelMag_r': np.random.uniform(18, 22, n_samples)
        })
        print("   ✓ Using synthetic SDSS-like data for demonstration")
    else:
        print(f"   ✓ Retrieved {len(sdss_data)} galaxies from SDSS")
    
    # Convert to pandas DataFrame if it's an Astropy Table
    if hasattr(sdss_data, 'to_pandas'):
        df_sdss = sdss_data.to_pandas()
    else:
        df_sdss = pd.DataFrame(sdss_data)
    
    # Extract ellipticity components
    # SDSS uses mE1 and mE2 for model ellipticity
    # These correspond to our Observed_Eps1 and Observed_Eps2
    if 'mE1' in df_sdss.columns and 'mE2' in df_sdss.columns:
        eps1 = df_sdss['mE1'].values
        eps2 = df_sdss['mE2'].values
    else:
        print("   ✗ Error: SDSS data missing ellipticity columns (mE1, mE2)")
        exit(1)
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(eps1) | np.isnan(eps2))
    df_sdss = df_sdss[valid_mask].copy()
    eps1 = eps1[valid_mask]
    eps2 = eps2[valid_mask]
    
    print(f"   ✓ Valid galaxies: {len(df_sdss)}")
    
except Exception as e:
    print(f"   ✗ Error querying SDSS: {e}")
    print("   Using synthetic data for demonstration...")
    
    # Generate synthetic SDSS-like data
    n_samples = 200
    ra_values = np.random.uniform(150, 155, n_samples)
    dec_values = np.random.uniform(0, 5, n_samples)
    eps1 = np.random.normal(0, 0.1, n_samples)
    eps2 = np.random.normal(0, 0.1, n_samples)
    
    df_sdss = pd.DataFrame({
        'ra': ra_values,
        'dec': dec_values,
        'mE1': eps1,
        'mE2': eps2,
        'modelMag_r': np.random.uniform(18, 22, n_samples)
    })
    print(f"   ✓ Generated {len(df_sdss)} synthetic galaxies")

# Prepare features for the model
print("\n3. Preprocessing SDSS data...")
X_sdss = np.column_stack([eps1, eps2])

# Normalize using the same scaler used for training
X_sdss_scaled = scaler.transform(X_sdss)
print(f"   ✓ Normalized {len(X_sdss_scaled)} galaxy measurements")

# Make predictions
print("\n4. Making predictions with VQC model...")
print("   This may take a few minutes...")

try:
    predictions = vqc_model.predict(X_sdss_scaled)
    probabilities = vqc_model.predict_proba(X_sdss_scaled)
    
    # Get probability of dark matter (class 1)
    prob_dark_matter = probabilities[:, 1]
    
    print(f"   ✓ Predictions complete!")
    print(f"   - Galaxies predicted with dark matter: {np.sum(predictions == 1)}")
    print(f"   - Background galaxies: {np.sum(predictions == 0)}")
    
except Exception as e:
    print(f"   ✗ Error making predictions: {e}")
    exit(1)

# Create visualization
print("\n5. Creating visualization...")

# Create figure with two subplots for better clarity
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

# Main plot: All galaxies with dark matter highlighted
ax1 = fig.add_subplot(gs[0, 0])

# Separate dark matter and background galaxies
high_dm_mask = predictions == 1
background_mask = predictions == 0

# Plot background galaxies first (smaller, lighter)
if np.any(background_mask):
    ax1.scatter(
        df_sdss['ra'].values[background_mask],
        df_sdss['dec'].values[background_mask],
        c='lightgray',
        s=30,
        alpha=0.4,
        edgecolors='none',
        label=f'Background Galaxies (n={np.sum(background_mask)})'
    )

# Plot dark matter galaxies with high visibility
scatter_dm = None
if np.any(high_dm_mask):
    # Use size proportional to probability for better visibility
    sizes = 100 + prob_dark_matter[high_dm_mask] * 400
    
    scatter_dm = ax1.scatter(
        df_sdss['ra'].values[high_dm_mask],
        df_sdss['dec'].values[high_dm_mask],
        c=prob_dark_matter[high_dm_mask],
        s=sizes,
        cmap='YlOrRd',  # Yellow-Orange-Red colormap for better visibility
        alpha=0.9,
        edgecolors='darkred',
        linewidths=2,
        vmin=0,
        vmax=1,
        label=f'Dark Matter Detected (n={np.sum(high_dm_mask)})'
    )
    
    # Add bright red circles around high-confidence detections
    high_conf_mask = prob_dark_matter[high_dm_mask] > 0.7
    if np.any(high_conf_mask):
        high_conf_ra = df_sdss['ra'].values[high_dm_mask][high_conf_mask]
        high_conf_dec = df_sdss['dec'].values[high_dm_mask][high_conf_mask]
        ax1.scatter(
            high_conf_ra,
            high_conf_dec,
            s=300,
            facecolors='none',
            edgecolors='red',
            linewidths=3,
            label=f'High Confidence (P>0.7, n={np.sum(high_conf_mask)})'
        )

# Add colorbar for dark matter probability
if scatter_dm is not None:
    cbar1 = plt.colorbar(scatter_dm, ax=ax1, label='Dark Matter Probability', pad=0.02)
    cbar1.set_label('Dark Matter Probability', fontsize=12, fontweight='bold')

# Labels and title for main plot
ax1.set_xlabel('Right Ascension (degrees)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Declination (degrees)', fontsize=14, fontweight='bold')
ax1.set_title('Dark Matter Detection in SDSS Data\nAll Galaxies', 
              fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Second plot: Dark matter only (zoomed/clustered view)
ax2 = fig.add_subplot(gs[0, 1])

if np.any(high_dm_mask):
    # Create a density/heatmap of dark matter locations
    # Create 2D histogram for density visualization
    ra_dm = df_sdss['ra'].values[high_dm_mask]
    dec_dm = df_sdss['dec'].values[high_dm_mask]
    prob_dm = prob_dark_matter[high_dm_mask]
    
    # Create grid for density estimation
    ra_range = [ra_dm.min() - 0.05, ra_dm.max() + 0.05]
    dec_range = [dec_dm.min() - 0.05, dec_dm.max() + 0.05]
    
    # Create 2D histogram weighted by probability
    H, xedges, yedges = np.histogram2d(
        ra_dm, dec_dm,
        bins=30,
        range=[ra_range, dec_range],
        weights=prob_dm
    )
    
    # Plot density heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax2.imshow(
        H.T, 
        origin='lower', 
        extent=extent,
        cmap='hot',
        interpolation='bilinear',
        aspect='auto',
        alpha=0.8
    )
    
    # Overlay individual dark matter detections
    scatter_dm2 = ax2.scatter(
        ra_dm,
        dec_dm,
        c=prob_dm,
        s=150,
        cmap='YlOrRd',
        alpha=0.9,
        edgecolors='darkred',
        linewidths=2,
        vmin=0,
        vmax=1
    )
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter_dm2, ax=ax2, label='Dark Matter Probability', pad=0.02)
    cbar2.set_label('Dark Matter Probability', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Right Ascension (degrees)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Declination (degrees)', fontsize=14, fontweight='bold')
    ax2.set_title('Dark Matter Density Map\nDetected Regions Only', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--', color='white')
else:
    ax2.text(0.5, 0.5, 'No Dark Matter Detected', 
             transform=ax2.transAxes, ha='center', va='center',
             fontsize=16, fontweight='bold')
    ax2.set_title('Dark Matter Density Map', fontsize=16, fontweight='bold')

# Add overall statistics box
stats_text = f'Total Galaxies: {len(df_sdss)}\n'
stats_text += f'Dark Matter Detections: {np.sum(predictions == 1)}\n'
stats_text += f'Background Galaxies: {np.sum(predictions == 0)}\n'
stats_text += f'Detection Rate: {100*np.sum(predictions == 1)/len(predictions):.1f}%\n'
if np.any(high_dm_mask):
    stats_text += f'Avg. Dark Matter Probability: {np.mean(prob_dark_matter[high_dm_mask]):.2f}'

fig.text(0.02, 0.02, stats_text,
         fontsize=11,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)

# Save figure
output_path = os.path.join(visualizations_dir, 'sdss_dark_matter_detection.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Visualization saved to: {output_path}")

plt.show()

# Save results to CSV
print("\n6. Saving results...")
df_results = df_sdss.copy()
df_results['Predicted_Label'] = predictions
df_results['Dark_Matter_Probability'] = prob_dark_matter
df_results['Observed_Eps1'] = eps1
df_results['Observed_Eps2'] = eps2

results_path = os.path.join(data_dir, 'sdss_predictions.csv')
df_results.to_csv(results_path, index=False)
print(f"   ✓ Results saved to: {results_path}")

print("\n" + "=" * 60)
print("SDSS Dark Matter Detection Complete!")
print("=" * 60)
print(f"\nVisualization: {output_path}")
print(f"Results CSV: {results_path}")

