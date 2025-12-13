#!/usr/bin/env python3
"""
Compare validation results between original and improved VQR models.
Creates side-by-side comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')
vis_dir = os.path.join(project_root, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

print("=" * 70)
print("Model Comparison: Original vs Improved VQR")
print("=" * 70)

# Load data
print("\n1. Loading data...")
df_original = pd.read_csv(os.path.join(data_dir, 'concentration_validation.csv'))
df_improved = pd.read_csv(os.path.join(data_dir, 'sdss_predictions_with_improved.csv'))

# Extract values
log10_m_orig = df_original['log10_M_halo'].values
log10_c_ref = df_original['log10_c_ref'].values
log10_c_pred_orig = df_original['log10_c_pred'].values
log10_c_pred_improved = df_improved['log10_c_pred_improved'].values

# Calculate metrics
mse_orig = np.mean((log10_c_pred_orig - log10_c_ref)**2)
mse_improved = np.mean((log10_c_pred_improved - log10_c_ref)**2)

bias_orig = np.mean(log10_c_pred_orig - log10_c_ref)
bias_improved = np.mean(log10_c_pred_improved - log10_c_ref)

resid_orig = log10_c_pred_orig - log10_c_ref
resid_improved = log10_c_pred_improved - log10_c_ref

print(f"   ✓ Original model:")
print(f"     MSE:  {mse_orig:.6f}")
print(f"     Bias: {bias_orig:+.6f}")
print(f"   ✓ Improved model:")
print(f"     MSE:  {mse_improved:.6f}")
print(f"     Bias: {bias_improved:+.6f}")

# ============================================================================
# Plot 1: Predicted vs Reference (Side-by-side)
# ============================================================================
print("\n2. Creating comparison plots...")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Original model
ax1.scatter(log10_c_ref, log10_c_pred_orig, s=30, alpha=0.6, c='steelblue', edgecolors='none')
ax1.plot([log10_c_ref.min(), log10_c_ref.max()], 
         [log10_c_ref.min(), log10_c_ref.max()], 
         'r--', linewidth=2, label='y=x')
ax1.set_xlabel(r'$\log_{10} c_{\mathrm{ref}}$ (Reference)', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'$\log_{10} c_{\mathrm{pred}}$ (Original)', fontsize=12, fontweight='bold')
ax1.set_title('Original Model\nPredicted vs Reference', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.text(0.05, 0.95, f'MSE: {mse_orig:.6f}\nBias: {bias_orig:+.6f}',
         transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Improved model
ax2.scatter(log10_c_ref, log10_c_pred_improved, s=30, alpha=0.6, c='darkgreen', edgecolors='none')
ax2.plot([log10_c_ref.min(), log10_c_ref.max()], 
         [log10_c_ref.min(), log10_c_ref.max()], 
         'r--', linewidth=2, label='y=x')
ax2.set_xlabel(r'$\log_{10} c_{\mathrm{ref}}$ (Reference)', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'$\log_{10} c_{\mathrm{pred}}$ (Improved)', fontsize=12, fontweight='bold')
ax2.set_title('Improved Model (Phase 2 & 3)\nPredicted vs Reference', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.text(0.05, 0.95, f'MSE: {mse_improved:.6f}\nBias: {bias_improved:+.6f}',
         transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
out1 = os.path.join(vis_dir, 'concentration_comparison_pred_vs_ref.png')
fig1.savefig(out1, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {out1}")
plt.close(fig1)

# ============================================================================
# Plot 2: Residuals vs Mass (Side-by-side)
# ============================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Original model
ax1.scatter(log10_m_orig, resid_orig, s=30, alpha=0.6, c='steelblue', edgecolors='none')
ax1.axhline(0.0, color='black', linestyle='--', linewidth=2)
ax1.set_xlabel(r'$\log_{10} M_{\mathrm{halo}}$', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'$\log_{10} c_{\mathrm{pred}} - \log_{10} c_{\mathrm{ref}}$', fontsize=12, fontweight='bold')
ax1.set_title('Original Model\nResiduals vs Halo Mass', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'Bias: {bias_orig:+.6f}',
         transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Improved model
ax2.scatter(log10_m_orig, resid_improved, s=30, alpha=0.6, c='darkgreen', edgecolors='none')
ax2.axhline(0.0, color='black', linestyle='--', linewidth=2)
ax2.set_xlabel(r'$\log_{10} M_{\mathrm{halo}}$', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'$\log_{10} c_{\mathrm{pred}} - \log_{10} c_{\mathrm{ref}}$', fontsize=12, fontweight='bold')
ax2.set_title('Improved Model (Phase 2 & 3)\nResiduals vs Halo Mass', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, f'Bias: {bias_improved:+.6f}',
         transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
out2 = os.path.join(vis_dir, 'concentration_comparison_residual_vs_mass.png')
fig2.savefig(out2, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {out2}")
plt.close(fig2)

# ============================================================================
# Plot 3: Direct Comparison - Residuals Overlay
# ============================================================================
fig3, ax = plt.subplots(figsize=(12, 7))

ax.scatter(log10_m_orig, resid_orig, s=40, alpha=0.5, c='steelblue', 
           edgecolors='none', label='Original Model', marker='o')
ax.scatter(log10_m_orig, resid_improved, s=40, alpha=0.5, c='darkgreen', 
           edgecolors='none', label='Improved Model (Phase 2 & 3)', marker='s')
ax.axhline(0.0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel(r'$\log_{10} M_{\mathrm{halo}}$', fontsize=14, fontweight='bold')
ax.set_ylabel(r'$\log_{10} c_{\mathrm{pred}} - \log_{10} c_{\mathrm{ref}}$', fontsize=14, fontweight='bold')
ax.set_title('Residual Comparison: Original vs Improved Model', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, framealpha=0.9)

# Add metrics text
metrics_text = f'Original:  MSE={mse_orig:.6f}, Bias={bias_orig:+.6f}\n'
metrics_text += f'Improved: MSE={mse_improved:.6f}, Bias={bias_improved:+.6f}\n'
metrics_text += f'\nImprovement:\n'
metrics_text += f'  MSE reduction: {(mse_orig - mse_improved) / mse_orig * 100:.1f}%\n'
metrics_text += f'  Bias reduction: {abs(bias_orig) - abs(bias_improved):.6f}'

ax.text(0.02, 0.98, metrics_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()
out3 = os.path.join(vis_dir, 'concentration_comparison_residuals_overlay.png')
fig3.savefig(out3, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {out3}")
plt.close(fig3)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"\nOriginal Model:")
print(f"  MSE:  {mse_orig:.6f}")
print(f"  Bias: {bias_orig:+.6f}")
print(f"  RMSE: {np.sqrt(mse_orig):.6f}")

print(f"\nImproved Model (Phase 2 & 3):")
print(f"  MSE:  {mse_improved:.6f}")
print(f"  Bias: {bias_improved:+.6f}")
print(f"  RMSE: {np.sqrt(mse_improved):.6f}")

print(f"\nImprovements:")
print(f"  MSE reduction:  {(mse_orig - mse_improved) / mse_orig * 100:.1f}%")
print(f"  RMSE reduction: {(np.sqrt(mse_orig) - np.sqrt(mse_improved)) / np.sqrt(mse_orig) * 100:.1f}%")
print(f"  Bias reduction: {abs(bias_orig) - abs(bias_improved):.6f}")
print(f"  Bias improvement: {abs(bias_orig) / abs(bias_improved):.1f}x better")

print(f"\nVisualizations saved:")
print(f"  1. {out1}")
print(f"  2. {out2}")
print(f"  3. {out3}")
print("=" * 70)

