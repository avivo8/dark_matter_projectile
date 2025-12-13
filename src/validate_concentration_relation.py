#!/usr/bin/env python3
"""
Action 3.1 + 3.2: External validation baseline for halo concentration.

This script computes an external reference mass–concentration relation c_ref(M_halo)
from the literature, then compares model-predicted concentrations (c_pred) against
that baseline using MSE and bias, plus two diagnostic plots.

Reference relation (default):
  Duffy et al. (2008), MNRAS 390, L64–L68 (commonly used N-body calibration)
  For "all halos", Delta=200 (roughly M200), they provide:
      c200 = A * (M200 / (2e12 h^-1 Msun))^B * (1+z)^C
  with (A, B, C) depending on halo definition/sample. We use the widely quoted
  values for "all halos" as a baseline. This is a *baseline sanity check*,
  not a precision lensing calibration.

Expected input columns in the CSV:
  - log10_M_halo : log10(M_halo) in Msun/h (M200 preferred if using the default relation)
  - log10_c_pred : log10(c_pred) from your VQR/VQC regression pipeline
Optional:
  - z            : redshift (float). If absent, assumed z=0.

Outputs:
  - data/concentration_validation.csv (augmented table)
  - visualizations/concentration_pred_vs_ref.png
  - visualizations/concentration_residual_vs_mass.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Duffy2008Params:
    """Parameters for c(M,z) = A * (M/M_pivot)^B * (1+z)^C"""

    A: float = 5.71
    B: float = -0.084
    C: float = -0.47
    # M_pivot in Msun/h
    M_pivot: float = 2e12


def log10_c_ref_duffy2008(log10_m_halo: np.ndarray, z: np.ndarray, p: Duffy2008Params) -> np.ndarray:
    """
    Compute log10(c_ref) from Duffy+2008-style power law.

    Inputs are arrays:
      - log10_m_halo: log10(M_halo [Msun/h])
      - z: redshift
    """
    m = np.power(10.0, log10_m_halo)
    c = p.A * np.power(m / p.M_pivot, p.B) * np.power(1.0 + z, p.C)
    return np.log10(c)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


def bias(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a - b))


def main() -> int:
    parser = argparse.ArgumentParser(description="External validation baseline: c–M relation comparison.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input CSV containing log10_M_halo and log10_c_pred. Default: data/sdss_predictions.csv",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Where to write augmented CSV. Default: data/concentration_validation.csv",
    )
    parser.add_argument(
        "--z-col",
        default="z",
        help="Name of redshift column (optional). Default: z",
    )
    parser.add_argument(
        "--mass-col",
        default="log10_M_halo",
        help="Name of halo mass column (required). Default: log10_M_halo",
    )
    parser.add_argument(
        "--pred-col",
        default="log10_c_pred",
        help="Name of predicted concentration column (required). Default: log10_c_pred",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    vis_dir = os.path.join(project_root, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    input_path = args.input or os.path.join(data_dir, "sdss_predictions.csv")
    output_csv = args.output_csv or os.path.join(data_dir, "concentration_validation.csv")

    df = pd.read_csv(input_path)
    required = [args.mass_col, args.pred_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nProvide them in the input CSV:\n"
            + f"  - {args.mass_col}: log10(M_halo) in Msun/h\n"
            + f"  - {args.pred_col}: log10(c_pred)\n"
            + "Optional:\n"
            + f"  - {args.z_col}: redshift (defaults to 0 if missing)\n"
        )

    log10_m = df[args.mass_col].to_numpy(dtype=float)
    log10_c_pred = df[args.pred_col].to_numpy(dtype=float)
    if args.z_col in df.columns:
        z = df[args.z_col].to_numpy(dtype=float)
    else:
        z = np.zeros_like(log10_m)

    # Compute reference concentrations
    params = Duffy2008Params()
    log10_c_ref = log10_c_ref_duffy2008(log10_m, z, params)

    # Metrics
    mse_val = mse(log10_c_pred, log10_c_ref)
    bias_val = bias(log10_c_pred, log10_c_ref)
    resid = log10_c_pred - log10_c_ref

    # Store for later comparison
    df = df.copy()
    df["log10_c_ref"] = log10_c_ref
    df["c_ref_source"] = "Duffy+2008 (all halos, ~M200)"
    df["c_ref_A"] = params.A
    df["c_ref_B"] = params.B
    df["c_ref_C"] = params.C
    df["c_ref_M_pivot_Msun_h"] = params.M_pivot
    df["residual_log10c_pred_minus_ref"] = resid
    df.to_csv(output_csv, index=False)

    # Plot 1: Predicted vs Reference
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(log10_c_ref, log10_c_pred, s=28, alpha=0.7, edgecolors="none")
    lo = float(min(np.min(log10_c_ref), np.min(log10_c_pred)))
    hi = float(max(np.max(log10_c_ref), np.max(log10_c_pred)))
    pad = 0.05 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad
    ax1.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5, label="y = x")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel(r"$\log_{10} c_{\mathrm{ref}}$", fontweight="bold")
    ax1.set_ylabel(r"$\log_{10} c_{\mathrm{pred}}$", fontweight="bold")
    ax1.set_title("Predicted vs Reference Concentration", fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.text(
        0.02,
        0.02,
        f"N={len(df)}\nMSE={mse_val:.4f}\nBias={bias_val:+.4f}",
        transform=ax1.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )
    out1 = os.path.join(vis_dir, "concentration_pred_vs_ref.png")
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Residuals vs Halo Mass
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(log10_m, resid, s=28, alpha=0.7, edgecolors="none")
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax2.set_xlabel(r"$\log_{10} M_{\mathrm{halo}}$", fontweight="bold")
    ax2.set_ylabel(r"$\log_{10} c_{\mathrm{pred}} - \log_{10} c_{\mathrm{ref}}$", fontweight="bold")
    ax2.set_title("Residuals vs Halo Mass", fontweight="bold")
    ax2.grid(True, alpha=0.25)
    ax2.text(
        0.02,
        0.98,
        "Reference: Duffy+2008\n(all halos, ~M200)\n(Assumes Msun/h)",
        transform=ax2.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )
    out2 = os.path.join(vis_dir, "concentration_residual_vs_mass.png")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("Concentration validation complete.")
    print(f"Augmented CSV: {output_csv}")
    print(f"Plot 1: {out1}")
    print(f"Plot 2: {out2}")
    print(f"MSE: {mse_val:.6f}")
    print(f"Bias: {bias_val:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


