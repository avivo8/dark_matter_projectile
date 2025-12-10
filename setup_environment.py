#!/usr/bin/env python3
"""
Environment setup script for PyTorch and Qiskit.
This script checks for required packages and provides installation instructions.
"""

import sys
import subprocess


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Main function to check and set up the environment."""
    print("=" * 60)
    print("PyTorch and Qiskit Environment Setup")
    print("=" * 60)
    
    # Check PyTorch
    print("\n[1/2] Checking PyTorch...")
    if check_package("torch", "torch"):
        import torch
        print(f"✓ PyTorch is installed (version: {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ PyTorch is not installed")
        print("  Install with: pip install torch")
        print("  Or for CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Check Qiskit
    print("\n[2/2] Checking Qiskit...")
    if check_package("qiskit", "qiskit"):
        import qiskit
        print(f"✓ Qiskit is installed (version: {qiskit.__version__})")
        
        # Check Qiskit components
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            print("  Core components: ✓")
        except ImportError as e:
            print(f"  Core components: ✗ ({e})")
    else:
        print("✗ Qiskit is not installed")
        print("  Install with: pip install qiskit")
        print("  Or for full installation: pip install qiskit[visualization]")
    
    print("\n" + "=" * 60)
    print("Environment check complete!")
    print("=" * 60)
    
    # Example usage
    print("\nExample usage:")
    print("-" * 60)
    
    if check_package("torch", "torch") and check_package("qiskit", "qiskit"):
        print("\n# PyTorch example:")
        print("import torch")
        print("x = torch.tensor([1.0, 2.0, 3.0])")
        print("print(x)")
        
        print("\n# Qiskit example:")
        print("from qiskit import QuantumCircuit")
        print("from qiskit_aer import AerSimulator")
        print("qc = QuantumCircuit(2)")
        print("qc.h(0)")
        print("qc.cx(0, 1)")
        print("simulator = AerSimulator()")
        print("print(qc)")
    else:
        print("\nPlease install missing packages before running examples.")


if __name__ == "__main__":
    main()

