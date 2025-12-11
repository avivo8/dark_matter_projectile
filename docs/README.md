# Dark Matter Halo Detection using Variational Quantum Classifier (VQC)

A quantum machine learning project that uses a Variational Quantum Classifier (VQC) to detect dark matter halos by analyzing galaxy ellipticity measurements. The project simulates gravitational lensing effects and uses quantum computing techniques to classify galaxies based on their observed ellipticity patterns.

## 🌌 Overview

This project demonstrates how quantum machine learning can be applied to astrophysical problems, specifically the detection of dark matter through weak gravitational lensing. Dark matter halos cause gravitational shear, which distorts the shapes of background galaxies. By analyzing these distortions (measured as ellipticity), we can identify regions with dark matter concentrations.

## ✨ Features

- **Quantum Machine Learning**: Uses Qiskit's Variational Quantum Classifier (VQC) for classification
- **Synthetic Data Generation**: Generates realistic galaxy ellipticity datasets with known dark matter labels
- **Spatial Visualization**: Creates 2D spatial maps showing dark matter distribution and prediction accuracy
- **Performance Analysis**: Comprehensive visualization comparing model predictions with ground truth
- **PyTorch Integration**: Environment setup includes PyTorch for potential hybrid classical-quantum approaches

## 📋 Requirements

- Python 3.10+
- Qiskit 1.4.5+
- Qiskit Machine Learning 0.8.4+
- NumPy, Pandas, Scikit-learn
- Matplotlib

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dark_energy_density_proj.git
cd dark_energy_density_proj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify the environment:
```bash
python3 setup_environment.py
```

## 📖 Usage

### 1. Generate Dataset

Generate a labeled dataset of galaxy ellipticity measurements:

```bash
python3 generate_dark_matter_dataset.py
```

This creates `dark_matter_dataset.csv` with:
- `Observed_Eps1`, `Observed_Eps2`: Observed ellipticity components
- `Total_Shear_Gamma`: Total gravitational shear magnitude
- `Label`: Binary label (1 = Dark Matter present, 0 = Background)

### 2. Train the Quantum Model

Train the Variational Quantum Classifier:

```bash
python3 train_model.py
```

This will:
- Load and preprocess the dataset
- Train a VQC model with ZZFeatureMap and RealAmplitudes ansatz
- Save the trained model and scaler for later use
- Display training accuracy

### 3. Visualize Results

Generate comprehensive visualizations:

```bash
python3 visualize_dark_matter.py
```

This creates:
- **Probability heatmaps** showing dark matter concentration
- **Spatial maps** with x-y coordinates showing prediction accuracy
- **Ground truth comparisons** 
- **Confusion matrix** analysis

The visualization is saved as `dark_matter_visualization.png`.

## 📊 Project Structure

```
dark_energy_density_proj/
├── generate_dark_matter_dataset.py  # Dataset generation script
├── train_model.py                    # VQC training script
├── visualize_dark_matter.py          # Visualization script
├── setup_environment.py               # Environment verification
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── dark_matter_dataset.csv            # Generated dataset (after running generator)
```

## 🔬 Methodology

### Data Generation

The dataset simulates:
- **Intrinsic Ellipticity**: Random values between -0.05 and +0.05
- **Gravitational Shear**: 
  - Strong shear (γ > 0.05) for lensed galaxies (dark matter present)
  - Weak/zero shear for background galaxies
- **Observed Ellipticity**: ε_obs = ε_intrinsic + γ

### Quantum Model

- **Feature Map**: ZZFeatureMap with 2 repetitions
- **Ansatz**: RealAmplitudes with 3 repetitions
- **Optimizer**: COBYLA (100 iterations)
- **Loss Function**: Cross-entropy

### Classification

Binary classification:
- **Label = 1**: Dark Matter detected (γ_tot > 0.05)
- **Label = 0**: Background galaxy (γ_tot ≤ 0.05)

## 📈 Results

The model typically achieves:
- **Training Accuracy**: ~90% on test set
- **Spatial Visualization**: Shows clear separation between dark matter regions and background
- **Quantum Advantage**: Demonstrates quantum machine learning capabilities for astrophysical classification

## 🎯 Visualizations

The visualization script generates a 2×3 grid showing:

1. **Model Predictions** (Probability Heatmap)
2. **Ground Truth Labels** (Actual Dark Matter)
3. **Confusion Matrix**
4. **Prediction Accuracy** (Feature Space)
5. **Spatial Map: Prediction Accuracy** (NEW - x-y coordinates)
6. **Spatial Map: Ground Truth vs Predictions** (NEW - overlay comparison)

## 🔧 Configuration

Key parameters can be adjusted in the scripts:

- `n_samples`: Number of galaxy samples (default: 200)
- `reps_feature`: Feature map repetitions (default: 2)
- `reps_ansatz`: Ansatz repetitions (default: 3)
- `maxiter`: Optimizer iterations (default: 100)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Qiskit team for quantum computing framework
- Astrophysics community for gravitational lensing theory
- Quantum machine learning research community

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a research/educational project demonstrating quantum machine learning applications in astrophysics. For production use, additional validation and optimization would be required.


