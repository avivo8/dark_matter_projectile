# Dark Matter Halo Detection using Variational Quantum Classifier (VQC)

A quantum machine learning project that uses a Variational Quantum Classifier (VQC) to detect dark matter halos by analyzing galaxy ellipticity measurements. The project simulates gravitational lensing effects and uses quantum computing techniques to classify galaxies based on their observed ellipticity patterns.

## 🌌 Overview

This project demonstrates how quantum machine learning can be applied to astrophysical problems, specifically the detection of dark matter through weak gravitational lensing. Dark matter halos cause gravitational shear, which distorts the shapes of background galaxies. By analyzing these distortions (measured as ellipticity), we can identify regions with dark matter concentrations.

## 📁 Project Structure

```
dark_matter_projectile/
├── src/                    # Python source code
│   ├── generate_dark_matter_dataset.py
│   ├── train_model.py
│   ├── visualize_dark_matter.py
│   └── setup_environment.py
├── data/                   # Data files
│   └── dark_matter_dataset.csv
├── models/                 # Trained models and scalers
│   ├── vqc_model.pkl
│   ├── vqc_model_config.pkl
│   └── scaler.pkl
├── visualizations/         # Generated visualization images
│   ├── 1_model_predictions.png
│   ├── 2_ground_truth_labels.png
│   ├── 3_prediction_accuracy.png
│   └── 4_confusion_matrix.png
├── website/                # Website files
│   ├── index.html
│   ├── interactive.html   # Interactive dark matter detection
│   ├── styles.css
│   ├── script.js
│   └── interactive.js     # Interactive feature JavaScript
├── docs/                   # Documentation
│   ├── README.md (detailed)
│   ├── WEBSITE_README.md
│   └── PUSH_TO_GITHUB.md
├── python-requirements.txt        # Python dependencies
└── README.md              # This file
```

## ✨ Features

- **Quantum Machine Learning**: Uses Qiskit's Variational Quantum Classifier (VQC) for classification
- **Synthetic Data Generation**: Generates realistic galaxy ellipticity datasets with known dark matter labels
- **Spatial Visualization**: Creates 2D spatial maps showing dark matter distribution and prediction accuracy
- **Performance Analysis**: Comprehensive visualization comparing model predictions with ground truth
- **Modern Website**: Beautiful, responsive website showcasing the project

## 📋 Requirements

- Python 3.10+
- Qiskit 1.4.5+
- Qiskit Machine Learning 0.8.4+
- NumPy, Pandas, Scikit-learn
- Matplotlib

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/avivo8/dark_matter_projectile.git
cd dark_matter_projectile
```

2. Install dependencies:
```bash
pip install -r python-requirements.txt
```

3. Verify the environment:
```bash
python3 src/setup_environment.py
```

## 📖 Usage

### 1. Generate Dataset

Generate a labeled dataset of galaxy ellipticity measurements:

```bash
python3 src/generate_dark_matter_dataset.py
```

This creates `data/dark_matter_dataset.csv` with:
- `Observed_Eps1`, `Observed_Eps2`: Observed ellipticity components
- `Total_Shear_Gamma`: Total gravitational shear magnitude
- `Label`: Binary label (1 = Dark Matter present, 0 = Background)

### 2. Train the Quantum Model

Train the Variational Quantum Classifier:

```bash
python3 src/train_model.py
```

This will:
- Load and preprocess the dataset
- Train a VQC model with ZZFeatureMap and RealAmplitudes ansatz
- Save the trained model and scaler to `models/` directory
- Display training accuracy

### 3. Visualize Results

Generate comprehensive visualizations:

```bash
python3 src/visualize_dark_matter.py
```

This creates visualization images in `visualizations/` directory:
- **Probability heatmaps** showing dark matter concentration
- **Ground truth comparisons** 
- **Confusion matrix** analysis
- **Prediction accuracy** visualizations

### 4. View Website

Open `website/index.html` in your browser or use a local server:

```bash
cd website
python3 -m http.server 8000
# Then visit: http://localhost:8000
```

### 5. Interactive Dark Matter Detection

Use the interactive feature to upload galaxy images and get predictions:

1. Start the API server:
```bash
python3 src/api_server.py
```

2. In another terminal, start the web server:
```bash
cd website
python3 -m http.server 8000
```

3. Open `http://localhost:8000/interactive.html` in your browser

4. Upload a galaxy grid image, mark dark matter regions, and get predictions!

See `docs/INTERACTIVE_GUIDE.md` for detailed instructions.

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

The visualization script generates separate figure files:

1. **Model Predictions** (Probability Heatmap)
2. **Ground Truth Labels** (Actual Dark Matter)
3. **Confusion Matrix**
4. **Prediction Accuracy** (Feature Space)

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

## 👤 Author

**Aviv Solan**

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a research/educational project demonstrating quantum machine learning applications in astrophysics. For production use, additional validation and optimization would be required.

