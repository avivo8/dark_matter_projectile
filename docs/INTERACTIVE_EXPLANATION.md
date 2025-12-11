# Interactive Dark Matter Detection - Complete Explanation

## What is the Interactive Feature?

The interactive feature is a **web-based demonstration** that lets you:
1. **Visualize** a galaxy field image
2. **Manually mark** which galaxies you think have dark matter nearby
3. **See predictions** from the quantum machine learning model
4. **Compare** your markings with the model's predictions

It's like a **hands-on demonstration** of how astronomers use quantum ML to detect dark matter through gravitational lensing!

---

## How It Works - Step by Step

### Step 1: Image Selection
**What happens:**
- You choose one of 3 pre-generated example galaxy field images
- OR upload your own galaxy image
- The image is displayed on a canvas

**Behind the scenes:**
- Example images are generated programmatically with:
  - Dark space background with stars
  - 30-40 synthetic galaxies
  - Some galaxies are distorted (simulating gravitational lensing)
  - Some galaxies are normal (background)

### Step 2: Marking Galaxies
**What you do:**
- Click on galaxies to mark them
- Switch between two modes:
  - **"Mark Dark Matter"** (red markers) - for galaxies you think have dark matter nearby
  - **"Mark Background"** (blue markers) - for normal background galaxies

**What happens:**
- Each click creates a marker at that location
- The system stores the X, Y coordinates
- Statistics update showing how many you've marked

**Why this matters:**
- You're creating **ground truth labels** - your expert opinion about where dark matter is
- This simulates what astronomers do when analyzing real telescope data

### Step 3: Generate Galaxy Grid
**What happens when you click "Generate Galaxy Grid":**

1. **Feature Extraction:**
   - For each marked galaxy, the system:
     - Extracts a 20x20 pixel region around your marker
     - Analyzes the pixel brightness values
     - Calculates synthetic ellipticity features (eps1, eps2)

2. **Feature Calculation:**
   ```javascript
   // For each marked galaxy:
   - Get pixel data around the marker
   - Calculate average brightness
   - Generate ellipticity components:
     * eps1: Based on brightness and distortion
     * eps2: Random component (simulating shape)
   
   // Dark matter regions get:
   - Higher ellipticity (eps1: 0.05-0.08, eps2: 0.05-0.08)
   - Stronger shear signal
   
   // Background regions get:
   - Lower ellipticity (eps1: -0.02 to 0.02, eps2: -0.02 to 0.02)
   - Weak/no shear signal
   ```

3. **Data Structure Created:**
   ```javascript
   galaxyData = [
     {
       x: 150,           // X coordinate on image
       y: 200,           // Y coordinate on image
       eps1: 0.065,      // Ellipticity component 1
       eps2: 0.072,      // Ellipticity component 2
       label: 1          // 1 = dark matter, 0 = background
     },
     // ... more galaxies
   ]
   ```

4. **Ground Truth Visualization:**
   - Creates a side-by-side view showing your markings
   - Red = dark matter regions you marked
   - Blue = background regions you marked

### Step 4: Run Prediction
**What happens when you click "Run Prediction":**

1. **Send to API:**
   ```javascript
   // Features are sent to the Flask API server
   POST http://localhost:5000/predict
   {
     "features": [
       [0.065, 0.072],  // eps1, eps2 for galaxy 1
       [0.012, -0.008], // eps1, eps2 for galaxy 2
       // ... more galaxies
     ]
   }
   ```

2. **API Processing:**
   - Receives the ellipticity features
   - Scales them using the same scaler from training
   - Runs them through the trained VQC model
   - Returns predictions and probabilities

3. **Model Prediction:**
   ```python
   # In api_server.py:
   X_scaled = scaler.transform(features)  # Scale to 0-1 range
   predictions = vqc_model.predict(X_scaled)  # Binary: 0 or 1
   probabilities = vqc_model.predict_proba(X_scaled)  # [prob_no_dm, prob_dm]
   ```

4. **Quantum ML Processing:**
   - Features are encoded into quantum states using ZZFeatureMap
   - Quantum circuit processes the data
   - Outputs probability of dark matter presence
   - Threshold: probability > 0.5 = dark matter detected

5. **Results Returned:**
   ```json
   {
     "predictions": [1, 0, 1, 0, ...],  // 1 = dark matter, 0 = background
     "probabilities": [
       [0.2, 0.8],  // 20% background, 80% dark matter
       [0.9, 0.1],  // 90% background, 10% dark matter
       ...
     ]
   }
   ```

### Step 5: Visualization
**What you see:**

1. **Your Markings (Ground Truth):**
   - Left side shows your manual markings
   - Red circles = you marked as dark matter
   - Blue circles = you marked as background

2. **Model Predictions:**
   - Right side shows model predictions
   - Yellow stars ⭐ = model predicts dark matter
   - Cyan circles = model predicts background

3. **Accuracy Calculation:**
   ```
   Accuracy = (Correct Predictions) / (Total Galaxies)
   
   Correct = Model prediction matches your marking
   Example:
   - You marked: Dark Matter (1)
   - Model predicted: Dark Matter (1)
   → Correct! ✅
   
   - You marked: Background (0)
   - Model predicted: Dark Matter (1)
   → Incorrect ❌
   ```

---

## Technical Flow Diagram

```
┌─────────────────┐
│  Select Image   │
│  (Example/Upload)│
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Mark Galaxies  │
│  (Click to mark)│
│  Red = DM       │
│  Blue = BG      │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Generate Grid   │
│ Extract Features│
│ eps1, eps2      │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Send to API    │
│  POST /predict  │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Flask API      │
│  Scale Features │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  VQC Model      │
│  Quantum ML     │
│  Prediction     │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Return Results │
│  Predictions    │
│  Probabilities  │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Visualize      │
│  Compare        │
│  Calculate Acc  │
└─────────────────┘
```

---

## Key Concepts Explained

### 1. Ellipticity (eps1, eps2)
- **What it is:** Measures how elliptical (stretched) a galaxy appears
- **Why it matters:** Dark matter causes gravitational lensing, which distorts galaxy shapes
- **How it's measured:**
  - eps1: Horizontal/vertical stretching
  - eps2: Diagonal stretching
  - Together they describe the galaxy's shape distortion

### 2. Gravitational Lensing
- **What it is:** Dark matter bends light, making background galaxies appear distorted
- **Effect:** Galaxies behind dark matter halos look stretched/elliptical
- **Our simulation:** Distorted galaxies in examples simulate this effect

### 3. Quantum Machine Learning
- **Why quantum?** Can process complex patterns in high-dimensional data faster
- **Our model:** Variational Quantum Classifier (VQC)
- **Process:**
  1. Encode ellipticity features into quantum states
  2. Apply quantum gates (rotations, entanglements)
  3. Measure output to get probability
  4. Classify: dark matter or background

### 4. Feature Extraction
- **Real astronomy:** Uses sophisticated algorithms to measure galaxy shapes
- **Our demo:** Simulates this by:
  - Analyzing pixel brightness around markers
  - Generating synthetic ellipticity values
  - Dark matter regions → higher ellipticity
  - Background regions → lower ellipticity

---

## What Makes This Special?

1. **Interactive Learning:** You see how the model works by marking galaxies yourself
2. **Real-time Comparison:** Instantly see if the model agrees with your markings
3. **Educational:** Demonstrates the connection between:
   - Visual galaxy appearance
   - Ellipticity measurements
   - Quantum ML predictions
   - Dark matter detection

4. **Scalable Concept:** Shows how this could work on:
   - Millions of galaxies from telescope surveys
   - Real-time dark matter mapping
   - Large-scale universe analysis

---

## Limitations & Future Improvements

**Current Limitations:**
- Feature extraction is simplified (uses brightness, not true shape analysis)
- Example images are synthetic
- Model uses small training dataset (100 samples)

**Real-World Implementation Would:**
- Use proper galaxy shape measurement algorithms
- Process real telescope data (SDSS, DES, LSST)
- Train on millions of galaxies
- Include more features (redshift, brightness, size, etc.)
- Use actual gravitational lensing measurements

---

## Summary

The interactive feature is a **hands-on demonstration** that:
1. Lets you mark galaxies you think have dark matter
2. Extracts ellipticity features from your markings
3. Uses quantum ML to predict dark matter locations
4. Compares your expert opinion with the model's predictions
5. Shows accuracy metrics

It's designed to be **educational and intuitive** - you don't need to be an astronomer to understand how dark matter detection works!

