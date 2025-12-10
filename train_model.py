import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler

df = pd.read_csv('/home/aviv/Documents/dark_energy_density_proj/dark_matter_dataset.csv')
# 1. הפרדת פיצ'רים ותווית
X = df[['Observed_Eps1', 'Observed_Eps2']].values
Y = df['Label'].values

# 2. נרמול הנתונים (מ-0 ל-1, נדרש ל-Feature Map)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. חלוקה לאימון ובדיקה (נשתמש בנתונים קטנים לאימון QML)
# הגבלה של הנתונים ל-100 דוגמאות בגלל מגבלות סימולציה.
X_train_small, X_test_small, Y_train_small, Y_test_small = train_test_split(
    X_scaled[:100], Y[:100], test_size=0.3, random_state=42
)

num_qubits = X_train_small.shape[1]
reps_feature = 2
feature_map = ZZFeatureMap(
    feature_dimension=num_qubits, 
    reps=reps_feature, 
    entanglement='linear'
)

reps_ansatz = 3 # מספר החזרות של המעגל הפרמטרי

ansatz = RealAmplitudes(
    num_qubits=num_qubits, 
    reps=reps_ansatz, 
    entanglement='linear'
)
# print(ansatz.decompose().draw(output='mpl', style='iqp'))

optimizer = COBYLA(maxiter=100) # מספר איטרציות נמוך להתחלה מהי

sampler = Sampler()

vqc_model = VQC(feature_map=feature_map,
ansatz=ansatz, optimizer = optimizer, sampler = sampler, loss='cross_entropy', callback=None)

print("training starts")

vqc_model.fit(X_train_small, Y_train_small)

print("training ended")

from sklearn.metrics import accuracy_score

# 1. חיזוי על קבוצת הבדיקה
Y_pred_small = vqc_model.predict(X_test_small)

# 2. חישוב דיוק
accuracy = accuracy_score(Y_test_small, Y_pred_small)

print(f"\n--- תוצאות הערכה ---")
print(f"דיוק המודל הקוונטי (VQC) על קבוצת הבדיקה: {accuracy*100:.2f}%")
print(f"כמות דגימות האימון: {len(X_train_small)}")

# Save the model, scaler, and configuration for later use
import pickle

# Save the entire model using pickle
try:
    with open('vqc_model.pkl', 'wb') as f:
        pickle.dump(vqc_model, f)
    print("Full model saved successfully!")
except Exception as e:
    print(f"Warning: Could not save full model: {e}")
    # Try to extract weights from neural_network
    try:
        weights = vqc_model.neural_network.weights
        print(f"Extracted weights: {len(weights)} parameters")
    except:
        weights = None
        print("Could not extract weights")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save model configuration with weights
model_config = {
    'num_qubits': num_qubits,
    'reps_feature': reps_feature,
    'reps_ansatz': reps_ansatz,
    'weights': weights if 'weights' in locals() else None
}
with open('vqc_model_config.pkl', 'wb') as f:
    pickle.dump(model_config, f)

print("Scaler and model configuration saved successfully!")