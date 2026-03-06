import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# STEP 1 — Load the saved model
# ─────────────────────────────────────────
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")
print(f"   Model type: {type(model).__name__}")

# ─────────────────────────────────────────
# STEP 2 — Label mapping (for readable output)
# ─────────────────────────────────────────
# Based on LabelEncoder alphabetical order:
# 0 = High, 1 = Low, 2 = Medium
label_map = {0: 'High', 1: 'Low', 2: 'Medium'}

# ─────────────────────────────────────────
# STEP 3 — Option A: Predict on your test CSV
# Just make sure Dataset_campus_wifi_Feature_Selected.csv is in the same folder
# ─────────────────────────────────────────
df = pd.read_csv('Dataset_campus_wifi_Feature_Selected.csv')
X = df.drop('Congestion_Encoded', axis=1).values
y_actual = df['Congestion_Encoded'].values

y_pred = model.predict(X)

print("\n─────────────────────────────────────────")
print("PREDICTIONS ON LOADED DATASET")
print("─────────────────────────────────────────")
print(f"Total Records  : {len(y_pred)}")

results_df = pd.DataFrame({
    'Actual_Label'   : [label_map[i] for i in y_actual],
    'Predicted_Label': [label_map[i] for i in y_pred],
    'Match'          : ['✅' if a == p else '❌' for a, p in zip(y_actual, y_pred)]
})

print(results_df.head(20).to_string(index=False))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_actual, y_pred)
print(f"\nOverall Accuracy on this data: {acc * 100:.2f}%")

# ─────────────────────────────────────────
# STEP 4 — Option B: Predict on a single custom input
# Replace the values below with any real/test values you want
# Number of values must match the number of PCA components you saved
# ─────────────────────────────────────────
print("\n─────────────────────────────────────────")
print("SINGLE CUSTOM PREDICTION")
print("─────────────────────────────────────────")

# Get number of features the model expects
n_features = model.n_features_in_
print(f"Model expects {n_features} input features (PCA components)")

# Create a dummy input with zeros — replace with your actual values
custom_input = np.zeros((1, n_features))

# Example: if you want to test a specific row from the dataset
# custom_input = X[0].reshape(1, -1)  # uses first row of dataset

prediction = model.predict(custom_input)
print(f"Predicted Congestion Level: {label_map[prediction[0]]}")

# If your model supports probability output
try:
    proba = model.predict_proba(custom_input)
    print(f"Prediction Probabilities:")
    for cls, prob in zip(['High', 'Low', 'Medium'], proba[0]):
        print(f"   {cls}: {prob * 100:.2f}%")
except:
    print("(This model does not support probability output)")

print("\n─────────────────────────────────────────")
print("Done.")
print("─────────────────────────────────────────")

