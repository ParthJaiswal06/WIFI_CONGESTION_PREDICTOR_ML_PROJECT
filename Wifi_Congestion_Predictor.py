import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load the saved trained model
# ─────────────────────────────────────────────────────────────
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=" * 55)
print("   Wi-Fi Congestion Predictor — ML Project")
print("=" * 55)
print(f"✅ Model Loaded Successfully!")
print(f"   Model Type : {type(model).__name__}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Label mapping
# ─────────────────────────────────────────────────────────────
# LabelEncoder maps alphabetically:
# 0 = High | 1 = Low | 2 = Medium
label_map = {0: 'High 🔴', 1: 'Low 🟢', 2: 'Medium 🟡'}

# ─────────────────────────────────────────────────────────────
# STEP 3 — Load Feature Selected CSV directly
# No need to refit scaler or PCA — already transformed data
# ─────────────────────────────────────────────────────────────
feature_df   = pd.read_csv('Dataset_campus_wifi_Feature_Selected.csv')
feature_cols = feature_df.drop('Congestion_Encoded', axis=1).columns.tolist()
n_components = len(feature_cols)

print(f"   Features expected by model : {n_components}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Menu Selection
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("   SELECT PREDICTION MODE")
print("=" * 55)
print("   [1] Enter custom network feature values manually")
print("   [2] Predict from a specific record in the dataset")
print("   [3] Run predictions on the entire dataset")
print("=" * 55)

choice = input("\nEnter your choice (1 / 2 / 3) : ").strip()

# ─────────────────────────────────────────────────────────────
# MODE 1 — Manual User Input
# ─────────────────────────────────────────────────────────────
if choice == '1':
    print("\n" + "=" * 55)
    print("   ENTER NETWORK FEATURE VALUES")
    print("=" * 55)
    print("(Enter realistic values based on the ranges below)\n")

    prompts = {
        'Num_Connected_Devices'       : ('Number of Connected Devices',  '5 to 300'  ),
        'Bandwidth_Usage_Mbps'        : ('Bandwidth Usage (Mbps)',        '5 to 350'  ),
        'AP_Load_Percent'             : ('Access Point Load (%)',         '0 to 100'  ),
        'Network_Latency_ms'          : ('Network Latency (ms)',          '1 to 500'  ),
        'Packet_Loss_Rate_Percent'    : ('Packet Loss Rate (%)',          '0 to 20'   ),
        'Signal_Strength_dBm'        : ('Signal Strength (dBm)',         '-90 to -30'),
        'Retransmission_Rate_Percent' : ('Retransmission Rate (%)',       '0 to 25'   ),
        'Throughput_Mbps'             : ('Throughput (Mbps)',             '1 to 340'  ),
        'Active_Sessions'             : ('Number of Active Sessions',     '5 to 450'  ),
        'Channel_Utilization_Percent' : ('Channel Utilization (%)',       '0 to 100'  ),
    }

    user_values = {}
    for key, (label, range_hint) in prompts.items():
        while True:
            try:
                val = float(input(f"   {label} [{range_hint}] : "))
                user_values[key] = val
                break
            except ValueError:
                print("   ⚠️  Please enter a valid number.")

    # Load cleaned CSV to find closest matching row
    cleaned_df   = pd.read_csv('Dataset_campus_wifi_Cleaned.csv')
    numeric_cols = list(prompts.keys())

    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.preprocessing import StandardScaler as SS

    temp_scaler  = SS()
    cleaned_nums = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    temp_scaled  = temp_scaler.fit_transform(cleaned_nums)

    user_row    = np.array([[user_values[c] for c in numeric_cols]])
    user_scaled = temp_scaler.transform(user_row)

    distances   = euclidean_distances(user_scaled, temp_scaled)
    closest_idx = np.argmin(distances)

    # Grab the closest matching row from Feature Selected CSV
    feature_df = pd.read_csv('Dataset_campus_wifi_Feature_Selected.csv')
    input_pca  = feature_df.drop('Congestion_Encoded', axis=1).values[closest_idx].reshape(1, -1)

    # Predict
    prediction = model.predict(input_pca)
    congestion = label_map[prediction[0]]

    try:
        proba       = model.predict_proba(input_pca)
        prob_high   = proba[0][0] * 100
        prob_low    = proba[0][1] * 100
        prob_medium = proba[0][2] * 100
    except:
        prob_high = prob_low = prob_medium = None

    print("\n" + "=" * 55)
    print("   PREDICTION RESULT")
    print("=" * 55)
    print(f"\n   Predicted Congestion Level : {congestion}")

    if prob_high is not None:
        print(f"\n   Prediction Probabilities:")
        print(f"      🔴 High   : {prob_high:.2f}%")
        print(f"      🟢 Low    : {prob_low:.2f}%")
        print(f"      🟡 Medium : {prob_medium:.2f}%")

# ─────────────────────────────────────────────────────────────
# MODE 2 — Predict from a specific record in the dataset
# ─────────────────────────────────────────────────────────────
elif choice == '2':
    feature_df = pd.read_csv('Dataset_campus_wifi_Feature_Selected.csv')
    total      = len(feature_df)

    print(f"\n   Dataset has {total} records (Row numbers: 0 to {total - 1})")
    while True:
        try:
            row_num = int(input(f"   Enter row number to predict (0 to {total - 1}) : "))
            if 0 <= row_num < total:
                break
            else:
                print(f"   ⚠️  Please enter a number between 0 and {total - 1}.")
        except ValueError:
            print("   ⚠️  Please enter a valid integer.")

    X_row    = feature_df.drop('Congestion_Encoded', axis=1).values
    y_actual = feature_df['Congestion_Encoded'].values

    single_input = X_row[row_num].reshape(1, -1)
    prediction   = model.predict(single_input)

    actual_label    = label_map[y_actual[row_num]]
    predicted_label = label_map[prediction[0]]
    match           = '✅ Correct' if y_actual[row_num] == prediction[0] else '❌ Incorrect'

    try:
        proba       = model.predict_proba(single_input)
        prob_high   = proba[0][0] * 100
        prob_low    = proba[0][1] * 100
        prob_medium = proba[0][2] * 100
    except:
        prob_high = prob_low = prob_medium = None

    print("\n" + "=" * 55)
    print(f"   PREDICTION RESULT FOR ROW {row_num}")
    print("=" * 55)
    print(f"\n   Actual Congestion Level    : {actual_label}")
    print(f"   Predicted Congestion Level : {predicted_label}")
    print(f"   Match                      : {match}")

    if prob_high is not None:
        print(f"\n   Prediction Probabilities:")
        print(f"      🔴 High   : {prob_high:.2f}%")
        print(f"      🟢 Low    : {prob_low:.2f}%")
        print(f"      🟡 Medium : {prob_medium:.2f}%")

# ─────────────────────────────────────────────────────────────
# MODE 3 — Full Dataset Prediction
# ─────────────────────────────────────────────────────────────
elif choice == '3':
    feature_df = pd.read_csv('Dataset_campus_wifi_Feature_Selected.csv')
    X          = feature_df.drop('Congestion_Encoded', axis=1).values
    y_actual   = feature_df['Congestion_Encoded'].values
    y_pred     = model.predict(X)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_actual, y_pred)

    results_df = pd.DataFrame({
        'Actual_Label'    : [label_map[i] for i in y_actual],
        'Predicted_Label' : [label_map[i] for i in y_pred],
        'Match'           : ['✅' if a == p else '❌' for a, p in zip(y_actual, y_pred)]
    })

    print("\n" + "=" * 55)
    print("   PREDICTIONS ON FULL DATASET")
    print("=" * 55)
    print(f"\n   Total Records : {len(y_pred)}")
    print(f"\n{results_df.head(20).to_string(index=False)}")
    print(f"\n   Overall Accuracy : {acc * 100:.2f}%")

else:
    print("\n   ⚠️  Invalid choice. Please run the program again and enter 1, 2, or 3.")

print("\n" + "=" * 55)
print("   Done.")
print("=" * 55)