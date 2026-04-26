# Wi-Fi Congestion Predictor — ML Project

An end-to-end machine learning classification system that predicts 
campus Wi-Fi network congestion levels as **Low**, **Medium**, or **High** 
based on real-time network performance data.

## Features Used
Connected Devices, Bandwidth Usage, AP Load, Network Latency,
Packet Loss Rate, Signal Strength, Throughput, and more.

## Pipeline
Raw Dataset → Data Cleaning → EDA → Encoding & Normalization
→ Feature Selection (PCA, Forward, Backward) → Model Training → Evaluation

## Best Model
**Random Forest Classifier — 98.20% Accuracy**

## Tech Stack
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, SMOTE

## Files
- `01_data_cleaning.ipynb` — `02_eda.ipynb` — `03_feature_selection.ipynb` — `04_model_evaluation.ipynb`
- `Wifi_Congestion_Predictor.py` — Standalone prediction script
- `models/best_model.pkl` — Saved trained model
- `Synopsis PBL Phase 1 TCS 203.pdf` — PDF or Docx format of Summary of the project.
