# 🌫️ Cross-City PM2.5 Air Quality Forecasting & Benchmark Suite

An end-to-end benchmark suite for **hourly PM2.5 air quality forecasting** across major Indian metropolitan areas using **PyTorch Deep Learning (CUDA GPU)**, **Gradient Boosted Decision Trees**, and **Classical Machine Learning**. 

The pipeline trains on **4 major training cities** (Delhi, Mumbai, Kolkata, Chennai) and evaluates zero-shot cross-city generalization performance on **2 unseen test cities** (Bengaluru & Hyderabad) comprising **389,974 test instances**.

---

## 📌 Table of Contents
- [Directory Architecture](#-directory-architecture)
- [Dataset Architecture & Preprocessing](#-dataset-architecture--preprocessing)
- [Model Architectures](#-model-architectures)
- [Benchmark Results](#-benchmark-results)
- [Installation & Quickstart](#-installation--quickstart)
- [How to Run Benchmarks](#-how-to-run-benchmarks)

---

## 📂 Directory Architecture

The workspace is organized into clean, dedicated subdirectories:

```
c:/Users/aarya/spm/
├── models/                                # 🧠 All Trained Weights & Scalers
│   ├── lstm_pm25.pt                       # PyTorch LSTM weights
│   ├── gru_pm25.pt                        # PyTorch GRU weights
│   ├── cnn_lstm_pm25.pt                   # PyTorch CNN-LSTM weights
│   ├── bilstm_pm25.pt                     # PyTorch BiLSTM weights
│   ├── stacked_lstm_pm25.pt               # PyTorch Stacked LSTM weights
│   ├── lightgbm.pkl                       # Saved LightGBM model
│   ├── xgboost.pkl                        # Saved XGBoost GPU model
│   ├── random_forest.pkl                  # Saved Random Forest model
│   ├── linear_regression.pkl              # Saved Linear Regression model
│   ├── ridge_regression.pkl               # Saved Ridge Regression model
│   ├── feature_scaler.pkl                 # Scaler for 7 input features
│   └── target_scaler.pkl                  # Scaler for target PM2.5
│
├── results/                               # 📊 All CSV Metric Tables
│   ├── overall_unseen_city_results.csv    # Summary regression metrics across unseen test cities
│   ├── per_city_unseen_results.csv        # Per-city regression metrics (Bengaluru & Hyderabad)
│   ├── extended_overall_benchmark_results.csv # F1, Precision, Recall, Accuracy & extended metrics
│   └── extended_per_city_benchmark_results.csv # Per-city F1 and classification metrics
│
├── plots/                                 # 📈 All Generated Charts
│   ├── r2_unseen_cities.png               # R2 comparison across unseen cities
│   ├── rmse_unseen_cities.png             # RMSE comparison across unseen cities
│   ├── prediction_Bengaluru_LightGBM.png  # Time-series prediction vs actual plot (Bengaluru)
│   └── prediction_Hyderabad_LightGBM.png  # Time-series prediction vs actual plot (Hyderabad)
│
├── data/                                  # OpenCity raw CSV dataset folders
├── cross_city_air_quality_prediction.py   # Main pipeline (preprocessing, model training & evaluation)
├── evaluate_extended_benchmarks.py       # Extended classification & F1 score evaluation script
├── AQI.py                                # PyTorch single-city AQI modeling script
├── cached_dataset.npz                     # Preprocessed sequence arrays (X_train, y_train, X_test, y_test)
├── walkthrough.md                         # Detailed benchmark walkthrough report
└── README.md                              # Main documentation file
```

---

## 📊 Dataset Architecture & Preprocessing

The dataset consists of hourly PM2.5 concentrations ($\mu g/m^3$) collected from monitoring stations across 6 metropolitan cities via OpenCity:

### City Splits
- **Training & Validation Cities (602,020 Sequences)**:
  - 🏛️ **Delhi**: Anand Vihar, Mandir Marg, Punjabi Bagh, R.K. Puram
  - 🌊 **Mumbai**: BKC, Borivali East, Chembur, Powai, Sion MPCB
  - 🌉 **Kolkata**: Ballygunge, Fort William, Jadavpur, Rabindra Bharati University
  - 🌴 **Chennai**: Kodungaiyur, Manali CPCB, Perungudi, Royapuram
- **Unseen Generalization Test Cities (389,974 Sequences)**:
  - 🌳 **Bengaluru** (190,898 seqs): BTM Layout, Hebbal, Hombegowda Nagar, Peenya
  - 🏰 **Hyderabad** (199,076 seqs): Bollaram Industrial Area, ICRISAT Patancheru, Sanathnagar, Zoo Park

### Feature Engineering & Sequence Formulation
- **Lookback Window ($T$)**: 24 hours ($t-23$ to $t$)
- **Prediction Horizon**: 1 hour ahead ($t+1$)
- **Feature Set (7 features per timestep)**: `pm25`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`
- **Input Tensor Dimensions**: `(N, 24, 7)` for PyTorch DL models; flattened to `(N, 168)` for tree/linear baselines.

---

## 📈 Benchmark Results Summary

Evaluated across **389,974 test instances** in unseen test cities (**Bengaluru & Hyderabad**):

| Rank | Model | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | $R^2$ | MAPE (%) | Category Accuracy (%) | Weighted F1 | Exceedance F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **LightGBM** | **12.02** | **23.47** | **0.8040** | **17.13%** | 73.56% | 0.7356 | 0.9268 |
| 🥈 | **LSTM (PyTorch CUDA)** | **12.72** | **23.52** | **0.8033** | 19.92% | 70.78% | 0.6985 | 0.9248 |
| 🥉 | **XGBoost (GPU CUDA)** | **12.03** | **23.55** | **0.8027** | **17.16%** | **73.65%** | **0.7363** | 0.9269 |
| 4 | **GRU (PyTorch CUDA)** | 12.81 | 23.59 | 0.8020 | 20.01% | 71.08% | 0.7021 | 0.9241 |
| 5 | **BiLSTM (PyTorch CUDA)** | 12.97 | 23.72 | 0.7999 | 19.95% | 70.31% | 0.6973 | 0.9220 |
| 6 | **Random Forest** | 12.13 | 23.75 | 0.7995 | 17.42% | 73.33% | 0.7337 | **0.9277** |
| 7 | **Linear Regression** | 12.50 | 23.80 | 0.7986 | 18.29% | 71.72% | 0.7172 | 0.9232 |
| 8 | **Ridge Regression** | 12.50 | 23.80 | 0.7986 | 18.30% | 71.72% | 0.7171 | 0.9232 |
| 9 | **CNN-LSTM (PyTorch CUDA)** | 13.98 | 24.21 | 0.7915 | 22.60% | 66.97% | 0.6532 | 0.9153 |
| 10 | **Stacked LSTM (PyTorch CUDA)** | 15.07 | 24.73 | 0.7825 | 26.08% | 64.46% | 0.6189 | 0.9063 |

---

## ⚙️ Installation & Execution

```bash
# Evaluate extended classification & F1 score benchmarks:
python evaluate_extended_benchmarks.py

# Run complete benchmark pipeline:
python cross_city_air_quality_prediction.py all
```
