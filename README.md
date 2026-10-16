# 🌫️ Air Quality Stacking & Hybrid Ensemble Benchmark Suite (5 Indian Megacities)

An advanced benchmark suite for **hourly PM2.5 air quality forecasting** across major Indian metropolitan centers using **Hybrid Ensembles (Machine Learning + Deep Learning Fusion)**.

The framework implements a **dual-tier, cross-dataset validation paradigm**:
1. **Tier 1 (Training)**: Historical Continuous Ambient Air Quality Monitoring Stations (CAAQMS) from CPCB / OpenCity (**Delhi, Mumbai, Kolkata, Chennai, Bengaluru**) comprising **1,586,340 sequence samples**.
2. **Tier 2 (Zero-Leakage Testing)**: Independent **Copernicus Atmosphere Monitoring Service (CAMS) 2024 satellite reanalysis** for the exact same 5 cities (**43,800 hourly sequence samples**).

---

## 📌 Table of Contents
- [Dual-Dataset Architecture](#-dual-dataset-architecture)
- [Hybrid Ensemble Architectures](#-hybrid-ensemble-architectures)
- [Benchmark Results Leaderboard](#-benchmark-results-leaderboard)
- [Per-City Performance (5 Cities)](#-per-city-performance-5-cities)
- [Directory Structure](#-directory-structure)
- [Execution & Quickstart](#-execution--quickstart)

---

## 🌐 Dual-Dataset Architecture

To eliminate single-sensor bias and guarantee strict zero-leakage generalizability, models are trained on physical ground IoT sensors and evaluated against independent satellite-modeled atmospheric reanalysis:

| City | Coordinates | In-Situ Ground Stations (CPCB Tier-1 Training) | CAMS 2024 (Tier-2 Testing) |
| :--- | :---: | :--- | :--- |
| **Delhi** | $28.6139^\circ\text{N}, 77.2090^\circ\text{E}$ | Anand Vihar, Mandir Marg, Punjabi Bagh, R.K. Puram | 8,760 hourly sequences (Full Year 2024) |
| **Mumbai** | $19.0760^\circ\text{N}, 72.8777^\circ\text{E}$ | BKC, Borivali East, Chembur, Powai, Sion MPCB | 8,760 hourly sequences (Full Year 2024) |
| **Kolkata** | $22.5726^\circ\text{N}, 88.3639^\circ\text{E}$ | Ballygunge, Fort William, Jadavpur, Rabindra Bharati Univ. | 8,760 hourly sequences (Full Year 2024) |
| **Chennai** | $13.0827^\circ\text{N}, 80.2707^\circ\text{E}$ | Kodungaiyur, Manali CPCB, Perungudi, Royapuram | 8,760 hourly sequences (Full Year 2024) |
| **Bengaluru**| $12.9716^\circ\text{N}, 77.5946^\circ\text{E}$ | BTM Layout, Hebbal, Hombegowda Nagar, Peenya | 8,760 hourly sequences (Full Year 2024) |

### Sequence Formulation
- **Lookback ($T$)**: 24 hours ($t-23 \dots t$).
- **Prediction Horizon**: 1 hour ahead ($t+1$).
- **Features ($d = 7$)**: $\text{PM}_{2.5}$, $\sin(\text{hour})$, $\cos(\text{hour})$, $\sin(\text{DoW})$, $\cos(\text{DoW})$, $\sin(\text{month})$, $\cos(\text{month})$.
- **Scaling**: Standardized using statistics calculated strictly on Tier-1 ground data with zero test-set leakage.

---

## 🤝 Hybrid Ensemble Architectures

The suite evaluates individual models alongside **Hybrid ML + Deep Learning Ensembles**:
- **Hybrid Stacking (Optimal Blend)**: Combines tree-based decision splitting with recurrent latent sequence representations:
  $$\hat{y}_{\text{Hybrid}} = \alpha^* \cdot \hat{y}_{\text{ML}} + (1 - \alpha^*) \cdot \hat{y}_{\text{DL}}$$
  *Optimal weights learned on validation set: 34% XGBoost GPU + 66% PyTorch GRU.*
- **Hybrid (XGBoost + GRU)**: 50/50 fusion.
- **Hybrid (LightGBM + LSTM)**: 50/50 fusion.
- **Hybrid (LightGBM + CNN-LSTM)**: 50/50 fusion.
- **Hybrid (Ridge + BiLSTM)**: 50/50 fusion.

---

## 🏆 Benchmark Results Leaderboard

Evaluated on the independent **2024 CAMS External Reanalysis Dataset** across the 5 cities (**43,800 test sequences**):

| Rank | Model Architecture | Paradigm | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | $R^2$ Score | MAPE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Hybrid (0.34\*XGBoost + 0.66\*GRU)** | **Hybrid (ML + DL)** | **3.69** | **5.51** | **0.9752** | **15.88%** |
| 🥈 | **LightGBM** | Machine Learning / GBDT | 3.65 | 5.52 | 0.9752 | 16.40% |
| 🥉 | **Hybrid: XGBoost + GRU (50/50)** | **Hybrid (ML + DL)** | **3.67** | **5.54** | **0.9750** | **16.27%** |
| 4 | **GRU** | Deep Learning (PyTorch CUDA) | 3.83 | 5.59 | 0.9746 | 15.71% |
| 5 | **Hybrid: LightGBM + LSTM (50/50)** | **Hybrid (ML + DL)** | **4.04** | **5.73** | **0.9732** | **18.78%** |
| 6 | **Linear Regression** | Machine Learning (OLS) | 3.82 | 5.78 | 0.9728 | 15.52% |
| 7 | **BiLSTM** | Deep Learning (PyTorch CUDA) | 4.07 | 5.86 | 0.9720 | 19.53% |
| 8 | **XGBoost** | Machine Learning (GPU CUDA) | 3.79 | 5.87 | 0.9719 | 18.73% |
| 9 | **Random Forest** | Machine Learning (Ensemble) | 3.67 | 5.92 | 0.9714 | 17.51% |
| 10 | **Stacked LSTM** | Deep Learning (PyTorch CUDA) | 3.97 | 5.93 | 0.9713 | 14.99% |
| 11 | **LSTM** | Deep Learning (PyTorch CUDA) | 4.54 | 6.15 | 0.9692 | 21.53% |
| 12 | **Hybrid: LightGBM + CNN-LSTM** | **Hybrid (ML + DL)** | 5.30 | 6.81 | 0.9622 | 29.20% |
| 13 | **Hybrid: Ridge + BiLSTM** | **Hybrid (ML + DL)** | 4.49 | 7.73 | 0.9513 | 20.63% |
| 14 | **CNN-LSTM** | Deep Learning (PyTorch CUDA) | 7.32 | 8.82 | 0.9366 | 43.10% |
| 15 | **Ridge Regression** | Machine Learning ($L_2$ linear) | 5.32 | 12.08 | 0.8810 | 23.67% |

---

## 🏙️ Per-City Performance (5 Cities)

| City | Best Model | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| 🌳 **Bengaluru** | **Hybrid (0.34\*XGBoost + 0.66\*GRU)** | **3.04** | **3.77** | **0.9237** |
| 🌴 **Chennai** | **LightGBM** *(Runner-up: Hybrid XGBoost+GRU)* | **2.25** | **2.72** | **0.9378** |
| 🏛️ **Delhi** | **LightGBM** *(Runner-up: Hybrid 0.34\*XGB+0.66\*GRU)* | **5.63** | **8.21** | **0.9607** |
| 🌉 **Kolkata** | **Hybrid (0.34\*XGBoost + 0.66\*GRU)** | **4.51** | **6.83** | **0.9764** |
| 🌊 **Mumbai** | **Random Forest** *(Runner-up: LightGBM)* | **2.61** | **3.60** | **0.9721** |

---

## 📂 Directory Structure

```
c:/Users/aarya/spm/
├── models/                                # 🧠 Trained Weights & Scalers
│   ├── lstm_pm25.pt                       # PyTorch LSTM weights
│   ├── gru_pm25.pt                        # PyTorch GRU weights
│   ├── bilstm_pm25.pt                     # PyTorch BiLSTM weights
│   ├── stacked_lstm_pm25.pt               # PyTorch Stacked LSTM weights
│   ├── cnn_lstm_pm25.pt                   # PyTorch CNN-LSTM weights
│   ├── xgboost.pkl                        # Saved XGBoost GPU model
│   ├── lightgbm.pkl                       # Saved LightGBM model
│   ├── random_forest.pkl                  # Saved Random Forest model
│   ├── linear_regression.pkl              # Saved Linear Regression model
│   ├── ridge_regression.pkl               # Saved Ridge Regression model
│   └── feature_scaler.pkl                 # Fitted feature scaler
│
├── results/                               # 📊 Metric Tables
│   ├── overall_results.csv                # Overall metrics on 2024 CAMS dataset
│   └── per_city_results.csv               # Per-city breakdown metrics
│
├── plots/                                 # 📈 Visualizations
│   ├── rmse_cross_dataset.png             # RMSE comparison chart
│   └── r2_cross_dataset.png               # R2 comparison chart
│
├── data/                                  # CPCB OpenCity ground station data (5 cities)
├── data_external/                         # 2024 CAMS hourly PM2.5 CSVs (5 cities)
├── cached_cross_dataset_5cities.npz       # Preprocessed multi-city sequence cache
├── cross_dataset_5cities.py               # Main cross-dataset pipeline script
└── README.md                              # Documentation
```

---

## ⚙️ Execution & Quickstart

```bash
# Clone the repository
git clone https://github.com/Hiiarrow/aqi.git
cd aqi

# Install PyTorch with CUDA acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn joblib requests

# Run the 5-city cross-dataset benchmark pipeline:
python cross_dataset_5cities.py
```
