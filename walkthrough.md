# Cross-City Air Quality PM2.5 Prediction & Classification Benchmark

All 10 forecasting models have been thoroughly evaluated across **389,974 test instances** in unseen test cities (**Bengaluru** & **Hyderabad**) across both **regression metrics** and **classification / air quality category metrics**.

---

## 🏆 Overall Benchmark Summary (Unseen Test Cities)

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

## 🏷️ Classification & F1-Score Benchmarks (CPCB AQI Categories)

Air Quality Index (AQI) categories defined under standard CPCB thresholds (Good: 0-30, Satisfactory: 30-60, Moderate: 60-90, Poor: 90-120, Very Poor: 120-250, Severe: >250 $\mu g/m^3$) and Binary Exceedance Threshold (>60 $\mu g/m^3$):

### 1. AQI Multi-Class Category Classification Benchmark

| Rank | Model | Category Accuracy (%) | Weighted F1 | Macro F1 | Weighted Precision | Weighted Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost (GPU CUDA)** | **73.65%** | **0.7363** | **0.7152** | **0.7413** | **0.7365** |
| 🥈 | **LightGBM** | **73.56%** | **0.7356** | **0.7155** | **0.7408** | **0.7356** |
| 🥉 | **Random Forest** | **73.33%** | **0.7337** | **0.7136** | **0.7383** | **0.7333** |
| 4 | **Linear Regression** | 71.72% | 0.7172 | 0.6986 | 0.7241 | 0.7172 |
| 5 | **Ridge Regression** | 71.72% | 0.7171 | 0.6985 | 0.7240 | 0.7172 |
| 6 | **GRU (PyTorch CUDA)** | 71.08% | 0.7021 | 0.6723 | 0.7281 | 0.7108 |
| 7 | **LSTM (PyTorch CUDA)** | 70.78% | 0.6985 | 0.6658 | 0.7276 | 0.7078 |
| 8 | **BiLSTM (PyTorch CUDA)** | 70.31% | 0.6973 | 0.6688 | 0.7217 | 0.7031 |
| 9 | **CNN-LSTM (PyTorch CUDA)** | 66.97% | 0.6532 | 0.6192 | 0.6980 | 0.6697 |
| 10 | **Stacked LSTM (PyTorch CUDA)** | 64.46% | 0.6189 | 0.5817 | 0.6778 | 0.6446 |

### 2. High Pollution Exceedance Benchmark (Binary PM2.5 > 60 $\mu g/m^3$)

| Rank | Model | Exceedance F1 | Exceedance Precision | Exceedance Recall |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 | **Random Forest** | **0.9277** | 0.9077 | 0.9486 |
| 🥈 | **XGBoost (GPU CUDA)** | **0.9269** | **0.9089** | 0.9455 |
| 🥉 | **LightGBM** | **0.9268** | 0.9087 | 0.9457 |
| 4 | **LSTM (PyTorch CUDA)** | 0.9248 | 0.8955 | 0.9561 |
| 5 | **GRU (PyTorch CUDA)** | 0.9241 | 0.8958 | 0.9543 |
| 6 | **Linear Regression** | 0.9232 | 0.9043 | 0.9428 |
| 7 | **Ridge Regression** | 0.9232 | 0.9044 | 0.9428 |
| 8 | **BiLSTM (PyTorch CUDA)** | 0.9220 | 0.8852 | 0.9620 |
| 9 | **CNN-LSTM (PyTorch CUDA)** | 0.9153 | 0.8728 | 0.9621 |
| 10 | **Stacked LSTM (PyTorch CUDA)** | 0.9063 | 0.8463 | **0.9753** |

---

## 📐 Extended Regression Benchmarks

| Model | $R^2$ | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | MAPE (%) | EVS | MedAE ($\mu g/m^3$) | MBE ($\mu g/m^3$) | NMBE (%) | CV-RMSE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.8040** | **12.02** | **23.47** | **17.13%** | 0.8048 | 6.62 | +1.47 | +1.80% | **28.62%** |
| **LSTM** | **0.8033** | 12.72 | **23.52** | 19.92% | **0.8063** | 7.85 | +2.92 | +3.55% | 28.67% |
| **XGBoost** | **0.8027** | **12.03** | **23.55** | **17.16%** | 0.8035 | **6.56** | +1.50 | +1.82% | 28.72% |
| **GRU** | 0.8020 | 12.81 | 23.59 | 20.01% | 0.8060 | 7.85 | +3.35 | +4.09% | 28.76% |
| **BiLSTM** | 0.7999 | 12.97 | 23.72 | 19.95% | 0.8050 | 8.07 | +3.80 | +4.63% | 28.92% |
| **Random Forest** | 0.7995 | 12.13 | 23.75 | 17.42% | 0.8004 | 6.64 | +1.63 | +1.98% | 28.95% |
| **Linear Reg** | 0.7986 | 12.50 | 23.80 | 18.29% | 0.7992 | 7.26 | **+1.24** | **+1.52%** | 29.01% |
| **Ridge Reg** | 0.7986 | 12.50 | 23.80 | 18.30% | 0.7991 | 7.27 | **+1.24** | **+1.52%** | 29.01% |
| **CNN-LSTM** | 0.7915 | 13.98 | 24.21 | 22.60% | 0.8006 | 9.48 | +5.05 | +6.15% | 29.52% |
| **Stacked LSTM** | 0.7825 | 15.07 | 24.73 | 26.08% | 0.8024 | 11.17 | +7.47 | +9.11% | 30.15% |

---

## 🏙️ Per-City Benchmark Breakdown

### 1. Bengaluru (190,898 Test Instances)

| Model | $R^2$ | MAE | RMSE | Category Accuracy (%) | Weighted F1 | Exceedance F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.7148** | **11.14** | **25.28** | **73.57%** | **0.7354** | **0.8964** |
| **GRU** | 0.7147 | 12.07 | 25.28 | 70.38% | 0.6934 | 0.8921 |
| **LSTM** | 0.7139 | 12.02 | 25.31 | 70.21% | 0.6898 | 0.8939 |
| **XGBoost** | 0.7122 | 11.18 | 25.39 | 73.47% | 0.7345 | 0.8958 |
| **BiLSTM** | 0.7076 | 12.42 | 25.59 | 69.14% | 0.6839 | 0.8884 |
| **Linear Reg** | 0.7075 | 11.84 | 25.60 | 71.24% | 0.7113 | 0.8908 |
| **Random Forest** | 0.7075 | 11.32 | 25.60 | 73.31% | 0.7337 | 0.8979 |
| **Ridge Reg** | 0.7075 | 11.85 | 25.60 | 71.23% | 0.7112 | 0.8909 |
| **CNN-LSTM** | 0.7018 | 13.34 | 25.84 | 65.43% | 0.6338 | 0.8799 |
| **Stacked LSTM** | 0.6859 | 14.77 | 26.53 | 62.36% | 0.5945 | 0.8663 |

### 2. Hyderabad (199,076 Test Instances)

| Model | $R^2$ | MAE | RMSE | Category Accuracy (%) | Weighted F1 | Exceedance F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.8473** | 12.86 | **21.60** | 73.54% | 0.7356 | 0.9483 |
| **XGBoost** | 0.8467 | **12.84** | 21.65 | **73.81%** | **0.7382** | **0.9487** |
| **LSTM** | 0.8466 | 13.39 | 21.65 | 71.33% | 0.7069 | 0.9468 |
| **BiLSTM** | 0.8449 | 13.50 | 21.77 | 71.43% | 0.7102 | 0.9463 |
| **Random Forest** | 0.8442 | 12.91 | 21.82 | 73.35% | 0.7335 | 0.9487 |
| **GRU** | 0.8438 | 13.52 | 21.85 | 71.75% | 0.7109 | 0.9470 |
| **Linear Reg** | 0.8426 | 13.13 | 21.93 | 72.18% | 0.7224 | 0.9461 |
| **Ridge Reg** | 0.8426 | 13.13 | 21.93 | 72.18% | 0.7224 | 0.9461 |
| **CNN-LSTM** | 0.8338 | 14.59 | 22.53 | 68.45% | 0.6721 | 0.9410 |
| **Stacked LSTM** | 0.8288 | 15.35 | 22.87 | 66.47% | 0.6426 | 0.9358 |

---

## 📁 Generated Files in Project Workspace (`c:\Users\aarya\spm`)

- [walkthrough.md](file:///c:/Users/aarya/spm/walkthrough.md)
- [extended_overall_benchmark_results.csv](file:///c:/Users/aarya/spm/extended_overall_benchmark_results.csv)
- [extended_per_city_benchmark_results.csv](file:///c:/Users/aarya/spm/extended_per_city_benchmark_results.csv)
- [overall_unseen_city_results.csv](file:///c:/Users/aarya/spm/overall_unseen_city_results.csv)
- [per_city_unseen_results.csv](file:///c:/Users/aarya/spm/per_city_unseen_results.csv)
