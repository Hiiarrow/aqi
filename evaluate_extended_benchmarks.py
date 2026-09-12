import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, median_absolute_error
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def resolve_path(filename):
    p_sub = os.path.join(MODELS_DIR, filename)
    if os.path.exists(p_sub):
        return p_sub
    return filename

# Load preprocessed dataset
print("Loading dataset...", flush=True)
data = np.load("cached_dataset.npz", allow_pickle=True)
X_test = data["X_test"]
y_test_scaled = data["y_test"]
test_city_labels = data["test_city_labels"]

feature_scaler = joblib.load(resolve_path("feature_scaler.pkl"))
target_scaler = joblib.load(resolve_path("target_scaler.pkl"))

pm25_idx = 0
pm25_mean = feature_scaler.mean_[pm25_idx]
pm25_scale = feature_scaler.scale_[pm25_idx]

def inverse_pm25(y_scaled):
    return np.asarray(y_scaled) * pm25_scale + pm25_mean

y_test_real = inverse_pm25(y_test_scaled)
n_samples, seq_len, n_features = X_test.shape
X_test_flat = X_test.reshape(n_samples, -1)

# PyTorch Model Classes
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, 32)
        self.fc2  = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.drop(out[:, -1, :])
        return  self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, 32)
        self.fc2  = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out    = self.drop(out[:, -1, :])
        return  self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout)
        self.lstm  = nn.LSTM(64, hidden, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden, 32)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        x   = x.permute(0, 2, 1)
        x   = torch.relu(self.conv1(x))
        x   = torch.relu(self.conv2(x))
        x   = self.pool(x)
        x   = self.drop1(x)
        x   = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.drop2(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden * 2, 32)
        self.fc2  = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.drop(out[:, -1, :])
        return  self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class StackedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, 32, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(32, 32)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out    = self.drop(out[:, -1, :])
        return  self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

def predict_torch(model, X_data, batch_size=2048):
    model.eval()
    dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            p = model(bx).cpu().numpy().ravel()
            preds.append(p)
    return np.concatenate(preds)

def get_cpcb_aqi_category(pm25_vals):
    bins = [-np.inf, 30.0, 60.0, 90.0, 120.0, 250.0, np.inf]
    return np.digitize(pm25_vals, bins) - 1

y_true_cat = get_cpcb_aqi_category(y_test_real)
y_true_exceed = (y_test_real > 60.0).astype(int)

models_dict = {
    "LSTM": ("torch", "lstm_pm25.pt", LSTMModel),
    "GRU": ("torch", "gru_pm25.pt", GRUModel),
    "CNN-LSTM": ("torch", "cnn_lstm_pm25.pt", CNNLSTMModel),
    "BiLSTM": ("torch", "bilstm_pm25.pt", BiLSTMModel),
    "Stacked LSTM": ("torch", "stacked_lstm_pm25.pt", StackedLSTMModel),
    "Linear Regression": ("sklearn", "linear_regression.pkl", None),
    "Ridge Regression": ("sklearn", "ridge_regression.pkl", None),
    "Random Forest": ("sklearn", "random_forest.pkl", None),
    "XGBoost": ("sklearn", "xgboost.pkl", None),
    "LightGBM": ("sklearn", "lightgbm.pkl", None),
}

overall_results = []
per_city_results = []

for model_name, (mtype, filename, mclass) in models_dict.items():
    print(f"Evaluating {model_name}...", flush=True)
    mpath = resolve_path(filename)
    if mtype == "torch":
        model = mclass(n_features).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device))
        preds_scaled = predict_torch(model, X_test)
    else:
        model = joblib.load(mpath)
        preds_scaled = model.predict(X_test_flat)

    preds_real = inverse_pm25(preds_scaled)
    
    # Regression metrics
    mae = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    r2 = r2_score(y_test_real, preds_real)
    mape = np.mean(np.abs((y_test_real - preds_real) / np.maximum(y_test_real, 1e-5))) * 100
    evs = explained_variance_score(y_test_real, preds_real)
    medae = median_absolute_error(y_test_real, preds_real)
    mbe = np.mean(preds_real - y_test_real)
    nmbe = (np.sum(preds_real - y_test_real) / np.sum(y_test_real)) * 100
    cv_rmse = (rmse / np.mean(y_test_real)) * 100

    # Classification metrics
    preds_cat = get_cpcb_aqi_category(preds_real)
    f1_weighted = f1_score(y_true_cat, preds_cat, average="weighted")
    f1_macro = f1_score(y_true_cat, preds_cat, average="macro")
    precision_w = precision_score(y_true_cat, preds_cat, average="weighted")
    recall_w = recall_score(y_true_cat, preds_cat, average="weighted")
    acc = accuracy_score(y_true_cat, preds_cat) * 100

    # Binary exceedance metrics (> 60 ug/m3)
    preds_exceed = (preds_real > 60.0).astype(int)
    f1_exceed = f1_score(y_true_exceed, preds_exceed)
    prec_exceed = precision_score(y_true_exceed, preds_exceed)
    rec_exceed = recall_score(y_true_exceed, preds_exceed)

    overall_results.append({
        "Model": model_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
        "EVS": evs,
        "MedAE": medae,
        "MBE": mbe,
        "NMBE_%": nmbe,
        "CV_RMSE_%": cv_rmse,
        "F1_Weighted": f1_weighted,
        "F1_Macro": f1_macro,
        "Precision_Weighted": precision_w,
        "Recall_Weighted": recall_w,
        "Accuracy_%": acc,
        "Exceedance_F1": f1_exceed,
        "Exceedance_Precision": prec_exceed,
        "Exceedance_Recall": rec_exceed
    })

    # Per City
    for city in np.unique(test_city_labels):
        c_mask = (test_city_labels == city)
        c_true = y_test_real[c_mask]
        c_pred = preds_real[c_mask]
        c_true_cat = y_true_cat[c_mask]
        c_pred_cat = preds_cat[c_mask]
        c_true_ex = y_true_exceed[c_mask]
        c_pred_ex = preds_exceed[c_mask]

        per_city_results.append({
            "City": city,
            "Model": model_name,
            "R2": r2_score(c_true, c_pred),
            "MAE": mean_absolute_error(c_true, c_pred),
            "RMSE": np.sqrt(mean_squared_error(c_true, c_pred)),
            "MAPE_%": np.mean(np.abs((c_true - c_pred) / np.maximum(c_true, 1e-5))) * 100,
            "F1_Weighted": f1_score(c_true_cat, c_pred_cat, average="weighted"),
            "F1_Macro": f1_score(c_true_cat, c_pred_cat, average="macro"),
            "Accuracy_%": accuracy_score(c_true_cat, c_pred_cat) * 100,
            "Exceedance_F1": f1_score(c_true_ex, c_pred_ex),
        })

df_overall = pd.DataFrame(overall_results).sort_values("R2", ascending=False)
df_per_city = pd.DataFrame(per_city_results).sort_values(["City", "R2"], ascending=[True, False])

df_overall.to_csv(os.path.join(RESULTS_DIR, "extended_overall_benchmark_results.csv"), index=False)
df_per_city.to_csv(os.path.join(RESULTS_DIR, "extended_per_city_benchmark_results.csv"), index=False)

print("\n=== EXTENDED OVERALL BENCHMARK RESULTS ===")
print(df_overall.to_string(index=False))

print(f"\nSaved extended benchmark results to '{RESULTS_DIR}/'.")
