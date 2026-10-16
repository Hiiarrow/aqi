"""
cross_dataset_5cities.py
────────────────────────
Full-Epoch Cross-Dataset Benchmark with Hybrid ML + Deep Learning Ensembles.
- Training Data: Dataset 1 (OpenCity Ground Stations, 5 Cities, 1.58M records)
- Testing Data:  Dataset 2 (CAMS 2024 Satellite Reanalysis, Same 5 Cities, 43.8K records)
- Models: 5 PyTorch CUDA DL + 5 Classical/GBDT ML + Hybrid (ML + Deep Learning) Blends

Usage:
    python cross_dataset_5cities.py
"""
import sys, io, os, glob, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLR

# ── Device Detection ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == "cuda":
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"NumPy    : {np.__version__}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── City Coordinates for External Dataset 2 ──────────────────────────────────
CITIES_INFO = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
}
CITIES = list(CITIES_INFO.keys())

LOOKBACK = 24
HORIZON = 1
FEATURES = ["pm25", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
GROUP_COLS = ["city", "station_name"]

# Full Training Hyperparameters
DL_EPOCHS = 25
DL_PATIENCE = 5
DL_BATCH_SIZE = 4096
RF_ESTIMATORS = 30
RF_MAX_DEPTH = 12
XGB_ESTIMATORS = 300
LGB_ESTIMATORS = 300

os.makedirs("models_cross_dataset", exist_ok=True)
os.makedirs("results_cross_dataset", exist_ok=True)
os.makedirs("plots_cross_dataset", exist_ok=True)
os.makedirs("data_external", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_opencity_csv(filepath):
    records = []
    current_month_date = None
    hours = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "Year" or parts[0] in ("", "\\n", r"\n"):
                continue
            if re.match(r"[A-Za-z]+-\d{4}$", parts[0]):
                try:
                    current_month_date = pd.to_datetime(parts[0], format="%B-%Y")
                    hours = parts[1:]
                except Exception:
                    current_month_date = None
                continue
            if current_month_date is None:
                continue
            try:
                day = int(parts[0])
            except (ValueError, IndexError):
                continue
            for h_idx, h_str in enumerate(hours):
                col_idx = h_idx + 1
                val_str = parts[col_idx] if col_idx < len(parts) else ""
                try:
                    pm25_val = float(val_str)
                except (ValueError, TypeError):
                    pm25_val = np.nan
                try:
                    hour = pd.to_datetime(h_str, format="%H:%M:%S").hour
                    dt = pd.Timestamp(year=current_month_date.year,
                                      month=current_month_date.month,
                                      day=day, hour=hour)
                    records.append({"datetime": dt, "pm25": pm25_val})
                except Exception:
                    continue
    return pd.DataFrame(records)

def load_train_dataset(cities):
    frames = []
    for city in cities:
        folders = [
            os.path.join("data", city),
            os.path.join("data", "train", city),
            os.path.join("data", "test", city)
        ]
        city_files = []
        for fld in folders:
            if os.path.isdir(fld):
                city_files.extend(glob.glob(os.path.join(fld, "*.csv")))
        city_files = sorted(list(set(city_files)))
        print(f"  [Dataset 1: OpenCity] {city}: {len(city_files)} station files")
        for f in city_files:
            try:
                df = parse_opencity_csv(f)
                if not df.empty:
                    df["station_name"] = os.path.splitext(os.path.basename(f))[0]
                    df["city"] = city
                    frames.append(df)
            except Exception as e:
                pass
    return pd.concat(frames, ignore_index=True)

def fetch_external_dataset(cities_info, year=2024):
    frames = []
    for city, coords in cities_info.items():
        cache_csv = os.path.join("data_external", f"{city}_{year}_hourly_pm25.csv")
        if os.path.exists(cache_csv):
            print(f"  [Dataset 2: External] Loading cached {cache_csv}")
            df_city = pd.read_csv(cache_csv)
            df_city["datetime"] = pd.to_datetime(df_city["datetime"])
        else:
            print(f"  [Dataset 2: External] Fetching {city} {year} hourly PM2.5 from Open-Meteo CAMS...")
            url = (
                f"https://air-quality-api.open-meteo.com/v1/air-quality?"
                f"latitude={coords['lat']}&longitude={coords['lon']}&"
                f"hourly=pm2_5&start_date={year}-01-01&end_date={year}-12-31"
            )
            r = requests.get(url, timeout=30)
            data = r.json().get("hourly", {})
            times = data.get("time", [])
            pm25 = data.get("pm2_5", [])
            df_city = pd.DataFrame({"datetime": pd.to_datetime(times), "pm25": pm25})
            df_city["city"] = city
            df_city["station_name"] = f"CAMS_{city}"
            df_city.to_csv(cache_csv, index=False)
            print(f"  [Dataset 2: External] Saved {len(df_city)} hours to {cache_csv}")
        frames.append(df_city)
    return pd.concat(frames, ignore_index=True)

def add_time_features(df):
    df = df.copy()
    dt = df["datetime"]
    df["hour_sin"]  = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)
    return df

def clean_data(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["pm25"]     = pd.to_numeric(df["pm25"], errors="coerce")
    df.loc[(df["pm25"] < 0) | (df["pm25"] > 1000), "pm25"] = np.nan
    df = df.dropna(subset=["datetime", "pm25"]).copy()
    df = df.sort_values(["city", "station_name", "datetime"]).reset_index(drop=True)
    return df

def impute_groups(df):
    parts = []
    for keys, g in df.groupby(GROUP_COLS, dropna=False):
        g = g.sort_values("datetime").copy().set_index("datetime")
        numeric_cols = [c for c in FEATURES if c in g.columns]
        g[numeric_cols] = g[numeric_cols].interpolate(method="time", limit=6, limit_direction="both")
        g = g.dropna(subset=["pm25"])
        for i, col in enumerate(GROUP_COLS):
            g[col] = keys[i] if isinstance(keys, tuple) else keys
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)

def make_sequences(df, lookback=24, horizon=1):
    X, y, city_labels = [], [], []
    for (city, station), g in df.groupby(GROUP_COLS):
        g = g.sort_values("datetime").reset_index(drop=True)
        values = g[FEATURES].values.astype(np.float32)
        target = g["pm25"].values.astype(np.float32)
        for i in range(lookback, len(g) - horizon + 1):
            end = i + horizon - 1
            seq = values[i - lookback : i]
            t = target[end]
            if np.isfinite(seq).all() and np.isfinite(t):
                X.append(seq)
                y.append(t)
                city_labels.append(city)
    return np.array(X), np.array(y), np.array(city_labels)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD, MERGE & SCALE DATASETS
# ══════════════════════════════════════════════════════════════════════════════

CACHE_FILE = "cached_cross_dataset_5cities.npz"

if os.path.exists(CACHE_FILE):
    print(f"\nLoading combined cross-dataset arrays from {CACHE_FILE} ...", flush=True)
    d = np.load(CACHE_FILE, allow_pickle=True)
    X_train = d["X_train"]
    y_train = d["y_train"]
    train_city_labels = d["train_city_labels"]
    X_test = d["X_test"]
    y_test = d["y_test"]
    test_city_labels = d["test_city_labels"]
    feature_scaler = joblib.load("models_cross_dataset/feature_scaler.pkl")
else:
    print("\n--- Step 1: Loading Dataset 1 (Ground Station Training Data) ---", flush=True)
    df_train = load_train_dataset(CITIES)
    df_train = clean_data(df_train)
    df_train = add_time_features(df_train)
    df_train = impute_groups(df_train)

    print("\n--- Step 2: Loading Dataset 2 (External 2024 CAMS Test Data) ---", flush=True)
    df_test = fetch_external_dataset(CITIES_INFO, year=2024)
    df_test = clean_data(df_test)
    df_test = add_time_features(df_test)
    df_test = impute_groups(df_test)

    feature_scaler = StandardScaler()
    df_train[FEATURES] = feature_scaler.fit_transform(df_train[FEATURES])
    df_test[FEATURES]  = feature_scaler.transform(df_test[FEATURES])
    joblib.dump(feature_scaler, "models_cross_dataset/feature_scaler.pkl")

    print("\n--- Step 3: Generating 24-hr Lagged Sequences ---", flush=True)
    X_train, y_train, train_city_labels = make_sequences(df_train, LOOKBACK, HORIZON)
    X_test,  y_test,  test_city_labels  = make_sequences(df_test,  LOOKBACK, HORIZON)

    np.savez_compressed(CACHE_FILE,
                        X_train=X_train, y_train=y_train,
                        train_city_labels=train_city_labels,
                        X_test=X_test, y_test=y_test,
                        test_city_labels=test_city_labels)
    print(f"Saved combined cached arrays to {CACHE_FILE}", flush=True)

print(f"\nTraining Set (Dataset 1: OpenCity Stations) : X={X_train.shape}, y={y_train.shape}")
print(f"Testing Set  (Dataset 2: External CAMS 2024) : X={X_test.shape},  y={y_test.shape}")
print("\nTest sequences per city in External Dataset:")
print(pd.Series(test_city_labels).value_counts())

pm25_idx = FEATURES.index("pm25")
pm25_mean = feature_scaler.mean_[pm25_idx]
pm25_scale = feature_scaler.scale_[pm25_idx]

def inverse_pm25(y_scaled):
    return np.asarray(y_scaled) * pm25_scale + pm25_mean

n_features = X_train.shape[2]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, in_features, hidden=64, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc1  = nn.Linear(hidden, 32)
        self.fc2  = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, in_features, hidden=64, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class CNNLSTMModel(nn.Module):
    def __init__(self, in_features, hidden=64, drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(drop)
        self.lstm  = nn.LSTM(64, hidden, batch_first=True)
        self.drop2 = nn.Dropout(drop)
        self.fc1   = nn.Linear(hidden, 32)
        self.fc2   = nn.Linear(32, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.drop2(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class BiLSTMModel(nn.Module):
    def __init__(self, in_features, hidden=64, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc1  = nn.Linear(hidden * 2, 32)
        self.fc2  = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class StackedLSTMModel(nn.Module):
    def __init__(self, in_features, hidden=64, drop=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(in_features, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, 32, batch_first=True)
        self.drop  = nn.Dropout(drop)
        self.fc1   = nn.Linear(32, 32)
        self.fc2   = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

def train_torch_model(model, X_tr, y_tr, X_val, y_val, epochs=DL_EPOCHS, patience=DL_PATIENCE, batch_size=DL_BATCH_SIZE):
    model = model.to(device)
    ds_train = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    ds_val   = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=(device.type=="cuda"))
    loader_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = TorchReduceLR(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in loader_val:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(by)
        val_loss /= len(ds_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = " ⭐ (best)"
        else:
            no_improve += 1
            star = ""

        if epoch % 5 == 0 or epoch == 1 or star != "":
            print(f"    Epoch {epoch:2d}/{epochs:2d} | val_loss: {val_loss:.6f}{star}", flush=True)

        if no_improve >= patience:
            print(f"    Early stopping triggered at epoch {epoch} (best val_loss: {best_val_loss:.6f})", flush=True)
            break

    model.load_state_dict(best_weights)
    model.eval()
    return model

def predict_torch(model, X, batch_size=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            preds.append(model(bx).cpu().numpy())
    return np.concatenate(preds)

# ── Split Training set into Train/Val for PyTorch & Ensemble Stacking (90/10) ─
n_val = int(len(X_train) * 0.1)
X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
X_val, y_val = X_train[-n_val:], y_train[-n_val:]

X_tr_flat   = X_tr.reshape(len(X_tr), -1)
X_val_flat  = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# ══════════════════════════════════════════════════════════════════════════════
# FULL TRAINING: 5 DEEP LEARNING + 5 CLASSICAL ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
preds_test = {}
preds_val = {}

print("\n" + "="*70)
print(f"🔥 TRAINING PYTORCH DEEP LEARNING MODELS ({DL_EPOCHS} Max Epochs with Early Stopping on CUDA)")
print("="*70, flush=True)

print("[1/10] Training PyTorch LSTM ...", flush=True)
m_lstm = train_torch_model(LSTMModel(n_features), X_tr, y_tr, X_val, y_val)
preds_test["LSTM"] = predict_torch(m_lstm, X_test)
preds_val["LSTM"]  = predict_torch(m_lstm, X_val)
torch.save(m_lstm.state_dict(), "models_cross_dataset/lstm_pm25.pt")

print("[2/10] Training PyTorch GRU ...", flush=True)
m_gru = train_torch_model(GRUModel(n_features), X_tr, y_tr, X_val, y_val)
preds_test["GRU"] = predict_torch(m_gru, X_test)
preds_val["GRU"]  = predict_torch(m_gru, X_val)
torch.save(m_gru.state_dict(), "models_cross_dataset/gru_pm25.pt")

print("[3/10] Training PyTorch CNN-LSTM ...", flush=True)
m_cnn = train_torch_model(CNNLSTMModel(n_features), X_tr, y_tr, X_val, y_val)
preds_test["CNN-LSTM"] = predict_torch(m_cnn, X_test)
preds_val["CNN-LSTM"]  = predict_torch(m_cnn, X_val)
torch.save(m_cnn.state_dict(), "models_cross_dataset/cnn_lstm_pm25.pt")

print("[4/10] Training PyTorch BiLSTM ...", flush=True)
m_bi = train_torch_model(BiLSTMModel(n_features), X_tr, y_tr, X_val, y_val)
preds_test["BiLSTM"] = predict_torch(m_bi, X_test)
preds_val["BiLSTM"]  = predict_torch(m_bi, X_val)
torch.save(m_bi.state_dict(), "models_cross_dataset/bilstm_pm25.pt")

print("[5/10] Training PyTorch Stacked LSTM ...", flush=True)
m_stk = train_torch_model(StackedLSTMModel(n_features), X_tr, y_tr, X_val, y_val)
preds_test["Stacked LSTM"] = predict_torch(m_stk, X_test)
preds_val["Stacked LSTM"]  = predict_torch(m_stk, X_val)
torch.save(m_stk.state_dict(), "models_cross_dataset/stacked_lstm_pm25.pt")

print("\n" + "="*70)
print("🌲 TRAINING CLASSICAL ML & GRADIENT BOOSTED MODELS")
print("="*70, flush=True)

print("[6/10] Training Linear Regression ...", flush=True)
m_lr = LinearRegression()
m_lr.fit(X_tr_flat, y_tr)
preds_test["Linear Regression"] = m_lr.predict(X_test_flat)
preds_val["Linear Regression"]  = m_lr.predict(X_val_flat)
joblib.dump(m_lr, "models_cross_dataset/linear_regression.pkl")

print("[7/10] Training Ridge Regression ...", flush=True)
m_ridge = Ridge(alpha=1.0)
m_ridge.fit(X_tr_flat, y_tr)
preds_test["Ridge Regression"] = m_ridge.predict(X_test_flat)
preds_val["Ridge Regression"]  = m_ridge.predict(X_val_flat)
joblib.dump(m_ridge, "models_cross_dataset/ridge_regression.pkl")

print("[8/10] Training Random Forest (30 trees, depth 12) ...", flush=True)
m_rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=SEED, n_jobs=1)
m_rf.fit(X_tr_flat, y_tr)
preds_test["Random Forest"] = m_rf.predict(X_test_flat)
preds_val["Random Forest"]  = m_rf.predict(X_val_flat)
joblib.dump(m_rf, "models_cross_dataset/random_forest.pkl")

print("[9/10] Training XGBoost (300 trees on GPU CUDA) ...", flush=True)
m_xgb = XGBRegressor(n_estimators=XGB_ESTIMATORS, max_depth=8, learning_rate=0.05,
                     tree_method="hist", device="cuda", random_state=SEED)
m_xgb.fit(X_tr_flat, y_tr)
preds_test["XGBoost"] = m_xgb.predict(X_test_flat)
preds_val["XGBoost"]  = m_xgb.predict(X_val_flat)
joblib.dump(m_xgb, "models_cross_dataset/xgboost.pkl")

print("[10/10] Training LightGBM (300 trees, 63 leaves) ...", flush=True)
m_lgb = lgb.LGBMRegressor(n_estimators=LGB_ESTIMATORS, learning_rate=0.05, num_leaves=63,
                          random_state=SEED, n_jobs=4, verbose=-1)
m_lgb.fit(X_tr_flat, y_tr)
preds_test["LightGBM"] = m_lgb.predict(X_test_flat)
preds_val["LightGBM"]  = m_lgb.predict(X_val_flat)
joblib.dump(m_lgb, "models_cross_dataset/lightgbm.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# HYBRID ENSEMBLES (COMBINING ML + DEEP LEARNING)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🤝 BUILDING HYBRID ENSEMBLES (ML + DEEP LEARNING FUSION)")
print("="*70, flush=True)

# 1. Hybrid: LightGBM + LSTM
preds_test["Hybrid: LightGBM + LSTM"] = 0.5 * preds_test["LightGBM"] + 0.5 * preds_test["LSTM"]

# 2. Hybrid: XGBoost + GRU
preds_test["Hybrid: XGBoost + GRU"] = 0.5 * preds_test["XGBoost"] + 0.5 * preds_test["GRU"]

# 3. Hybrid: Ridge + BiLSTM
preds_test["Hybrid: Ridge + BiLSTM"] = 0.5 * preds_test["Ridge Regression"] + 0.5 * preds_test["BiLSTM"]

# 4. Hybrid: LightGBM + CNN-LSTM
preds_test["Hybrid: LightGBM + CNN-LSTM"] = 0.5 * preds_test["LightGBM"] + 0.5 * preds_test["CNN-LSTM"]

# 5. Hybrid Stacking: Optimal Learned Weights on Validation Set (Best ML + Best DL)
ml_models = ["LightGBM", "XGBoost", "Random Forest", "Ridge Regression", "Linear Regression"]
dl_models = ["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Stacked LSTM"]

best_ml = min(ml_models, key=lambda m: mean_squared_error(y_val, preds_val[m]))
best_dl = min(dl_models, key=lambda m: mean_squared_error(y_val, preds_val[m]))

# Find optimal alpha in [0, 1] minimizing MSE on validation set
best_alpha = 0.5
best_val_mse = float("inf")
for alpha in np.linspace(0.0, 1.0, 101):
    blend_val = alpha * preds_val[best_ml] + (1 - alpha) * preds_val[best_dl]
    mse = mean_squared_error(y_val, blend_val)
    if mse < best_val_mse:
        best_val_mse = mse
        best_alpha = alpha

hybrid_opt_name = f"Hybrid (Optimal: {best_alpha:.2f}*{best_ml} + {1-best_alpha:.2f}*{best_dl})"
preds_test[hybrid_opt_name] = best_alpha * preds_test[best_ml] + (1 - best_alpha) * preds_test[best_dl]
print(f"Optimal Hybrid Blend Found on Validation: {best_alpha*100:.1f}% {best_ml} + {(1-best_alpha)*100:.1f}% {best_dl}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE CROSS-DATASET BENCHMARK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def eval_metrics(y_true_s, y_pred_s):
    yt = inverse_pm25(y_true_s)
    yp = inverse_pm25(y_pred_s)
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    valid = np.abs(yt) > 1.0
    mape = np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100 if valid.sum() > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}

# Categorize model types
def get_model_category(name):
    if name.startswith("Hybrid"):
        return "Hybrid (ML + DL)"
    elif name in ["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Stacked LSTM"]:
        return "Deep Learning (PyTorch CUDA)"
    else:
        return "Machine Learning / GBDT"

res_list = []
for name, p in preds_test.items():
    row = eval_metrics(y_test, p)
    row["Model"] = name
    row["Type"] = get_model_category(name)
    res_list.append(row)

df_overall = pd.DataFrame(res_list)[["Model", "Type", "MAE", "RMSE", "R2", "MAPE_%"]].sort_values("RMSE").reset_index(drop=True)
print("\n" + "="*70)
print("🏆 OVERALL BENCHMARK RESULTS (Matching on External 2024 CAMS Dataset)")
print("="*70)
print(df_overall.to_string(index=False))

# Per-city breakdown on external dataset 2
city_res = []
for city in CITIES:
    c_mask = (test_city_labels == city)
    for name, p in preds_test.items():
        row = eval_metrics(y_test[c_mask], p[c_mask])
        row["City"] = city
        row["Model"] = name
        row["Type"] = get_model_category(name)
        city_res.append(row)

df_per_city = pd.DataFrame(city_res)[["City", "Model", "Type", "MAE", "RMSE", "R2", "MAPE_%"]].sort_values(["City", "RMSE"]).reset_index(drop=True)
print("\n" + "="*70)
print("🏙️ PER-CITY BENCHMARK RESULTS (Matching on External 2024 CAMS Dataset)")
print("="*70)
print(df_per_city.to_string(index=False))

# Save results
df_overall.to_csv("results_cross_dataset/overall_results.csv", index=False)
df_per_city.to_csv("results_cross_dataset/per_city_results.csv", index=False)

# Plots
plt.figure(figsize=(16, 7))
sns.barplot(data=df_overall, x="Model", y="RMSE", hue="Type", dodge=False)
plt.title("Cross-Dataset Full-Epoch PM2.5 Forecast: ML vs Deep Learning vs Hybrid (RMSE)")
plt.ylabel("RMSE (ug/m3)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots_cross_dataset/rmse_cross_dataset.png")
plt.close()

plt.figure(figsize=(16, 7))
sns.barplot(data=df_overall, x="Model", y="R2", hue="Type", dodge=False)
plt.title("Cross-Dataset Full-Epoch PM2.5 Forecast: ML vs Deep Learning vs Hybrid (R2)")
plt.ylabel("R2 Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots_cross_dataset/r2_cross_dataset.png")
plt.close()

print("\nSaved models to 'models_cross_dataset/'")
print("Saved metric tables to 'results_cross_dataset/'")
print("Saved comparison plots to 'plots_cross_dataset/'")
print("✅ Full-epoch hybrid training & cross-dataset benchmark complete!")
