"""
same_city_air_quality_prediction.py
────────────────────────────────────
Train on 5 cities (Delhi, Mumbai, Kolkata, Chennai, Bengaluru)
and test on a HELD-OUT temporal slice from the SAME 5 cities.

Usage:
    python same_city_air_quality_prediction.py          # 1-epoch smoke test
    python same_city_air_quality_prediction.py --full    # full training run
"""
import subprocess, importlib.util, sys, io, os, glob, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings("ignore")

for pkg, pip_name in [
    ("pandas","pandas"), ("numpy","numpy"), ("sklearn","scikit-learn"),
    ("xgboost","xgboost"), ("matplotlib","matplotlib"), ("seaborn","seaborn"),
    ("joblib","joblib"), ("lightgbm","lightgbm"),
]:
    if importlib.util.find_spec(pkg) is None:
        try: subprocess.run([sys.executable, "-m", "pip", "-q", "install", pip_name])
        except Exception: pass

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

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == "cuda":
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"CUDA cap : {torch.cuda.get_device_capability(0)}")
print(f"NumPy    : {np.__version__}")

# ── Mode ──────────────────────────────────────────────────────────────────────
FULL_MODE = "--full" in sys.argv
MODE_STR = "FULL TRAINING" if FULL_MODE else "1-EPOCH SMOKE TEST"
print(f"\nMode: {MODE_STR}\n")

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = "data"
CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]
LOOKBACK = 24
HORIZON = 1
TEST_FRACTION = 0.2   # last 20% of each city's time-series = test

FEATURES = ["pm25", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
GROUP_COLS = ["city", "station_name"]

CACHE_FILE = "cached_same_city_dataset.npz"

# Hyperparams: tiny for smoke test, full for real training
if FULL_MODE:
    DL_EPOCHS = 50
    DL_PATIENCE = 8
    RF_ESTIMATORS = 250
    RF_MAX_DEPTH = 18
    XGB_ESTIMATORS = 500
    LGB_ESTIMATORS = 500
else:
    DL_EPOCHS = 1
    DL_PATIENCE = 1
    RF_ESTIMATORS = 5
    RF_MAX_DEPTH = 5
    XGB_ESTIMATORS = 10
    LGB_ESTIMATORS = 10

# ── Output directories ───────────────────────────────────────────────────────
os.makedirs("models_same_city", exist_ok=True)
os.makedirs("results_same_city", exist_ok=True)
os.makedirs("plots_same_city", exist_ok=True)

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
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No data parsed from {filepath}")
    return df


def read_city_folder(folder, city):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    frames = []
    print(f"  {city}: {len(files)} CSV file(s)")
    for f in files:
        try:
            df = parse_opencity_csv(f)
            df["station_name"] = os.path.splitext(os.path.basename(f))[0]
            df["city"] = city
            frames.append(df)
        except Exception as e:
            print(f"    SKIPPED: {os.path.basename(f)} | {e}")
    if not frames:
        raise ValueError(f"No usable CSV files in {folder}")
    return pd.concat(frames, ignore_index=True)


def load_all_cities(cities):
    """Load all cities from data/<CityName>/ folders."""
    all_frames = []
    for city in cities:
        folder = os.path.join(BASE_DIR, city)
        if not os.path.isdir(folder):
            # Try train/ subfolder as fallback
            folder = os.path.join(BASE_DIR, "train", city)
        if not os.path.isdir(folder):
            # Try test/ subfolder as fallback
            folder = os.path.join(BASE_DIR, "test", city)
        all_frames.append(read_city_folder(folder, city))
    return pd.concat(all_frames, ignore_index=True)


def prepare_city_data(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["pm25"]     = pd.to_numeric(df["pm25"], errors="coerce")
    df.loc[df["pm25"] < 0,    "pm25"] = np.nan
    df.loc[df["pm25"] > 1000, "pm25"] = np.nan
    df = df.dropna(subset=["datetime", "pm25"]).copy()
    df = df.sort_values(["city", "station_name", "datetime"]).reset_index(drop=True)
    return df


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


def impute_within_groups(df):
    parts = []
    for keys, g in df.groupby(GROUP_COLS, dropna=False):
        g = g.sort_values("datetime").copy()
        g = g.set_index("datetime")
        numeric_cols = [c for c in FEATURES if c in g.columns]
        g[numeric_cols] = g[numeric_cols].interpolate(
            method="time", limit=6, limit_direction="both"
        )
        g = g.dropna(subset=["pm25"])
        for i, col in enumerate(GROUP_COLS):
            if col in g.columns:
                g[col] = g[col].iloc[0]
            else:
                g[col] = keys[i] if isinstance(keys, tuple) else keys
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def apply_limits(df):
    df = df.copy()
    df.loc[(df["pm25"] < 0) | (df["pm25"] > 1000), "pm25"] = np.nan
    return df


def make_sequences(df, lookback=24, horizon=1):
    X, y, city_labels = [], [], []
    for (city, station), g in df.groupby(GROUP_COLS):
        g = g.sort_values("datetime").reset_index(drop=True)
        values = g[FEATURES].values.astype(np.float32)
        target = g["pm25"].values.astype(np.float32)
        for i in range(lookback, len(g) - horizon + 1):
            end = i + horizon - 1
            seq = values[i - lookback : i]
            target_val = target[end]
            if np.isfinite(seq).all() and np.isfinite(target_val):
                X.append(seq)
                y.append(target_val)
                city_labels.append(city)
    return np.array(X), np.array(y), np.array(city_labels)


def temporal_train_test_split(X, y, city_labels, test_fraction=0.2):
    """Split each city's sequences temporally: early = train, late = test."""
    train_idx, test_idx = [], []
    for city in np.unique(city_labels):
        idx = np.where(city_labels == city)[0]
        cut = int(len(idx) * (1 - test_fraction))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return (X[train_idx], y[train_idx], city_labels[train_idx],
            X[test_idx],  y[test_idx],  city_labels[test_idx])


# ── Load or build dataset ────────────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    print(f"Loading cached dataset from {CACHE_FILE} ...", flush=True)
    data = np.load(CACHE_FILE, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    train_city_labels = data["train_city_labels"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    test_city_labels = data["test_city_labels"]
    feature_scaler = joblib.load("models_same_city/feature_scaler.pkl")
else:
    print("Loading and preprocessing all 5 cities ...", flush=True)
    raw = load_all_cities(CITIES)
    raw = prepare_city_data(raw)
    raw = add_time_features(raw)
    raw = impute_within_groups(raw)
    raw = apply_limits(raw)
    raw = impute_within_groups(raw)

    # Fit scaler on ALL data, then scale
    feature_scaler = StandardScaler()
    raw[FEATURES] = feature_scaler.fit_transform(raw[FEATURES])
    joblib.dump(feature_scaler, "models_same_city/feature_scaler.pkl")

    # Build sequences
    X_all, y_all, city_labels_all = make_sequences(raw, LOOKBACK, HORIZON)
    print(f"Total sequences: {X_all.shape}", flush=True)

    # Temporal train/test split within each city
    X_train, y_train, train_city_labels, X_test, y_test, test_city_labels = \
        temporal_train_test_split(X_all, y_all, city_labels_all, TEST_FRACTION)

    # Save cache
    np.savez_compressed(CACHE_FILE,
                        X_train=X_train, y_train=y_train,
                        train_city_labels=train_city_labels,
                        X_test=X_test, y_test=y_test,
                        test_city_labels=test_city_labels)
    print(f"Cached dataset saved to {CACHE_FILE}", flush=True)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")

print("\nTrain sequences by city:")
print(pd.Series(train_city_labels).value_counts())
print("\nTest sequences by city:")
print(pd.Series(test_city_labels).value_counts())

# Inverse-transform helper
pm25_idx   = FEATURES.index("pm25")
pm25_mean  = feature_scaler.mean_[pm25_idx]
pm25_scale = feature_scaler.scale_[pm25_idx]

def inverse_pm25(y_scaled):
    return np.asarray(y_scaled) * pm25_scale + pm25_mean

n_features = X_train.shape[2]

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, 32)
        self.fc2  = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, 32)
        self.fc2  = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

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
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden * 2, 32)
        self.fc2  = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)

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
        out = self.drop(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)


# ── Training & Inference ─────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs=DL_EPOCHS, batch_size=256, patience=DL_PATIENCE, lr=1e-3):
    model = model.to(device)
    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=(device.type == "cuda"), num_workers=0)
    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_val, y_val, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = TorchReduceLR(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_losses.append(criterion(model(Xb), yb).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}  (best val_loss={best_val:.6f})", flush=True)
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.6f}", flush=True)
    model.load_state_dict(best_wts)
    model.eval()
    return model


def predict_model(model, X, batch_size=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            preds.append(model(Xb).cpu().numpy())
    return np.concatenate(preds)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL SPLIT (within training set)
# ══════════════════════════════════════════════════════════════════════════════

def city_time_split(X, y, labels, val_fraction=0.2):
    train_idx, val_idx = [], []
    for city in np.unique(labels):
        idx = np.where(labels == city)[0]
        cut = int(len(idx) * (1 - val_fraction))
        train_idx.extend(idx[:cut])
        val_idx.extend(idx[cut:])
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx])

X_tr, y_tr, X_val, y_val = city_time_split(X_train, y_train, train_city_labels)
print(f"\nTrain seqs: {X_tr.shape}  Val seqs: {X_val.shape}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ALL 10 MODELS
# ══════════════════════════════════════════════════════════════════════════════

predictions = {}

# ── Deep Learning Models ──────────────────────────────────────────────────────
print(f"\n[1/10] Training LSTM on {device} ...", flush=True)
lstm_model = train_model(LSTMModel(n_features), X_tr, y_tr, X_val, y_val)
predictions["LSTM"] = predict_model(lstm_model, X_test)
torch.save(lstm_model.state_dict(), "models_same_city/lstm_pm25.pt")

print(f"\n[2/10] Training GRU on {device} ...", flush=True)
gru_model = train_model(GRUModel(n_features), X_tr, y_tr, X_val, y_val)
predictions["GRU"] = predict_model(gru_model, X_test)
torch.save(gru_model.state_dict(), "models_same_city/gru_pm25.pt")

print(f"\n[3/10] Training CNN-LSTM on {device} ...", flush=True)
cnn_lstm_model = train_model(CNNLSTMModel(n_features), X_tr, y_tr, X_val, y_val)
predictions["CNN-LSTM"] = predict_model(cnn_lstm_model, X_test)
torch.save(cnn_lstm_model.state_dict(), "models_same_city/cnn_lstm_pm25.pt")

print(f"\n[4/10] Training BiLSTM on {device} ...", flush=True)
bilstm_model = train_model(BiLSTMModel(n_features), X_tr, y_tr, X_val, y_val)
predictions["BiLSTM"] = predict_model(bilstm_model, X_test)
torch.save(bilstm_model.state_dict(), "models_same_city/bilstm_pm25.pt")

print(f"\n[5/10] Training Stacked LSTM on {device} ...", flush=True)
stacked_lstm_model = train_model(StackedLSTMModel(n_features), X_tr, y_tr, X_val, y_val)
predictions["Stacked LSTM"] = predict_model(stacked_lstm_model, X_test)
torch.save(stacked_lstm_model.state_dict(), "models_same_city/stacked_lstm_pm25.pt")

# ── Classical ML Models ───────────────────────────────────────────────────────
X_tr_flat   = X_tr.reshape(len(X_tr), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

print("\n[6/10] Training Linear Regression...", flush=True)
lr_model = LinearRegression()
lr_model.fit(X_tr_flat, y_tr)
predictions["Linear Regression"] = lr_model.predict(X_test_flat)
joblib.dump(lr_model, "models_same_city/linear_regression.pkl")

print("\n[7/10] Training Ridge Regression...", flush=True)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_tr_flat, y_tr)
predictions["Ridge Regression"] = ridge_model.predict(X_test_flat)
joblib.dump(ridge_model, "models_same_city/ridge_regression.pkl")

print("\n[8/10] Training Random Forest...", flush=True)
rf_model = RandomForestRegressor(
    n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
    min_samples_leaf=2, random_state=SEED, n_jobs=1
)
rf_model.fit(X_tr_flat, y_tr)
predictions["Random Forest"] = rf_model.predict(X_test_flat)
joblib.dump(rf_model, "models_same_city/random_forest.pkl")
print("Random Forest saved.", flush=True)

print("\n[9/10] Training XGBoost (GPU cuda)...", flush=True)
xgb_model = XGBRegressor(
    n_estimators=XGB_ESTIMATORS, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
    tree_method="hist", device="cuda", random_state=SEED
)
xgb_model.fit(X_tr_flat, y_tr)
predictions["XGBoost"] = xgb_model.predict(X_test_flat)
joblib.dump(xgb_model, "models_same_city/xgboost.pkl")
print("XGBoost saved.", flush=True)

print("\n[10/10] Training LightGBM...", flush=True)
lgb_model = lgb.LGBMRegressor(
    n_estimators=LGB_ESTIMATORS, learning_rate=0.05, num_leaves=63,
    max_depth=-1, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=1, verbose=-1
)
lgb_model.fit(X_tr_flat, y_tr)
predictions["LightGBM"] = lgb_model.predict(X_test_flat)
joblib.dump(lgb_model, "models_same_city/lightgbm.pkl")
print("LightGBM saved.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true_scaled, y_pred_scaled):
    y_true = inverse_pm25(y_true_scaled)
    y_pred = inverse_pm25(y_pred_scaled)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 1.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}

# Overall results
results = []
for model_name, pred in predictions.items():
    row = evaluate(y_test, pred)
    row["Model"] = model_name
    results.append(row)

results_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2", "MAPE_%"]].sort_values("RMSE").reset_index(drop=True)
print("\n-- Overall Results (same-city temporal test split) --")
print(results_df.to_string(index=False))

# Per-city results
city_results = []
for city in CITIES:
    mask = test_city_labels == city
    if mask.sum() == 0:
        continue
    for model_name, pred in predictions.items():
        row = evaluate(y_test[mask], pred[mask])
        row["City"]  = city
        row["Model"] = model_name
        city_results.append(row)

city_results_df = pd.DataFrame(city_results)[
    ["City", "Model", "MAE", "RMSE", "R2", "MAPE_%"]
].sort_values(["City", "RMSE"]).reset_index(drop=True)

print("\n-- Per-City Results --")
print(city_results_df.to_string(index=False))


# ── Save CSVs ─────────────────────────────────────────────────────────────────
results_df.to_csv("results_same_city/overall_results.csv", index=False)
city_results_df.to_csv("results_same_city/per_city_results.csv", index=False)


# ── Plots ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 6))
sns.barplot(data=city_results_df, x="City", y="RMSE", hue="Model")
plt.title("Same-City PM2.5 Forecasting -- RMSE")
plt.ylabel("RMSE (ug/m3)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("plots_same_city/rmse_same_cities.png")
plt.close()

plt.figure(figsize=(14, 6))
sns.barplot(data=city_results_df, x="City", y="R2", hue="Model")
plt.title("Same-City PM2.5 Forecasting -- R2")
plt.ylabel("R2")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("plots_same_city/r2_same_cities.png")
plt.close()

best_model = results_df.iloc[0]["Model"]
print(f"\nBest overall model: {best_model}")

for city in CITIES:
    mask = test_city_labels == city
    city_idx = np.where(mask)[0]
    if len(city_idx) == 0:
        continue
    pred   = inverse_pm25(predictions[best_model][city_idx])
    actual = inverse_pm25(y_test[city_idx])
    n_plot = min(500, len(actual))
    plt.figure(figsize=(14, 5))
    plt.plot(actual[:n_plot], label="Actual")
    plt.plot(pred[:n_plot],   label="Predicted")
    plt.title(f"{best_model}: PM2.5 Prediction -- {city}")
    plt.xlabel("Test sequence")
    plt.ylabel("PM2.5 (ug/m3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots_same_city/prediction_{city}_{best_model.replace(' ', '_')}.png")
    plt.close()

print(f"\nSaved models to models_same_city/")
print(f"Saved results to results_same_city/")
print(f"Saved plots to plots_same_city/")
print(f"Device used: {device}")
print("DONE.")
