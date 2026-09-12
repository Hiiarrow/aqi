import subprocess, importlib.util, sys
for pkg, pip_name in [
    ("pandas","pandas"), ("numpy","numpy"), ("sklearn","scikit-learn"),
    ("xgboost","xgboost"), ("matplotlib","matplotlib"), ("seaborn","seaborn"),
    ("joblib","joblib"), ("lightgbm","lightgbm"), ("prophet","prophet"),
    ("statsmodels","statsmodels"),
]:
    if importlib.util.find_spec(pkg) is None:
        try:
            subprocess.run([sys.executable, "-m", "pip", "-q", "install", pip_name])
        except Exception:
            pass


import os
import glob
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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


# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = "data"

TRAIN_CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai"]
TEST_CITIES = ["Bengaluru", "Hyderabad"]

LOOKBACK = 24       # previous 24 hours
HORIZON = 1         # predict PM2.5 one hour ahead

os.makedirs(BASE_DIR, exist_ok=True)

for city in TRAIN_CITIES:
    os.makedirs(os.path.join(BASE_DIR, "train", city), exist_ok=True)
for city in TEST_CITIES:
    os.makedirs(os.path.join(BASE_DIR, "test", city), exist_ok=True)

print("Put downloaded CSV files into:")
for city in TRAIN_CITIES + TEST_CITIES:
    split = "train" if city in TRAIN_CITIES else "test"
    print(f"  {BASE_DIR}/{split}/{city}/")


def parse_opencity_csv(filepath):
    """
    Parse an OpenCity wide-format PM2.5 CSV into a long-format DataFrame.

    Format:
        Year,2017
        January-2017,00:00:00,01:00:00,...,23:00:00
        1,97.0,82.0,...
        2,55.0,...
        \\n   <- separator rows skipped
        Year,2018
        ...
    Returns DataFrame with columns: datetime (Timestamp), pm25 (float).
    """
    records = []
    current_month_date = None
    hours = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            parts = [p.strip() for p in line.split(",")]

            # Skip "Year,YYYY" header rows
            if parts[0] == "Year":
                continue

            # Skip blank / separator rows
            if parts[0] in ("", "\\n", r"\n"):
                continue

            # Month-header row: "January-2017", "February-2018", ...
            if re.match(r"[A-Za-z]+-\d{4}$", parts[0]):
                try:
                    current_month_date = pd.to_datetime(parts[0], format="%B-%Y")
                    hours = parts[1:]   # ["00:00:00", "01:00:00", ...]
                except Exception:
                    current_month_date = None
                continue

            # Data row: day_of_month, val_h0, val_h1, ..., val_h23
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
                    dt = pd.Timestamp(
                        year=current_month_date.year,
                        month=current_month_date.month,
                        day=day,
                        hour=hour,
                    )
                    records.append({"datetime": dt, "pm25": pm25_val})
                except Exception:
                    continue

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No data could be parsed from {filepath}")
    return df


def read_city_folder(folder, city):
    """Load all CSVs in a city folder and concatenate them."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}. "
            f"Put the city's OpenCity CSV files in this folder first."
        )

    frames = []
    print(f"\n{city}: {len(files)} CSV file(s)")
    for f in files:
        try:
            df = parse_opencity_csv(f)
            df["station_name"] = os.path.splitext(os.path.basename(f))[0]
            df["city"] = city
            frames.append(df)
            print(f"  loaded: {os.path.basename(f)}  {df.shape}")
        except Exception as e:
            print(f"  SKIPPED: {os.path.basename(f)} | {e}")

    if not frames:
        raise ValueError(f"Could not read any usable CSV files in {folder}")
    return pd.concat(frames, ignore_index=True)


def load_split(cities, split):
    all_frames = []
    for city in cities:
        folder = os.path.join(BASE_DIR, split, city)
        all_frames.append(read_city_folder(folder, city))
    return pd.concat(all_frames, ignore_index=True)


train_raw = load_split(TRAIN_CITIES, "train")
test_raw  = load_split(TEST_CITIES,  "test")

print("\nTrain raw shape:", train_raw.shape)
print("Test raw shape :", test_raw.shape)

# Inspect columns before preprocessing
print("TRAIN columns:")
print(sorted(train_raw.columns.tolist()))

print("\nTEST columns:")
print(sorted(test_raw.columns.tolist()))

print("\nTraining cities:")
print(train_raw["city"].value_counts())

print("\nTesting cities:")
print(test_raw["city"].value_counts())


def prepare_city_data(df):
    df = df.copy()

    # Ensure correct dtypes
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["pm25"]     = pd.to_numeric(df["pm25"], errors="coerce")

    # Drop physically impossible values
    df.loc[df["pm25"] < 0,    "pm25"] = np.nan
    df.loc[df["pm25"] > 1000, "pm25"] = np.nan

    df = df.dropna(subset=["datetime", "pm25"]).copy()
    df = df.sort_values(["city", "station_name", "datetime"]).reset_index(drop=True)
    return df


train = prepare_city_data(train_raw)
test  = prepare_city_data(test_raw)

print("Train after cleaning:", train.shape)
print("Test  after cleaning :", test.shape)

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


train = add_time_features(train)
test  = add_time_features(test)

# Only PM2.5 + cyclic time features are available in these CSVs
FEATURES = [
    "pm25",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
]

# Keep only features present in both splits (future-proof)
FEATURES = [f for f in FEATURES if f in train.columns and f in test.columns]

print("Features used:", FEATURES)

GROUP_COLS = ["city", "station_name"]


def impute_within_groups(df):
    """Interpolate short gaps inside each station time series."""
    parts = []
    for keys, g in df.groupby(GROUP_COLS, dropna=False):
        g = g.sort_values("datetime").copy()
        g = g.set_index("datetime")

        numeric_cols = [c for c in FEATURES if c in g.columns]
        g[numeric_cols] = g[numeric_cols].interpolate(
            method="time", limit=6, limit_direction="both"
        )
        g = g.dropna(subset=["pm25"])

        # Restore group identifier columns
        for i, col in enumerate(GROUP_COLS):
            if col in g.columns:
                g[col] = g[col].iloc[0]
            else:
                g[col] = keys[i] if isinstance(keys, tuple) else keys

        parts.append(g.reset_index())

    return pd.concat(parts, ignore_index=True)


train_p = impute_within_groups(train)
test_p  = impute_within_groups(test)

print("Prepared train:", train_p.shape)
print("Prepared test :", test_p.shape)

# Physically implausible PM2.5 -> NaN, then re-impute short gaps
def apply_limits(df):
    df = df.copy()
    df.loc[(df["pm25"] < 0) | (df["pm25"] > 1000), "pm25"] = np.nan
    return df

train_p = apply_limits(train_p)
test_p  = apply_limits(test_p)

# Re-impute short gaps after removing invalid values
train_p = impute_within_groups(train_p)
test_p  = impute_within_groups(test_p)

# Fit scalers ONLY on training-city data.
feature_scaler = StandardScaler()
target_scaler  = StandardScaler()

train_p[FEATURES] = feature_scaler.fit_transform(train_p[FEATURES])
test_p[FEATURES]  = feature_scaler.transform(test_p[FEATURES])

target_scaler.fit(train_p[["pm25"]])

joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler,  "target_scaler.pkl")

pm25_idx = FEATURES.index("pm25")
print("Scalers fitted using TRAINING CITIES ONLY.")
print(f"PM2.5 mean={feature_scaler.mean_[pm25_idx]:.2f}  std={feature_scaler.scale_[pm25_idx]:.2f}")

def make_sequences(df, lookback=24, horizon=1):
    """Build (X, y) sequences grouped by city+station so time series are not mixed."""
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


X_train, y_train, train_city_labels = make_sequences(train_p, LOOKBACK, HORIZON)
X_test,  y_test,  test_city_labels  = make_sequences(test_p,  LOOKBACK, HORIZON)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nTest sequences by city:")
print(pd.Series(test_city_labels).value_counts())

# Inverse-transform helpers
pm25_idx   = FEATURES.index("pm25")
pm25_mean  = feature_scaler.mean_[pm25_idx]
pm25_scale = feature_scaler.scale_[pm25_idx]


def inverse_pm25(y_scaled):
    """Convert scaled PM2.5 back to original ug/m3 units."""
    return np.asarray(y_scaled) * pm25_scale + pm25_mean


# Alias kept for backwards compatibility with the rest of the notebook
def inverse_pm25_from_feature_scaler(y_scaled):
    return inverse_pm25(y_scaled)

n_features = X_train.shape[2]

# ── PyTorch Model Definitions ─────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden, batch_first=True)
        self.drop    = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden, 32)
        self.fc2     = nn.Linear(32, 1)

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
    """Conv1D × 2 → MaxPool → LSTM → FC"""
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.conv1  = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2  = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool   = nn.MaxPool1d(kernel_size=2)
        self.drop1  = nn.Dropout(dropout)
        self.lstm   = nn.LSTM(64, hidden, batch_first=True)
        self.drop2  = nn.Dropout(dropout)
        self.fc1    = nn.Linear(hidden, 32)
        self.fc2    = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq, features) → permute for Conv1d
        x   = x.permute(0, 2, 1)
        x   = torch.relu(self.conv1(x))
        x   = torch.relu(self.conv2(x))
        x   = self.pool(x)
        x   = self.drop1(x)
        x   = x.permute(0, 2, 1)          # back to (batch, seq/2, 64)
        out, _ = self.lstm(x)
        out = self.drop2(out[:, -1, :])
        return self.fc2(torch.relu(self.fc1(out))).squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True,
                            bidirectional=True)
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


# ── Training & Inference Utilities ────────────────────────────────────────────

def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs=50, batch_size=256, patience=8, lr=1e-3):
    """
    Train a PyTorch model with Adam, MSE loss, ReduceLROnPlateau,
    and EarlyStopping (restoring best weights). Runs on `device`.
    """
    model = model.to(device)

    def make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float()
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=(device.type == "cuda"),
                          num_workers=0)

    train_loader = make_loader(X_tr,  y_tr,  shuffle=True)
    val_loader   = make_loader(X_val, y_val, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = TorchReduceLR(optimizer, mode="min", factor=0.5,
                              patience=4, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val  = float("inf")
    best_wts  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        # ── validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_losses.append(criterion(model(Xb), yb).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_wts   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}  (best val_loss={best_val:.6f})")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.6f}")

    model.load_state_dict(best_wts)
    model.eval()
    return model


def predict_model(model, X, batch_size=512):
    """Batched inference; returns 1-D numpy array."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            preds.append(model(Xb).cpu().numpy())
    return np.concatenate(preds)


# Train/validation split INSIDE TRAINING CITIES only.
# The validation set is the latest 20% of sequences from each training city.
# No Bengaluru/Hyderabad data is used here.

def city_time_split(X, y, labels, val_fraction=0.2):
    train_idx, val_idx = [], []

    for city in np.unique(labels):
        idx = np.where(labels == city)[0]
        cut = int(len(idx) * (1 - val_fraction))
        train_idx.extend(idx[:cut])
        val_idx.extend(idx[cut:])

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx]
    )

X_tr, y_tr, X_val, y_val = city_time_split(
    X_train, y_train, train_city_labels
)

print("Train seqs:", X_tr.shape, " Val seqs:", X_val.shape)

print(f"\nTraining LSTM on {device} ...")
lstm_model = train_model(LSTMModel(n_features), X_tr, y_tr, X_val, y_val)

print(f"\nTraining GRU on {device} ...")
gru_model = train_model(GRUModel(n_features), X_tr, y_tr, X_val, y_val)

print(f"\nTraining CNN-LSTM on {device} ...")
cnn_lstm_model = train_model(CNNLSTMModel(n_features), X_tr, y_tr, X_val, y_val)

print(f"\nTraining BiLSTM on {device} ...")
bilstm_model = train_model(BiLSTMModel(n_features), X_tr, y_tr, X_val, y_val)

print(f"\nTraining Stacked LSTM on {device} ...")
stacked_lstm_model = train_model(StackedLSTMModel(n_features), X_tr, y_tr, X_val, y_val)



# Classical ML models require 2-D input.
# Flatten the 24-hour sequence into one feature vector.
X_tr_flat   = X_tr.reshape(len(X_tr), -1)
X_val_flat  = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

lr_model    = LinearRegression()
ridge_model = Ridge(alpha=1.0)
rf_model    = RandomForestRegressor(
    n_estimators=250,
    max_depth=18,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1
)
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1
)
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

print("Training Linear Regression...")
lr_model.fit(X_tr_flat, y_tr)

print("Training Ridge Regression...")
ridge_model.fit(X_tr_flat, y_tr)

print("Training Random Forest...")
rf_model.fit(X_tr_flat, y_tr)


print("Training XGBoost...")
xgb_model.fit(X_tr_flat, y_tr)

print("Training LightGBM...")
lgb_model.fit(X_tr_flat, y_tr)

joblib.dump(lr_model,    "linear_regression.pkl")
joblib.dump(ridge_model, "ridge_regression.pkl")
joblib.dump(rf_model,    "random_forest.pkl")
joblib.dump(xgb_model,   "xgboost.pkl")
joblib.dump(lgb_model,   "lightgbm.pkl")


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_predictions(y_true_scaled, y_pred_scaled):
    """Evaluate on full arrays (no NaN expected)."""
    y_true = inverse_pm25_from_feature_scaler(y_true_scaled)
    y_pred = inverse_pm25_from_feature_scaler(y_pred_scaled)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    mask = np.abs(y_true) > 1.0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.sum() > 0 else np.nan
    )
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}


def evaluate_predictions_subset(y_true_scaled, y_pred_scaled):
    """Evaluate ignoring NaN entries (for ARIMA/Prophet subset runs)."""
    valid   = ~np.isnan(y_pred_scaled)
    n_valid = valid.sum()
    if n_valid == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "MAPE_%": np.nan}
    return evaluate_predictions(y_true_scaled[valid], y_pred_scaled[valid])


# ── Predictions ───────────────────────────────────────────────────────────────

predictions = {}

# Classical ML
predictions["Linear Regression"] = lr_model.predict(X_test_flat)
predictions["Ridge Regression"]  = ridge_model.predict(X_test_flat)
predictions["Random Forest"]     = rf_model.predict(X_test_flat)
predictions["XGBoost"]           = xgb_model.predict(X_test_flat)
predictions["LightGBM"]          = lgb_model.predict(X_test_flat)

# Deep Learning
predictions["LSTM"]         = predict_model(lstm_model,         X_test)
predictions["GRU"]          = predict_model(gru_model,          X_test)
predictions["CNN-LSTM"]     = predict_model(cnn_lstm_model,     X_test)
predictions["BiLSTM"]       = predict_model(bilstm_model,       X_test)
predictions["Stacked LSTM"] = predict_model(stacked_lstm_model, X_test)


# Rolling-window ARIMA and Prophet (subset)
print("\nRunning ARIMA rolling-window evaluation...")
arima_preds, _   = predict_arima_rolling(X_test, n_samples=2000)
predictions["ARIMA (window)"] = arima_preds

print("\nRunning Prophet rolling-window evaluation...")
prophet_preds, _ = predict_prophet_rolling(X_test, n_samples=2000)
predictions["Prophet (window)"] = prophet_preds


SUBSET_MODELS = {"ARIMA (window)", "Prophet (window)"}

# ── Overall results ───────────────────────────────────────────────────────────

results = []
for model_name, pred in predictions.items():
    if model_name in SUBSET_MODELS:
        row = evaluate_predictions_subset(y_test, pred)
    else:
        row = evaluate_predictions(y_test, pred)
    row["Model"] = model_name
    results.append(row)

results_df = pd.DataFrame(results)[
    ["Model", "MAE", "RMSE", "R2", "MAPE_%"]
].sort_values("RMSE").reset_index(drop=True)

print("\n── Overall Results ──")
print(results_df.to_string(index=False))


# ── Per-city results ──────────────────────────────────────────────────────────

city_results = []
for city in TEST_CITIES:
    mask = test_city_labels == city
    for model_name, pred in predictions.items():
        if model_name in SUBSET_MODELS:
            row = evaluate_predictions_subset(y_test[mask], pred[mask])
        else:
            row = evaluate_predictions(y_test[mask], pred[mask])
        row["City"]  = city
        row["Model"] = model_name
        city_results.append(row)

city_results_df = pd.DataFrame(city_results)[
    ["City", "Model", "MAE", "RMSE", "R2", "MAPE_%"]
].sort_values(["City", "RMSE"]).reset_index(drop=True)

print("\n── Per-City Results ──")
print(city_results_df.to_string(index=False))


# ── Plots ─────────────────────────────────────────────────────────────────────

plt.figure(figsize=(14, 6))
sns.barplot(data=city_results_df, x="City", y="RMSE", hue="Model")
plt.title("Cross-City PM2.5 Forecasting — RMSE on Unseen Cities")
plt.ylabel("RMSE (µg/m³)")
plt.xlabel("Unseen test city")
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=city_results_df, x="City", y="R2", hue="Model")
plt.title("Cross-City PM2.5 Forecasting — R² on Unseen Cities")
plt.ylabel("R²")
plt.xlabel("Unseen test city")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# Detailed plot for the best full-dataset model
best_model = (
    results_df[~results_df["Model"].isin(SUBSET_MODELS)]
    .iloc[0]["Model"]
)
print("Best overall unseen-city model:", best_model)

for city in TEST_CITIES:
    mask     = test_city_labels == city
    city_idx = np.where(mask)[0]

    if len(city_idx) == 0:
        continue

    pred   = predictions[best_model][city_idx]
    actual = y_test[city_idx]

    actual = inverse_pm25_from_feature_scaler(actual)
    pred   = inverse_pm25_from_feature_scaler(pred)

    n_plot = min(500, len(actual))

    plt.figure(figsize=(14, 5))
    plt.plot(actual[:n_plot], label="Actual")
    plt.plot(pred[:n_plot],   label="Predicted")
    plt.title(f"{best_model}: PM2.5 Prediction — Unseen City: {city}")
    plt.xlabel("Test sequence")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Save models ───────────────────────────────────────────────────────────────

torch.save(lstm_model.state_dict(),         "lstm_pm25.pt")
torch.save(gru_model.state_dict(),          "gru_pm25.pt")
torch.save(cnn_lstm_model.state_dict(),     "cnn_lstm_pm25.pt")
torch.save(bilstm_model.state_dict(),       "bilstm_pm25.pt")
torch.save(stacked_lstm_model.state_dict(), "stacked_lstm_pm25.pt")

results_df.to_csv("overall_unseen_city_results.csv", index=False)
city_results_df.to_csv("per_city_unseen_results.csv", index=False)

print("Saved models, scalers and result tables.")
print(f"Device used: {device}")


