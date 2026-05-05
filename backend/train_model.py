"""
Generate synthetic training data and train the Random Forest risk model.
All units are US Imperial:
  temperature   -> °F  (-20 to 120)
  precipitation -> in/hr (0 to 2.0)
  visibility    -> miles (0.1 to 6.2)
  wind_speed    -> mph
Run directly: python -m backend.train_model
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

FEATURE_NAMES = [
    "temperature",       # °F  (-20 to 120)
    "precipitation",     # in/hr (0 to 2.0)
    "visibility",        # miles (0.1 to 6.2)
    "hour_of_day",       # 0-23
    "accidents",         # count last 3yr
    "tickets",           # count last 3yr
    "vehicle_age",       # years
    "is_highway",        # 0/1
    "road_condition",    # 0=dry 1=wet 2=icy
]

np.random.seed(42)
N = 8_000


def _compute_base_risk(df: pd.DataFrame) -> np.ndarray:
    """Deterministic risk formula used to label synthetic data (Imperial units)."""
    risk = np.zeros(len(df))

    # Cold temps increase risk — risk rises below 40°F, peaks at -20°F
    risk += np.clip((40 - df["temperature"]) / 60, 0, 1) * 12

    # Precipitation: 2.0 in/hr = max rain risk
    risk += np.clip(df["precipitation"] / 2.0, 0, 1) * 25

    # Low visibility: 6.2 miles = clear, 0 = max risk
    risk += np.clip((6.2 - df["visibility"]) / 6.2, 0, 1) * 20

    # Nighttime (10pm–6am) adds risk
    night_mask = (df["hour_of_day"] <= 6) | (df["hour_of_day"] >= 22)
    risk += night_mask.astype(float) * 10

    # Driver history
    risk += np.clip(df["accidents"] * 8, 0, 30)
    risk += np.clip(df["tickets"] * 4, 0, 20)

    # Vehicle age
    risk += np.clip(df["vehicle_age"] / 50, 0, 1) * 10

    # Road condition
    risk += df["road_condition"] * 8

    # Highway
    risk += df["is_highway"] * 5

    # Noise
    risk += np.random.normal(0, 4, len(df))
    return np.clip(risk, 0, 100)


def generate_synthetic_data() -> pd.DataFrame:
    data = {
        # US temperature range: -20°F (extreme cold) to 120°F (desert heat)
        "temperature": np.random.uniform(-20, 120, N),
        # Precipitation in in/hr: exponential distribution, most events < 0.5 in/hr
        "precipitation": np.random.exponential(0.08, N).clip(0, 2.0),
        # Visibility in miles: 0.1 (dense fog) to 6.2 (10 km clear)
        "visibility": np.random.uniform(0.1, 6.2, N),
        "hour_of_day": np.random.randint(0, 24, N),
        "accidents": np.random.choice([0, 1, 2, 3, 4], N, p=[0.55, 0.25, 0.12, 0.05, 0.03]),
        "tickets": np.random.choice([0, 1, 2, 3, 4, 5], N, p=[0.45, 0.25, 0.15, 0.08, 0.04, 0.03]),
        "vehicle_age": np.random.randint(0, 30, N),
        "is_highway": np.random.randint(0, 2, N),
        "road_condition": np.random.choice([0, 1, 2], N, p=[0.60, 0.30, 0.10]),
    }
    df = pd.DataFrame(data)
    df["risk_score"] = _compute_base_risk(df)
    return df


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Generating synthetic training data (Imperial units)...")
    df = generate_synthetic_data()

    X = df[FEATURE_NAMES]
    y = df["risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE : {mae:.2f}")
    print(f"  R2  : {r2:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved  -> {MODEL_PATH}")
    print(f"Scaler saved -> {SCALER_PATH}")

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    return model, scaler, importances


if __name__ == "__main__":
    train()
