"""
ML inference layer: load the trained model, apply dynamic weighting rules,
and return a final 0-100 risk score.

All inputs are expected in US Imperial units:
  temperature   -> °F
  precipitation -> in/hr
  visibility    -> miles

Weighting rules (applied AFTER the base RF prediction):
  - +20 % if vehicle has an active safety recall
  - +15 % if it is nighttime (hour < 6 or hour >= 21)
  - +30 % if precipitation > 0.2 in/hr AND driver has a 'high-ticket' record (>= 3 tickets)
"""

import os
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "risk_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "scaler.pkl")

FEATURE_NAMES = [
    "temperature",
    "precipitation",
    "visibility",
    "hour_of_day",
    "accidents",
    "tickets",
    "vehicle_age",
    "is_highway",
    "road_condition",
]

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            from backend.train_model import train
            _model, _scaler, _ = train()
        else:
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)


def _road_condition(precipitation: float, temperature: float) -> int:
    """Infer road surface condition from weather (Imperial: °F, in/hr)."""
    if precipitation <= 0:
        return 0  # dry
    if temperature <= 32:
        return 2  # icy — freezing point in °F
    return 1  # wet


def predict_risk(
    temperature: float,
    precipitation: float,
    visibility: float,
    accidents: int,
    tickets: int,
    vehicle_age: int = 5,
    hour_of_day: int | None = None,
    has_recall: bool = False,
    is_highway: bool = False,
) -> Tuple[float, float, List[str], Dict[str, float]]:
    """
    Returns (base_score, final_score, applied_multipliers, feature_importances).

    base_score  — raw RF output (0-100)
    final_score — after dynamic weighting (capped at 100)
    """
    _load()

    if hour_of_day is None:
        hour_of_day = datetime.now().hour

    road_cond = _road_condition(precipitation, temperature)

    features = np.array([[
        temperature,
        precipitation,
        visibility,
        hour_of_day,
        accidents,
        tickets,
        vehicle_age,
        int(is_highway),
        road_cond,
    ]])

    scaled = _scaler.transform(features)
    base_score: float = float(np.clip(_model.predict(scaled)[0], 0, 100))

    # ---- Dynamic weighting rules ----------------------------------------
    multiplier = 1.0
    applied: List[str] = []

    if has_recall:
        multiplier += 0.20
        applied.append("recall_penalty_+20%")

    is_night = (hour_of_day < 6) or (hour_of_day >= 21)
    if is_night:
        multiplier += 0.15
        applied.append("nighttime_penalty_+15%")

    high_ticket = tickets >= 3
    if precipitation > 0.2 and high_ticket:  # 0.2 in/hr ~= 5mm/hr
        multiplier += 0.30
        applied.append("heavy_rain_x_high_ticket_+30%")

    final_score = float(np.clip(base_score * multiplier, 0, 100))

    # Feature importances
    importances = dict(zip(FEATURE_NAMES, _model.feature_importances_.tolist()))

    return base_score, final_score, applied, importances


def score_to_level(score: float) -> Tuple[str, str]:
    """Map numeric score to (level_label, color_name)."""
    if score < 30:
        return "LOW", "green"
    if score < 55:
        return "MEDIUM", "yellow"
    if score < 75:
        return "HIGH", "orange"
    return "CRITICAL", "red"
