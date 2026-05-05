"""FastAPI application — SafeDrive Predictor backend."""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Ensure the project root is on sys.path so 'backend.*' imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db, init_db
from backend.models import Vehicle, DrivingRecord, PredictionLog
from backend.schemas import (
    VehicleCreate, VehicleResponse,
    DrivingRecordCreate, DrivingRecordResponse,
    RiskPredictionRequest, RiskPredictionResponse,
    WeatherData, VINDecodeResponse,
)
from backend.api_clients import fetch_weather, full_vin_lookup
from backend.ml_model import predict_risk, score_to_level
from backend.recommendations import generate_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SafeDrive Predictor API",
    description="Real-time road risk scoring using ML, weather, and vehicle recall data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Database initialised.")
    # Pre-load / train model on startup
    from backend.ml_model import _load
    _load()
    logger.info("ML model ready.")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

@app.get("/weather/{city}", response_model=WeatherData, tags=["Weather"])
def get_weather(city: str):
    data = fetch_weather(city)
    if not data:
        raise HTTPException(status_code=502, detail=f"Could not fetch weather for '{city}'")
    return WeatherData(**data)


# ---------------------------------------------------------------------------
# VIN / Vehicle
# ---------------------------------------------------------------------------

@app.get("/vin/{vin}", response_model=VINDecodeResponse, tags=["Vehicle"])
def decode_vin_endpoint(vin: str, db: Session = Depends(get_db)):
    if len(vin) != 17:
        raise HTTPException(status_code=422, detail="VIN must be exactly 17 characters.")

    # Check cache first
    cached = db.query(Vehicle).filter(Vehicle.vin == vin.upper()).first()
    if cached:
        recalls_list = []
        if cached.recall_description:
            for comp in cached.recall_description.split("; "):
                recalls_list.append({"component": comp, "summary": ""})
        return VINDecodeResponse(
            vin=cached.vin,
            make=cached.make or "Unknown",
            model=cached.model or "Unknown",
            year=cached.year or 0,
            has_recall=cached.has_recall,
            recall_count=len(recalls_list),
            recalls=recalls_list,
        )

    result = full_vin_lookup(vin)
    if not result:
        raise HTTPException(status_code=502, detail="NHTSA API lookup failed.")

    # Persist to DB
    db_vehicle = Vehicle(
        vin=result["vin"],
        make=result["make"],
        model=result["model"],
        year=result["year"],
        has_recall=result["has_recall"],
        recall_description=result.get("recall_description"),
    )
    db.add(db_vehicle)
    db.commit()

    return VINDecodeResponse(**result)


# ---------------------------------------------------------------------------
# Driving Records
# ---------------------------------------------------------------------------

@app.post("/drivers/record", response_model=DrivingRecordResponse, tags=["Drivers"])
def create_driving_record(record: DrivingRecordCreate, db: Session = Depends(get_db)):
    db_record = DrivingRecord(**record.model_dump())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


@app.get("/drivers/{driver_id}/records", response_model=List[DrivingRecordResponse], tags=["Drivers"])
def get_driving_records(driver_id: str, db: Session = Depends(get_db)):
    records = db.query(DrivingRecord).filter(DrivingRecord.driver_id == driver_id).all()
    return records


@app.get("/drivers/{driver_id}/latest", response_model=Optional[DrivingRecordResponse], tags=["Drivers"])
def get_latest_record(driver_id: str, db: Session = Depends(get_db)):
    record = (
        db.query(DrivingRecord)
        .filter(DrivingRecord.driver_id == driver_id)
        .order_by(DrivingRecord.created_at.desc())
        .first()
    )
    return record


# ---------------------------------------------------------------------------
# Core Risk Prediction
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=RiskPredictionResponse, tags=["Prediction"])
def predict(req: RiskPredictionRequest, db: Session = Depends(get_db)):
    # 1. Resolve weather
    weather_data = None
    temp = req.temperature if req.temperature is not None else 15.0
    precip = req.precipitation if req.precipitation is not None else 0.0
    vis = req.visibility if req.visibility is not None else 10.0
    wind_speed = 0.0

    if req.city:
        raw = fetch_weather(req.city)
        if raw:
            weather_data = WeatherData(**raw)
            temp = weather_data.temperature
            precip = weather_data.precipitation
            vis = weather_data.visibility
            wind_speed = weather_data.wind_speed
        else:
            logger.warning("Weather fetch failed for city '%s'; using provided/defaults.", req.city)

    # Apply scenario overrides (what-if sliders)
    if req.scenario_precipitation is not None:
        precip = req.scenario_precipitation
    if req.scenario_visibility is not None:
        vis = req.scenario_visibility

    # 2. Resolve vehicle / VIN
    vehicle_data = None
    has_recall = False
    recalls = []
    vehicle_age = req.vehicle_age or 5

    if req.vin and len(req.vin) == 17:
        cached = db.query(Vehicle).filter(Vehicle.vin == req.vin.upper()).first()
        if cached:
            has_recall = cached.has_recall or False
            recalls = [{"component": c} for c in (cached.recall_description or "").split("; ") if c]
            year = cached.year or datetime.now().year
            vehicle_age = max(0, datetime.now().year - year)
            vehicle_data = VINDecodeResponse(
                vin=cached.vin,
                make=cached.make or "",
                model=cached.model or "",
                year=year,
                has_recall=has_recall,
                recall_count=len(recalls),
                recalls=recalls,
            )
        else:
            result = full_vin_lookup(req.vin)
            if result:
                has_recall = result.get("has_recall", False)
                recalls = result.get("recalls", [])
                year = result.get("year", datetime.now().year)
                vehicle_age = max(0, datetime.now().year - year)
                vehicle_data = VINDecodeResponse(**result)
                db_v = Vehicle(
                    vin=result["vin"],
                    make=result["make"],
                    model=result["model"],
                    year=result["year"],
                    has_recall=has_recall,
                    recall_description=result.get("recall_description"),
                )
                db.add(db_v)
                db.commit()

    hour = req.hour_of_day if req.hour_of_day is not None else datetime.now().hour

    # 3. Run ML prediction
    base_score, final_score, multipliers, importances = predict_risk(
        temperature=temp,
        precipitation=precip,
        visibility=vis,
        accidents=req.accidents_last_3yr,
        tickets=req.tickets_last_3yr,
        vehicle_age=vehicle_age,
        hour_of_day=hour,
        has_recall=has_recall,
    )

    level, color = score_to_level(final_score)

    # 4. Generate recommendations
    recs = generate_recommendations(
        risk_score=final_score,
        precipitation=precip,
        visibility=vis,
        temperature=temp,
        hour_of_day=hour,
        accidents=req.accidents_last_3yr,
        tickets=req.tickets_last_3yr,
        has_recall=has_recall,
        recalls=recalls,
        vehicle_age=vehicle_age,
        wind_speed=wind_speed,
    )

    # 5. Log prediction
    log = PredictionLog(
        driver_id=req.driver_id,
        vin=req.vin,
        temperature=temp,
        precipitation=precip,
        visibility=vis,
        hour_of_day=hour,
        accidents=req.accidents_last_3yr,
        tickets=req.tickets_last_3yr,
        has_recall=has_recall,
        vehicle_age=vehicle_age,
        base_score=round(base_score, 2),
        final_score=round(final_score, 2),
        risk_level=level,
    )
    db.add(log)
    db.commit()

    return RiskPredictionResponse(
        risk_score=round(final_score, 1),
        base_score=round(base_score, 1),
        risk_level=level,
        risk_color=color,
        applied_multipliers=multipliers,
        weather=weather_data,
        vehicle=vehicle_data,
        recommendations=recs,
        feature_importances={k: round(v, 4) for k, v in importances.items()},
    )


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------

@app.get("/history", tags=["Prediction"])
def get_prediction_history(
    driver_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, le=100),
    db: Session = Depends(get_db),
):
    q = db.query(PredictionLog)
    if driver_id:
        q = q.filter(PredictionLog.driver_id == driver_id)
    logs = q.order_by(PredictionLog.created_at.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "driver_id": l.driver_id,
            "vin": l.vin,
            "final_score": l.final_score,
            "risk_level": l.risk_level,
            "precipitation": l.precipitation,
            "visibility": l.visibility,
            "has_recall": l.has_recall,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        }
        for l in logs
    ]
