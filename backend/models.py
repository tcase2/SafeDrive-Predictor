from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.sql import func
from backend.database import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    vin = Column(String(17), unique=True, index=True, nullable=False)
    make = Column(String(100))
    model = Column(String(100))
    year = Column(Integer)
    has_recall = Column(Boolean, default=False)
    recall_description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DrivingRecord(Base):
    __tablename__ = "driving_records"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(String(50), index=True, nullable=False)
    vin = Column(String(17), nullable=True)
    accidents_last_3yr = Column(Integer, default=0)
    tickets_last_3yr = Column(Integer, default=0)
    risk_score = Column(Float, nullable=True)
    location_city = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(String(50), nullable=True)
    vin = Column(String(17), nullable=True)
    temperature = Column(Float)
    precipitation = Column(Float)
    visibility = Column(Float)
    hour_of_day = Column(Integer)
    accidents = Column(Integer)
    tickets = Column(Integer)
    has_recall = Column(Boolean)
    vehicle_age = Column(Integer)
    base_score = Column(Float)
    final_score = Column(Float)
    risk_level = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
