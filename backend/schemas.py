from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class VehicleBase(BaseModel):
    vin: str
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    has_recall: bool = False
    recall_description: Optional[str] = None


class VehicleCreate(VehicleBase):
    pass


class VehicleResponse(VehicleBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class DrivingRecordBase(BaseModel):
    driver_id: str
    vin: Optional[str] = None
    accidents_last_3yr: int = Field(default=0, ge=0, le=20)
    tickets_last_3yr: int = Field(default=0, ge=0, le=20)
    location_city: Optional[str] = None
    notes: Optional[str] = None


class DrivingRecordCreate(DrivingRecordBase):
    pass


class DrivingRecordResponse(DrivingRecordBase):
    id: int
    risk_score: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class RiskPredictionRequest(BaseModel):
    # Weather (optional — fetched live if city provided)
    city: Optional[str] = None
    temperature: Optional[float] = None          # °F
    precipitation: Optional[float] = None        # in/hr
    visibility: Optional[float] = None           # miles

    # Driver profile
    driver_id: Optional[str] = "anonymous"
    accidents_last_3yr: int = Field(default=0, ge=0, le=20)
    tickets_last_3yr: int = Field(default=0, ge=0, le=20)

    # Vehicle
    vin: Optional[str] = None
    vehicle_age: Optional[int] = Field(default=5, ge=0, le=50)

    # Time override (0-23); if None uses current hour
    hour_of_day: Optional[int] = Field(default=None, ge=0, le=23)

    # Scenario slider overrides
    scenario_precipitation: Optional[float] = None  # override for what-if
    scenario_visibility: Optional[float] = None


class WeatherData(BaseModel):
    city: str
    country: Optional[str] = None
    temperature: float
    precipitation: float
    visibility: float
    description: str
    humidity: int
    wind_speed: float


class VINDecodeResponse(BaseModel):
    vin: str
    make: str
    model: str
    year: int
    vehicle_type: Optional[str] = None
    has_recall: bool = False
    recall_count: int = 0
    recalls: List[dict] = []


class RiskPredictionResponse(BaseModel):
    risk_score: float = Field(description="Final weighted risk score 0-100")
    base_score: float = Field(description="Raw ML model score before weighting")
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    risk_color: str = Field(description="green | yellow | orange | red")
    applied_multipliers: List[str]
    weather: Optional[WeatherData] = None
    vehicle: Optional[VINDecodeResponse] = None
    recommendations: List[dict]
    feature_importances: Optional[dict] = None
