from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "safedrive.db")

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from backend.models import User, DrivingRecord, Vehicle, PredictionLog  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate()


def _migrate():
    """Add new columns to existing tables without dropping data."""
    migrations = [
        "ALTER TABLE prediction_logs ADD COLUMN user_id INTEGER REFERENCES users(id)",
        "ALTER TABLE prediction_logs ADD COLUMN city TEXT",
        "ALTER TABLE users ADD COLUMN location TEXT",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                pass  # Column already exists
