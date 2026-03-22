import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class Settings:
	APP_NAME = os.getenv("APP_NAME", "Fraud Detection API")
	APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
	LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
	API_HOST = os.getenv("API_HOST", "0.0.0.0")
	API_PORT = int(os.getenv("API_PORT", "8000"))
	STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
	MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
	THRESHOLD = float(os.getenv("THRESHOLD")) if os.getenv("THRESHOLD") else None
	MIN_THRESHOLD = float(os.getenv("MIN_THRESHOLD", "0.05"))


settings = Settings()
