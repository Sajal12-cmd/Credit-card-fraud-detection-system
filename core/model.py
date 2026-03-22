from pathlib import Path
from typing import Dict, List, Union
import logging

import joblib
import pandas as pd

import sys
from core.config import ROOT_DIR, settings

sys.path.append(str(ROOT_DIR / "src"))

from preprocessing import apply_preprocessor


logger = logging.getLogger(__name__)


MODEL_PATH = ROOT_DIR / settings.MODEL_PATH

if MODEL_PATH.exists():
	logger.info("Loading model artifact from %s", MODEL_PATH)
	artifact = joblib.load(MODEL_PATH)
	model = artifact["model"]
	preprocessor = artifact["preprocessor"]
	artifact_threshold = float(artifact.get("threshold", 0.5))
	threshold = settings.THRESHOLD if settings.THRESHOLD is not None else max(artifact_threshold, settings.MIN_THRESHOLD)
	metrics = artifact.get("metrics", {})
else:
	logger.warning("Model artifact not found at %s", MODEL_PATH)
	artifact = None
	model = None
	preprocessor = None
	threshold = 0.5
	metrics = {}


def _to_feature_df(features: Union[List[float], Dict[str, float]]) -> pd.DataFrame:
	if preprocessor is None:
		raise ValueError("Preprocessor is not available. Train and save the model first.")

	feature_columns = preprocessor.get("feature_columns", [])
	if isinstance(features, dict):
		missing = [col for col in feature_columns if col not in features]
		if missing:
			raise ValueError(f"Missing required feature keys: {missing}")
		return pd.DataFrame([features])[feature_columns]

	expected = len(feature_columns)
	if len(features) != expected:
		raise ValueError(f"Expected {expected} feature values, got {len(features)}.")

	return pd.DataFrame([list(features)], columns=feature_columns)


def predict(features: Union[List[float], Dict[str, float]]):
	if model is None:
		raise ValueError(f"Model file not found at {MODEL_PATH}")

	feature_df = _to_feature_df(features)
	processed = apply_preprocessor(feature_df, preprocessor)
	probability = float(model.predict_proba(processed)[0][1])
	prediction = "Fraud" if probability >= threshold else "Legit"
	return prediction, probability, threshold, processed


def get_feature_columns() -> List[str]:
	if preprocessor is None:
		return []
	return list(preprocessor.get("feature_columns", []))


def get_metrics() -> Dict[str, float]:
	return metrics if isinstance(metrics, dict) else {}
