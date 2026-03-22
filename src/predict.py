import joblib
import numpy as np
import pandas as pd

from preprocessing import apply_preprocessor

MODEL_PATH = "models/fraud_model.pkl"
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
preprocessor = artifact["preprocessor"]
threshold = max(float(artifact.get("threshold", 0.5)), 0.05)


def _as_feature_dataframe(data):
	"""Convert raw feature values to a one-row dataframe aligned with training features."""
	if isinstance(data, dict):
		return pd.DataFrame([data])

	if isinstance(data, (list, tuple, np.ndarray)):
		expected = len(preprocessor["feature_columns"])
		if len(data) != expected:
			raise ValueError(f"Expected {expected} feature values, got {len(data)}.")
		return pd.DataFrame([list(data)], columns=preprocessor["feature_columns"])

	raise ValueError("Input data must be a dict or list/tuple/ndarray of features.")


def predict(data):
	feature_df = _as_feature_dataframe(data)
	processed = apply_preprocessor(feature_df, preprocessor)
	prob = float(model.predict_proba(processed)[0][1])
	return {
		"prediction": "Fraud" if prob >= threshold else "Legit",
		"probability": prob,
		"threshold": threshold,
	}

