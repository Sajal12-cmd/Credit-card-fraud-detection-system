import logging

from fastapi import APIRouter, HTTPException

from api.schemas import PredictionResponse, SchemaResponse, Transaction
from core.explain import get_explanation, is_shap_enabled
from core.model import get_feature_columns, get_metrics, predict, threshold


router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.get("/")
def home():
	logger.info("Health check requested")
	return {"message": "Fraud Detection API Running"}


@router.post("/predict", response_model=PredictionResponse)
def predict_route(transaction: Transaction):
	logger.info("Prediction request received")
	try:
		prediction, probability, used_threshold, processed_data = predict(transaction.features)
	except ValueError as exc:
		logger.error("Prediction failed: %s", exc)
		raise HTTPException(status_code=503, detail=str(exc)) from exc

	logger.info("Prediction = %s, Prob = %.4f, Threshold = %.4f", prediction, probability, used_threshold)

	explanation = get_explanation(processed_data)
	return {
		"prediction": prediction,
		"probability": probability,
		"threshold": used_threshold,
		"top_features": explanation,
	}


@router.get("/metrics")
def metrics_route():
	logger.info("Metrics requested")
	return get_metrics()


@router.get("/schema", response_model=SchemaResponse)
def schema_route():
	logger.info("Schema requested")
	feature_columns = get_feature_columns()
	if not feature_columns:
		raise HTTPException(status_code=503, detail="Model schema not available.")

	return {
		"feature_columns": feature_columns,
		"feature_count": len(feature_columns),
		"threshold": threshold,
		"shap_enabled": is_shap_enabled(),
		"metrics": get_metrics(),
	}
