from typing import Dict, List, Union

from pydantic import BaseModel


class Transaction(BaseModel):
	features: Union[List[float], Dict[str, float]]


class FeatureImpact(BaseModel):
	feature: str
	impact: float


class PredictionResponse(BaseModel):
	prediction: str
	probability: float
	threshold: float
	top_features: List[FeatureImpact]


class SchemaResponse(BaseModel):
	feature_columns: List[str]
	feature_count: int
	threshold: float
	shap_enabled: bool
	metrics: Dict[str, float]
