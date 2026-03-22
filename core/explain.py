from typing import Dict, List
import logging

import numpy as np
import shap

from core.model import model


logger = logging.getLogger(__name__)
explainer = shap.Explainer(model) if model is not None else None


def is_shap_enabled() -> bool:
	return explainer is not None


def get_explanation(processed_data, top_k: int = 5) -> List[Dict[str, float]]:
	if explainer is None:
		logger.warning("SHAP explainer unavailable; returning empty explanation")
		return []

	shap_values = explainer(processed_data)
	values = np.array(shap_values.values)

	if values.ndim == 3:
		class_idx = 1 if values.shape[2] > 1 else 0
		row_values = values[0, :, class_idx]
	elif values.ndim == 2:
		row_values = values[0, :]
	else:
		return []

	feature_names = list(processed_data.columns)
	feature_importance = sorted(
		zip(feature_names, row_values),
		key=lambda x: abs(float(x[1])),
		reverse=True,
	)[:top_k]

	return [{"feature": f, "impact": float(v)} for f, v in feature_importance]
