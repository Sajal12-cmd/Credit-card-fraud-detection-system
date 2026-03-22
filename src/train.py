import os
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import apply_preprocessor, build_preprocessor, load_data, split_features_target


def select_recall_focused_threshold(y_true, y_proba, min_recall=0.90, min_threshold=0.05):
	"""Pick threshold prioritizing fewer missed frauds, then fewer false alarms."""
	thresholds = np.unique(np.clip(y_proba, 0.0, 1.0))
	thresholds = thresholds[thresholds >= min_threshold]
	if len(thresholds) == 0:
		return 0.5
	best_threshold = 0.5
	best_key = None

	for t in thresholds:
		y_pred = (y_proba >= t).astype(int)
		cm = confusion_matrix(y_true, y_pred)
		fn = int(cm[1, 0])
		fp = int(cm[0, 1])
		recall = recall_score(y_true, y_pred, zero_division=0)
		precision = precision_score(y_true, y_pred, zero_division=0)

		# Primary objective: minimize false negatives (missed frauds).
		# Secondary objective: minimize false positives while keeping recall high.
		if recall >= min_recall:
			key = (fn, fp, -precision)
			if best_key is None or key < best_key:
				best_key = key
				best_threshold = float(t)

	if best_key is not None:
		return best_threshold

	# Fallback if min_recall is unattainable.
	best_fallback = (float("inf"), float("inf"), float("inf"))
	for t in thresholds:
		y_pred = (y_proba >= t).astype(int)
		cm = confusion_matrix(y_true, y_pred)
		fn = int(cm[1, 0])
		fp = int(cm[0, 1])
		recall = recall_score(y_true, y_pred, zero_division=0)
		key = (fn, fp, -recall)
		if key < best_fallback:
			best_fallback = key
			best_threshold = float(t)

	return best_threshold


def train_model(data_path="data/creditcard.csv", model_path="models/fraud_model.pkl"):
	"""Train a fraud detection model and save it to disk."""
	df = load_data(data_path)
	X, y = split_features_target(df)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	preprocessor = build_preprocessor(X_train)
	X_train_processed = apply_preprocessor(X_train, preprocessor)
	X_test_processed = apply_preprocessor(X_test, preprocessor)

	model = RandomForestClassifier(
		n_estimators=300,
		random_state=42,
		n_jobs=-1,
		class_weight="balanced",
	)
	model.fit(X_train_processed, y_train)

	y_proba = model.predict_proba(X_test_processed)[:, 1]
	best_threshold = max(select_recall_focused_threshold(y_test, y_proba, min_recall=0.90), 0.05)
	y_pred = (y_proba >= best_threshold).astype(int)

	metrics = {
		"threshold": float(best_threshold),
		"recall": float(recall_score(y_test, y_pred)),
		"precision": float(precision_score(y_test, y_pred, zero_division=0)),
		"f1": float(f1_score(y_test, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_test, y_proba)),
		"false_negatives": int(confusion_matrix(y_test, y_pred)[1, 0]),
		"false_positives": int(confusion_matrix(y_test, y_pred)[0, 1]),
	}

	print("Model evaluation on test set:")
	print(classification_report(y_test, y_pred, digits=4))
	print("Confusion matrix:")
	print(confusion_matrix(y_test, y_pred))
	print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
	print(f"Selected threshold: {metrics['threshold']:.4f}")
	print(f"Recall (fraud catch rate): {metrics['recall']:.4f}")
	print(f"False negatives (missed frauds): {metrics['false_negatives']}")
	print(f"False positives (false alarms): {metrics['false_positives']}")

	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	artifact = {
		"model": model,
		"preprocessor": preprocessor,
		"threshold": best_threshold,
		"metrics": metrics,
	}
	joblib.dump(artifact, model_path)

	metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	print(f"Model trained and saved to: {model_path}")
	print(f"Metrics saved to: {metrics_path}")
	return artifact


if __name__ == "__main__":
	train_model()
