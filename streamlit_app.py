import io

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"


def get_json(url, timeout=5):
	response = requests.get(url, timeout=timeout)
	response.raise_for_status()
	return response.json()


def post_json(url, payload, timeout=10):
	response = requests.post(url, json=payload, timeout=timeout)
	response.raise_for_status()
	return response.json()


def load_schema(api_url):
	data = get_json(f"{api_url}/schema")
	feature_columns = data.get("feature_columns", [])
	if not feature_columns:
		raise ValueError("API returned empty feature schema.")
	return data


st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.caption("Recall-focused fraud screening with FastAPI backend")

if "schema" not in st.session_state:
	st.session_state.schema = None


with st.sidebar:
	st.header("Connection")
	api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL).rstrip("/")
	timeout = st.slider("Request timeout (seconds)", min_value=2, max_value=30, value=8)

	if st.button("Check API"):
		try:
			health = get_json(f"{api_url}/", timeout=timeout)
			st.success(f"API reachable: {health.get('message', 'OK')}")
		except Exception as exc:
			st.error(f"API check failed: {exc}")

	if st.button("Load Model Schema"):
		try:
			st.session_state.schema = load_schema(api_url)
			st.success("Schema loaded successfully")
		except Exception as exc:
			st.error(f"Failed to load schema: {exc}")

	if st.session_state.schema:
		metrics = st.session_state.schema.get("metrics", {})
		st.subheader("Model Info")
		st.write(f"Feature count: {st.session_state.schema.get('feature_count', 0)}")
		st.write(f"Threshold: {st.session_state.schema.get('threshold', 0.5):.4f}")
		if metrics:
			st.write(f"Recall: {metrics.get('recall', 0.0):.4f}")
			st.write(f"False Negatives: {metrics.get('false_negatives', 0)}")


if not st.session_state.schema:
	st.info("Click 'Load Model Schema' in the sidebar to initialize input fields.")
	st.stop()


feature_columns = st.session_state.schema["feature_columns"]

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch CSV Prediction"])


with tab_single:
	st.subheader("Manual Transaction Input")
	st.write("Enter feature values and send a prediction request to FastAPI.")

	input_values = {}
	ui_columns = st.columns(3)
	for idx, feature_name in enumerate(feature_columns):
		col = ui_columns[idx % 3]
		default_value = 0.0
		if feature_name.lower() == "amount":
			default_value = 50.0
		input_values[feature_name] = col.number_input(
			feature_name,
			value=float(default_value),
			format="%.6f",
		)

	if st.button("Predict Single Transaction", type="primary"):
		try:
			result = post_json(
				f"{api_url}/predict",
				{"features": input_values},
				timeout=timeout,
			)
			probability = float(result.get("probability", 0.0))
			decision = result.get("prediction", "Unknown")
			decision_threshold = float(result.get("threshold", 0.5))

			col1, col2, col3 = st.columns(3)
			col1.metric("Prediction", decision)
			col2.metric("Fraud Probability", f"{probability:.4f}")
			col3.metric("Threshold", f"{decision_threshold:.4f}")

			if decision.lower() == "fraud":
				st.warning("Transaction flagged as potential fraud.")
			else:
				st.success("Transaction appears legitimate.")

			st.subheader("Top Contributing Features")
			top_features = result.get("top_features", [])
			if not top_features:
				st.info("No SHAP feature impacts returned by API.")
			else:
				for item in top_features:
					impact = float(item["impact"])
					if impact >= 0:
						direction = "pushes toward Fraud"
						color = "#b91c1c"
						sign = "+"
					else:
						direction = "pushes toward Legit"
						color = "#166534"
						sign = ""

					st.markdown(
						f"<span style='color:{color}; font-weight:600'>{item['feature']}</span> "
						f"-> impact: {sign}{impact:.3f} ({direction})",
						unsafe_allow_html=True,
					)

				impact_df = pd.DataFrame(top_features)
				impact_df["impact"] = impact_df["impact"].astype(float)
				impact_df = impact_df.sort_values("impact", ascending=True)

				st.caption("SHAP impact chart (positive increases fraud likelihood, negative decreases it)")
				st.bar_chart(
					impact_df.set_index("feature")["impact"],
					horizontal=True,
					use_container_width=True,
				)
		except Exception as exc:
			st.error(f"Prediction failed: {exc}")


with tab_batch:
	st.subheader("Batch Scoring from CSV")
	st.write("Upload a CSV with all required feature columns to score multiple transactions.")

	uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
	if uploaded_file is not None:
		try:
			df = pd.read_csv(uploaded_file)
			required = set(feature_columns)
			available = set(df.columns)
			missing = sorted(required - available)
			if missing:
				st.error(f"Missing required columns: {missing}")
			else:
				st.success(f"CSV valid. Rows detected: {len(df)}")
				preview_count = min(5, len(df))
				st.dataframe(df.head(preview_count), use_container_width=True)

				if st.button("Run Batch Prediction"):
					results = []
					for _, row in df.iterrows():
						payload = {"features": {col: float(row[col]) for col in feature_columns}}
						pred = post_json(f"{api_url}/predict", payload, timeout=timeout)
						results.append(
							{
								"prediction": pred.get("prediction"),
								"probability": float(pred.get("probability", 0.0)),
								"threshold": float(pred.get("threshold", 0.5)),
							}
						)

					result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
					st.dataframe(result_df.head(min(20, len(result_df))), use_container_width=True)

					fraud_count = int((result_df["prediction"] == "Fraud").sum())
					st.info(f"Flagged {fraud_count} out of {len(result_df)} rows as Fraud.")

					buffer = io.StringIO()
					result_df.to_csv(buffer, index=False)
					st.download_button(
						"Download Results CSV",
						data=buffer.getvalue(),
						file_name="fraud_predictions.csv",
						mime="text/csv",
					)
		except Exception as exc:
			st.error(f"CSV processing failed: {exc}")
