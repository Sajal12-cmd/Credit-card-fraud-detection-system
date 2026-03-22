import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
	"""Load dataset from CSV file."""
	df = pd.read_csv(path)
	return df


def split_features_target(df):
	"""Split dataset into feature matrix X and target y."""
	if "Class" not in df.columns:
		raise ValueError("Target column 'Class' is missing from the dataset.")

	X = df.drop("Class", axis=1)
	y = df["Class"]
	return X, y


def build_preprocessor(X_train):
	"""Fit and return preprocessing components based on training features."""
	feature_columns = [c for c in X_train.columns if c != "Time"]

	scaler = StandardScaler()
	if "Amount" in X_train.columns:
		scaler.fit(X_train[["Amount"]])

	return {
		"feature_columns": feature_columns,
		"scaler": scaler,
		"scale_amount": "Amount" in X_train.columns,
	}


def apply_preprocessor(X, preprocessor):
	"""Apply fitted preprocessing components to a feature dataframe."""
	processed = X.copy()

	if "Time" in processed.columns:
		processed = processed.drop(["Time"], axis=1)

	if preprocessor.get("scale_amount") and "Amount" in processed.columns:
		scaler = preprocessor["scaler"]
		processed["Amount"] = scaler.transform(processed[["Amount"]])

	return processed[preprocessor["feature_columns"]]


def preprocess(df):
	"""Legacy helper: fit and apply preprocessing on a full dataframe."""
	if "Class" in df.columns:
		X, y = split_features_target(df)
		preprocessor = build_preprocessor(X)
		processed_X = apply_preprocessor(X, preprocessor)
		processed = processed_X.copy()
		processed["Class"] = y.values
		return processed

	preprocessor = build_preprocessor(df)
	return apply_preprocessor(df, preprocessor)


if __name__ == "__main__":
	input_path = "data/creditcard.csv"
	output_path = "data/creditcard_preprocessed.csv"

	data = load_data(input_path)
	preprocessed_data = preprocess(data)
	preprocessed_data.to_csv(output_path, index=False)

	print("Preprocessing complete.")
	print(f"Original shape: {data.shape}")
	print(f"Processed shape: {preprocessed_data.shape}")
	print(f"Saved to: {output_path}")
