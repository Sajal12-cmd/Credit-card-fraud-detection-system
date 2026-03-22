# Credit Card Fraud Detection

Production-style fraud detection project with modular API, model, and explainability layers.

## Project Structure

- `api/app.py`: FastAPI entry point
- `api/schemas.py`: Request/response models
- `api/routes.py`: API endpoints
- `core/model.py`: Model loading and prediction logic
- `core/explain.py`: SHAP explainability logic
- `models/fraud_model.pkl`: Trained model artifact
- `streamlit_app.py`: Frontend dashboard
- `requirements.txt`: Python dependencies

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start API:

```bash
uvicorn api.app:app --reload
```

3. Start Streamlit UI in a second terminal:

```bash
streamlit run streamlit_app.py
```

## Environment Configuration

1. Copy environment template:

```bash
cp .env.example .env
```

2. Configure values in `.env` if needed:

- `APP_NAME`, `APP_VERSION`
- `LOG_LEVEL`
- `API_HOST`, `API_PORT`
- `STREAMLIT_PORT`
- `MODEL_PATH`
- `MIN_THRESHOLD`

## API Endpoints

- `GET /`: Health check
- `POST /predict`: Predict fraud and return SHAP top features
- `GET /metrics`: Model evaluation metrics
- `GET /schema`: Model feature schema for clients

## Docker

Build and run both services:

```bash
docker compose up --build
```

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## CI

GitHub Actions workflow is available at `.github/workflows/ci.yml` and performs:

- dependency installation
- Python compile check
- FastAPI import smoke test
