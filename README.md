# Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)

Production-style credit card fraud detection project with:

- FastAPI inference service
- Streamlit interactive UI
- model training and preprocessing pipeline
- SHAP explainability support
- Dockerized local deployment

## Project Structure

```text
.
|-- api/
|   |-- app.py
|   |-- routes.py
|   `-- schemas.py
|-- core/
|   |-- config.py
|   |-- explain.py
|   |-- logging_config.py
|   `-- model.py
|-- data/
|   `-- creditcard.csv               # local-only (ignored by git)
|-- models/
|   |-- metrics.json
|   `-- metrics_dump.json
|-- src/
|   |-- predict.py
|   |-- preprocessing.py
|   `-- train.py
|-- .dockerignore
|-- .env.example
|-- .gitignore
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
`-- streamlit_app.py
```

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Create environment file.

```bash
copy .env.example .env
```

3. Run the API.

```bash
uvicorn api.app:app --reload
```

4. Run Streamlit in another terminal.

```bash
streamlit run streamlit_app.py
```

## API Endpoints

- `GET /`: service health check
- `POST /predict`: fraud prediction with top feature explanations
- `GET /metrics`: model evaluation metrics
- `GET /schema`: model feature schema

## Docker

Run the full stack:

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## Training Pipeline

- `src/preprocessing.py`: preprocessing and feature prep
- `src/train.py`: training and metric export
- `src/predict.py`: local prediction utility

## CI

GitHub Actions workflow is defined in `.github/workflows/ci.yml` and runs:

- dependency installation
- Python compile check
- FastAPI import smoke test
