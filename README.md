# Local MLOps Lifecycle Demo

An end-to-end local MLOps learning project demonstrating data versioning, experiment tracking, model registration, reproducibility, and containerized inference.

## Tech Stack

- **Conda** — Python development environment
- **DuckDB + Parquet** — Data preparation
- **Git + DVC** — Source code and dataset versioning
- **PyTorch** — Binary classification model
- **MLflow** — Experiment tracking and Model Registry
- **FastAPI + Docker** — Model serving

## Workflow

1. Generate a synthetic binary classification dataset (1,000 samples, 8 features).
2. Prepare train/validation/test datasets using DuckDB.
3. Track and restore dataset versions using Git and DVC.
4. Train PyTorch models and log experiments with MLflow.
5. Register validated models and assign a `champion` alias.
6. Reproduce a historical experiment using its Git commit, DVC snapshot, and training configuration.
7. Export a fixed model version and serve predictions with FastAPI and Docker.

## Getting Started

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate dnn-mlops
```

Generate and prepare data:

```bash
python scripts/generate_data.py
python scripts/prepare.py
```

Or restore a tracked dataset snapshot using DVC, provided the configured DVC remote is accessible:

```bash
dvc pull
```

Start a local MLflow server from the project root:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Run a training experiment:

```bash
python -m scripts.train --hidden-dim 16 --epochs 30
```

View experiments at `http://127.0.0.1:5000`.

## Model Serving

After registering a model and assigning the `champion` alias, export the selected version:

```bash
python -m scripts.export_model
```

Run the FastAPI service locally:

```bash
python -m uvicorn app.main:app --port 8000
```

Or build and run the Docker image:

```bash
docker build -t dnn-serving:v1 .
docker run --rm -p 127.0.0.1:8001:8000 dnn-serving:v1
```

Test the API:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0,0,0,0,0,0,0,0]}'
```

With both local and Docker services running, compare prediction consistency:

```bash
python tests/test_serving_parity.py
```

## Reproducibility

A historical MLflow training run was reproduced locally using the same Git commit, DVC data snapshot, random seed, and training parameters. The recorded training loss, validation loss, accuracy, and AUC matched to eight decimal places.

## Limitations

This is an educational MLOps demonstration, not a production deployment. The DVC remote and MLflow Registry are currently local; their stored data and model artifacts are not included in this Git repository. A fresh clone requires accessible artifact storage or regeneration of the required assets.

The project uses synthetic data and does not include live traffic, production monitoring, or automated retraining.