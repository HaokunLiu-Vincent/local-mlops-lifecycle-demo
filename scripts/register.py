
import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[1]

# Make the local src/ modules importable
sys.path.insert(0, str(ROOT / "src"))

from serving_model import DNNPyfunc  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    source_run = client.get_run(args.run_id)

    if source_run.info.status != "FINISHED":
        raise ValueError("Source run is not FINISHED")

    if "val_auc" not in source_run.data.metrics:
        raise ValueError("Source run has no val_auc metric")

    if source_run.data.tags.get(
        "git_worktree_dirty"
    ) != "False":
        raise ValueError(
            "Source run was not recorded from a clean Git worktree"
        )

    # Download the checkpoint produced by training
    checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=args.run_id,
        artifact_path="checkpoint/checkpoint.pt",
    )

    # Example raw input, without label or row_id
    example = pd.DataFrame(
        np.zeros((2, 8), dtype=np.float32),
        columns=[f"f{i}" for i in range(8)],
    )

    # Predict returns one probability per row
    signature = infer_signature(
        example,
        np.array([0.5, 0.5], dtype=np.float32),
    )

    with mlflow.start_run(
        run_name=f"package-{args.run_id[:8]}"
    ):
        mlflow.set_tags({
            "source_training_run_id": args.run_id,
            "source_git_commit": source_run.data.tags.get(
                "git_commit", "unknown"
            ),
            "source_dvc_pointer_sha256": source_run.data.tags.get(
                "dvc_processed_pointer_sha256", "unknown"
            ),
        })

        model_info = mlflow.pyfunc.log_model(
            name="classifier",
            python_model=DNNPyfunc(),
            code_paths=[str(ROOT / "src")],
            artifacts={
                "checkpoint": checkpoint_path
            },
            input_example=example,
            signature=signature,
        )

        # Register the packaged MLflow model
        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="binary-classifier",
        )

    # Explicitly retain the original training-run link
    client.set_model_version_tag(
        "binary-classifier",
        registered.version,
        "source_training_run_id",
        args.run_id,
    )

    print("Registered model: binary-classifier")
    print(f"Version: {registered.version}")
    print(f"Training Run: {args.run_id}")
    print(f"Model URI: models:/binary-classifier/{registered.version}")


if __name__ == "__main__":
    main()
