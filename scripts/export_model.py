
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
MODEL_DIR = RELEASE_DIR / "model"

MODEL_NAME = "binary-classifier"
MODEL_ALIAS = "champion"


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    # Resolve moving alias into an immutable version URI
    model_uri = (
        f"models:/{MODEL_NAME}/{version.version}"
    )

    print("Exporting:", model_uri)

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally overwriting an existing release
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} already exists. "
            "Remove it explicitly before exporting again."
        )

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=tmp,
        )

        shutil.copytree(downloaded, MODEL_DIR)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": version.version,
        "model_uri": model_uri,
    }

    (RELEASE_DIR / "manifest.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Release exported:", MODEL_DIR)
    print("Manifest:", metadata)


if __name__ == "__main__":
    main()
