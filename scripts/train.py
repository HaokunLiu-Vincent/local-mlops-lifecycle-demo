
import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.model import DNNClassifier


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"f{i}" for i in range(8)]


def load_data():
    train = pd.read_parquet(ROOT / "data/processed/train.parquet")
    val = pd.read_parquet(ROOT / "data/processed/val.parquet")

    x_train = train[FEATURES].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.float32)

    x_val = val[FEATURES].to_numpy(dtype=np.float32)
    y_val = val["label"].to_numpy(dtype=np.float32)

    # Fit preprocessing ONLY on training data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_val),
        y_val,
        mean,
        std,
    )


def get_git_info():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip()
    )

    return commit, dirty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    x_train, y_train, x_val, y_val, mean, std = load_data()

    model = DNNClassifier(
        input_dim=len(FEATURES),
        hidden_dim=args.hidden_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("dnn-classification-demo")

    git_commit, is_dirty = get_git_info()

    # Hash of the DVC pointer file for the processed dataset
    dvc_file = ROOT / "data/processed.dvc"
    dvc_pointer_hash = hashlib.sha256(
        dvc_file.read_bytes()
    ).hexdigest()

    with mlflow.start_run(
        run_name=f"dnn-hidden-{args.hidden_dim}"
    ) as run:

        mlflow.log_params({
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "seed": 42,
            "dataset": "Movie-independent synthetic dataset",
        })

        mlflow.set_tags({
            "git_commit": git_commit,
            "git_worktree_dirty": str(is_dirty),
            "dvc_processed_pointer_sha256": dvc_pointer_hash,
            "model_type": "PyTorch DNN",
        })

        for epoch in range(args.epochs):
            model.train()

            optimizer.zero_grad()

            logits = model(x_train)
            loss = loss_fn(logits, y_train)

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                val_probs = torch.sigmoid(
                    val_logits
                ).numpy()

                val_loss = loss_fn(
                    val_logits,
                    torch.from_numpy(y_val),
                ).item()

            val_predictions = (val_probs >= 0.5).astype(int)

            val_accuracy = accuracy_score(
                y_val, val_predictions
            )
            val_auc = roc_auc_score(y_val, val_probs)

            mlflow.log_metrics(
                {
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_auc": val_auc,
                },
                step=epoch + 1,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:02d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Val Acc {val_accuracy:.4f} | "
                    f"Val AUC {val_auc:.4f}"
                )

        # Save model AND preprocessing statistics together
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": len(FEATURES),
                    "hidden_dim": args.hidden_dim,
                    "feature_names": FEATURES,
                    "mean": mean,
                    "std": std,
                },
                checkpoint_path,
            )

            mlflow.log_artifact(
                str(checkpoint_path),
                artifact_path="checkpoint",
            )

        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"Final validation accuracy: {val_accuracy:.4f}")
        print(f"Final validation AUC: {val_auc:.4f}")


if __name__ == "__main__":
    main()
