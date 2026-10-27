
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from model import DNNClassifier


class DNNPyfunc(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        checkpoint = torch.load(
            context.artifacts["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

        self.features = checkpoint["feature_names"]
        self.mean = np.asarray(
            checkpoint["mean"], dtype=np.float32
        )
        self.std = np.asarray(
            checkpoint["std"], dtype=np.float32
        )

        self.model = DNNClassifier(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

        missing = set(self.features) - set(model_input.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        x = model_input[self.features].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(x).all():
            raise ValueError("Features must be finite")

        x = (x - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x))
            probabilities = torch.sigmoid(logits)

        return probabilities.numpy()
