
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()
model_name = "binary-classifier"

versions = client.search_model_versions(
    f"name='{model_name}'"
)

for version in versions:
    print(
        f"Version {version.version}, "
        f"Source: {version.source}"
    )

# Replace this with the version you just registered
version_number = "2"

model = mlflow.pyfunc.load_model(
    f"models:/binary-classifier@champion"
)

example = pd.DataFrame(
    np.zeros((2, 8), dtype=np.float32),
    columns=[f"f{i}" for i in range(8)],
)

predictions = model.predict(example)

assert len(predictions) == 2
assert np.isfinite(predictions).all()
assert ((predictions >= 0) & (predictions <= 1)).all()

print("Predicted probabilities:", predictions)
print("Registry load test passed!")
