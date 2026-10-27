
import json
from urllib.request import Request, urlopen

import numpy as np

LOCAL = "http://127.0.0.1:8000/predict"
DOCKER = "http://127.0.0.1:8001/predict"

inputs = [
    [0.0] * 8,
    [1.0] * 8,
    [-1.0] * 8,
    [0.2, -0.4, 0.8, 1.1, -0.7, 0.3, 0.5, -0.2],
]


def predict(url, features):
    request = Request(
        url,
        data=json.dumps({"features": features}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


for i, features in enumerate(inputs, 1):
    local = predict(LOCAL, features)
    docker = predict(DOCKER, features)

    difference = abs(
        local["probability"] - docker["probability"]
    )

    print(f"Case {i}: difference = {difference:.10f}")

    assert np.isclose(
        local["probability"],
        docker["probability"],
        atol=1e-5,
        rtol=0,
    )
    assert local["predicted_class"] == docker["predicted_class"]

print("All golden prediction tests passed!")
