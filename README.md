# Install conda env
conda create -n dnn-mlops python=3.11 pip -y
conda activate dnn-mlops

python -m pip install \
  torch numpy pandas scikit-learn \
  duckdb pyarrow mlflow dvc \
  fastapi uvicorn pytest
  