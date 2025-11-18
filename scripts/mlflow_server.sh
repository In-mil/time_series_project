#!/bin/bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root gs://time-series-dvc-storage/mlflow \
    --host 0.0.0.0 \
    --port 5001
