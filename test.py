import mlflow

mlflow.set_tracking_uri("https://mlflow-server-101264457040.europe-west3.run.app")
mlflow.set_experiment("test-experiment")

with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("test", "hello cloud!")
    mlflow.log_metric("value", 42)

print("Done! Check the MLflow UI.")