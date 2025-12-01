import requests
import json

# 20 timesteps × 67 features (scaler expects 67)
sequence = [[0.1] * 67 for _ in range(20)]

response = requests.post(
    "https://prediction-api-101264457040.europe-west3.run.app/predict",
    json={"sequence": sequence}
)
print(response.json())