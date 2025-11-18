# Utility Scripts

Standalone scripts for various project tasks.

## Contents
- `check_drift.py` - Monitors model drift
- `prepare_reference_data.py` - Prepares baseline data for drift detection
- `predict_real_data.py` - Makes predictions on real-world data
- `test_deployed_api.py` - Integration tests for deployed API
- `mlflow_server.sh` - Launches MLflow tracking server
- `delayed_push.sh` - Git helper utility

## Usage
```bash
# From project root
python scripts/check_drift.py
python scripts/test_deployed_api.py <API_URL>
./scripts/mlflow_server.sh
```
