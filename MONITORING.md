# Monitoring Setup

## Quick Start

```bash
# Build API with models
./build_docker.sh

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access services
open http://localhost:8000      # API
open http://localhost:9090      # Prometheus
open http://localhost:3000      # Grafana (admin/admin)
```

## Metrics

- **Request rate/latency**: HTTP metrics per endpoint
- **Prediction latency**: Per-model inference time
- **Prediction values**: Last prediction by model
- **Error rate**: Validation errors, 5xx errors
- **Total predictions**: Counter per model

## Endpoints

- `GET /`: Health check
- `POST /predict`: Predictions
- `GET /metrics`: Prometheus metrics

## Grafana Dashboard

Pre-configured dashboard shows:
- Request rate & error rate gauges
- Latency percentiles (p50, p95, p99) by model
- Predictions/sec by model
- Prediction values time series

Login: `admin` / `admin`

## Alerts (Optional)

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[1m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[1m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency (p95 > 1s)"
```

## Production Deployment

For Cloud Run, use Google Cloud Monitoring instead:
- Enable Cloud Monitoring API
- Metrics automatically collected
- Set up dashboards in GCP Console
