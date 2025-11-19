#!/bin/bash

# Query PostgreSQL predictions database
# Shows prediction statistics for all models as PERCENTAGE CHANGES

echo "========================================="
echo "Prediction Database Statistics"
echo "========================================="
echo ""

# Check if database is running
if ! docker ps --filter "name=postgres-timeseries" --format "{{.Names}}" | grep -q postgres-timeseries; then
    echo "❌ PostgreSQL container is not running!"
    echo "   Start it with: docker-compose up -d postgres"
    exit 1
fi

echo "📊 Average Predictions (5-day price change forecast)"
echo "-----------------------------------------"
PGPASSWORD=timeseries123 psql -h localhost -U timeseries -d predictions -c "
SELECT
    'ANN' as model,
    COUNT(*) as predictions,
    ROUND((AVG(prediction_ann) * 11.36 + 0.56)::numeric, 2) || '%' as avg_pct_change
FROM predictions
UNION ALL
SELECT
    'GRU',
    COUNT(*),
    ROUND((AVG(prediction_gru) * 11.36 + 0.56)::numeric, 2) || '%'
FROM predictions
UNION ALL
SELECT
    'LSTM',
    COUNT(*),
    ROUND((AVG(prediction_lstm) * 11.36 + 0.56)::numeric, 2) || '%'
FROM predictions
UNION ALL
SELECT
    'Transformer',
    COUNT(*),
    ROUND((AVG(prediction_transformer) * 11.36 + 0.56)::numeric, 2) || '%'
FROM predictions
UNION ALL
SELECT
    'Ensemble',
    COUNT(*),
    ROUND((AVG(prediction_ensemble) * 11.36 + 0.56)::numeric, 2) || '%'
FROM predictions
ORDER BY model;"

echo ""
echo "📈 Model Latency Performance"
echo "-----------------------------------------"
PGPASSWORD=timeseries123 psql -h localhost -U timeseries -d predictions -c "
SELECT
    'ANN' as model,
    ROUND(AVG(latency_ann_ms)::numeric, 2) as avg_latency_ms,
    ROUND(MIN(latency_ann_ms)::numeric, 2) as min_latency_ms,
    ROUND(MAX(latency_ann_ms)::numeric, 2) as max_latency_ms
FROM predictions
UNION ALL
SELECT
    'GRU',
    ROUND(AVG(latency_gru_ms)::numeric, 2),
    ROUND(MIN(latency_gru_ms)::numeric, 2),
    ROUND(MAX(latency_gru_ms)::numeric, 2)
FROM predictions
UNION ALL
SELECT
    'LSTM',
    ROUND(AVG(latency_lstm_ms)::numeric, 2),
    ROUND(MIN(latency_lstm_ms)::numeric, 2),
    ROUND(MAX(latency_lstm_ms)::numeric, 2)
FROM predictions
UNION ALL
SELECT
    'Transformer',
    ROUND(AVG(latency_transformer_ms)::numeric, 2),
    ROUND(MIN(latency_transformer_ms)::numeric, 2),
    ROUND(MAX(latency_transformer_ms)::numeric, 2)
FROM predictions
UNION ALL
SELECT
    'Ensemble',
    ROUND(AVG(latency_ms)::numeric, 2),
    ROUND(MIN(latency_ms)::numeric, 2),
    ROUND(MAX(latency_ms)::numeric, 2)
FROM predictions
ORDER BY avg_latency_ms;"

echo ""
echo "🕒 Recent Predictions (Last 10)"
echo "-----------------------------------------"
PGPASSWORD=timeseries123 psql -h localhost -U timeseries -d predictions -c "
SELECT
    timestamp,
    ROUND((prediction_ann * 11.36 + 0.56)::numeric, 2) || '%' as ann,
    ROUND((prediction_gru * 11.36 + 0.56)::numeric, 2) || '%' as gru,
    ROUND((prediction_lstm * 11.36 + 0.56)::numeric, 2) || '%' as lstm,
    ROUND((prediction_ensemble * 11.36 + 0.56)::numeric, 2) || '%' as ensemble,
    ROUND(latency_ms::numeric, 2) || 'ms' as latency
FROM predictions
ORDER BY timestamp DESC
LIMIT 10;"

echo ""
echo "========================================="
echo "ℹ️  Interpretation:"
echo "  Positive % = Price increase predicted"
echo "  Negative % = Price decrease predicted"
echo "  Forecast horizon: 5 days"
echo "========================================="
