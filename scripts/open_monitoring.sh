#!/bin/bash
# Open all monitoring UIs

echo "🚀 Opening Monitoring Dashboards..."
echo ""
echo "Services:"
echo "  📊 Grafana:      http://localhost:3000 (admin/admin)"
echo "  🔍 Prometheus:   http://localhost:9090"
echo "  🚨 Alertmanager: http://localhost:9093"
echo "  🤖 API:          http://localhost:8000"
echo "  📈 MLflow:       http://localhost:5001"
echo ""

# Check if services are running
if ! docker ps | grep -q prometheus; then
    echo "❌ Services not running! Start with:"
    echo "   docker-compose -f docker-compose.monitoring.yml up -d"
    exit 1
fi

echo "✓ Services are running"
echo ""

# Open browsers
echo "Opening browsers..."
sleep 1

open http://localhost:3000      # Grafana
sleep 0.5
open http://localhost:9090      # Prometheus
sleep 0.5
open http://localhost:9093      # Alertmanager

echo ""
echo "✓ All monitoring UIs opened!"
echo ""
echo "Quick Prometheus Queries:"
echo "  model_drift_score{drift_type=\"dataset\"}"
echo "  prediction_drift_score"
echo "  {__name__=~\".*drift.*\"}"
