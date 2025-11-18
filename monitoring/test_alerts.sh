#!/bin/bash
# Test alert firing by generating synthetic load and errors

set -e

API_URL="${1:-http://localhost:8000}"
PROMETHEUS_URL="${2:-http://localhost:9090}"
ALERTMANAGER_URL="${3:-http://localhost:9093}"

echo "=========================================="
echo "Alert Testing Script"
echo "=========================================="
echo "API URL: $API_URL"
echo "Prometheus: $PROMETHEUS_URL"
echo "Alertmanager: $ALERTMANAGER_URL"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if services are running
echo -e "\n${YELLOW}1. Checking services...${NC}"

if curl -s "$API_URL/" > /dev/null; then
    echo -e "${GREEN}✓ API is running${NC}"
else
    echo -e "${RED}✗ API is not reachable at $API_URL${NC}"
    exit 1
fi

if curl -s "$PROMETHEUS_URL/-/healthy" > /dev/null; then
    echo -e "${GREEN}✓ Prometheus is running${NC}"
else
    echo -e "${RED}✗ Prometheus is not reachable${NC}"
    exit 1
fi

if curl -s "$ALERTMANAGER_URL/-/healthy" > /dev/null; then
    echo -e "${GREEN}✓ Alertmanager is running${NC}"
else
    echo -e "${RED}✗ Alertmanager is not reachable${NC}"
    exit 1
fi

# Function to show active alerts
show_alerts() {
    echo -e "\n${YELLOW}Active Alerts:${NC}"
    curl -s "$PROMETHEUS_URL/api/v1/alerts" | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
alerts = data.get('data', {}).get('alerts', [])
if not alerts:
    print('  No active alerts')
else:
    for alert in alerts:
        name = alert.get('labels', {}).get('alertname', 'Unknown')
        state = alert.get('state', 'Unknown')
        severity = alert.get('labels', {}).get('severity', 'Unknown')
        print(f'  - {name} [{severity}]: {state}')
" || echo "  Failed to fetch alerts"
}

# Test 1: Trigger HighPredictionErrorRate by sending invalid data
echo -e "\n${YELLOW}2. Test: Triggering HighPredictionErrorRate alert...${NC}"
echo "   Sending 100 invalid prediction requests..."

for i in {1..100}; do
    curl -s -X POST "$API_URL/predict" \
        -H "Content-Type: application/json" \
        -d '{"sequence": []}' > /dev/null 2>&1 &
done
wait

echo -e "${GREEN}✓ Sent 100 invalid requests${NC}"
echo "   Wait 5 minutes for alert to fire (threshold: >5% error rate for 5min)"

# Test 2: Trigger HighPredictionLatency by generating load
echo -e "\n${YELLOW}3. Test: Triggering HighPredictionLatency alert...${NC}"
echo "   Sending 50 concurrent valid requests (may cause latency spike)..."

# Create a valid test sequence (20 timesteps, 68 features - adjust as needed)
TEST_SEQUENCE='{"sequence": [
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]
]}'

for i in {1..50}; do
    curl -s -X POST "$API_URL/predict" \
        -H "Content-Type: application/json" \
        -d "$TEST_SEQUENCE" > /dev/null 2>&1 &
done
wait

echo -e "${GREEN}✓ Sent 50 concurrent requests${NC}"

# Test 3: Send manual test alert
echo -e "\n${YELLOW}4. Test: Sending manual test alert to Alertmanager...${NC}"

curl -s -X POST "$ALERTMANAGER_URL/api/v1/alerts" \
    -H "Content-Type: application/json" \
    -d '[{
        "labels": {
            "alertname": "TestAlert",
            "severity": "warning",
            "service": "time-series-api"
        },
        "annotations": {
            "summary": "This is a test alert",
            "description": "Testing the alerting pipeline from test_alerts.sh script"
        }
    }]' > /dev/null

echo -e "${GREEN}✓ Test alert sent to Alertmanager${NC}"

# Show current alerts
show_alerts

# Display next steps
echo -e "\n${YELLOW}=========================================="
echo "Test Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Wait 5-10 minutes for alerts to fire"
echo "  2. Check Prometheus: $PROMETHEUS_URL/alerts"
echo "  3. Check Alertmanager: $ALERTMANAGER_URL"
echo "  4. Check Slack (if configured)"
echo ""
echo "To monitor alerts in real-time:"
echo "  watch -n 5 'curl -s $PROMETHEUS_URL/api/v1/alerts | python3 -m json.tool'"
echo ""
echo "To view Alertmanager logs:"
echo "  docker logs alertmanager --tail=50 -f"
echo ""
