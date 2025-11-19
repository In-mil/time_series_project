#!/bin/bash
# Traffic Generator for Grafana Demo

echo "🚀 Generating prediction traffic for Grafana demo..."
echo ""

# Check if API is running
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "❌ API is not running! Start it first:"
    echo "   docker-compose -f docker-compose.monitoring.yml up -d"
    exit 1
fi

echo "✅ API is running"
echo ""

# Function to make predictions
make_predictions() {
    local count=$1
    local delay=$2

    echo "Making $count predictions (1 every ${delay}s)..."

    for i in $(seq 1 $count); do
        result=$(curl -s -X POST http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d @sample_request.json)

        if [ $? -eq 0 ]; then
            ensemble=$(echo $result | python -c "import sys, json; print(json.load(sys.stdin)['prediction'])" 2>/dev/null)
            if [ -n "$ensemble" ]; then
                printf "  ✓ Prediction %3d/%d - Ensemble: %.4f\n" $i $count $ensemble
            else
                printf "  ✓ Prediction %3d/%d\n" $i $count
            fi
        else
            printf "  ✗ Failed %d/%d\n" $i $count
        fi

        [ $i -lt $count ] && sleep $delay
    done
}

# Menu
echo "Select traffic pattern:"
echo ""
echo "  1) Slow & Steady  - 20 predictions over 2 minutes  (~0.17/s)"
echo "  2) Medium         - 50 predictions over 1 minute   (~0.83/s)"
echo "  3) Fast Burst     - 100 predictions over 30 seconds (~3.3/s)"
echo "  4) Continuous     - Keep running until Ctrl+C"
echo "  5) Single Batch   - 10 predictions immediately"
echo ""
read -p "Choose (1-5): " choice

case $choice in
    1)
        echo ""
        make_predictions 20 6
        ;;
    2)
        echo ""
        make_predictions 50 1.2
        ;;
    3)
        echo ""
        make_predictions 100 0.3
        ;;
    4)
        echo ""
        echo "Continuous mode - Press Ctrl+C to stop"
        echo ""
        count=0
        while true; do
            count=$((count + 1))
            result=$(curl -s -X POST http://localhost:8000/predict \
                -H "Content-Type: application/json" \
                -d @sample_request.json)
            ensemble=$(echo $result | python -c "import sys, json; print(json.load(sys.stdin)['prediction'])" 2>/dev/null)
            printf "\r  ✓ Prediction %d - Ensemble: %.4f" $count $ensemble
            sleep 1
        done
        ;;
    5)
        echo ""
        make_predictions 10 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo ""
echo "✅ Done! Check Grafana for updated metrics:"
echo "   http://localhost:3000"
echo ""
