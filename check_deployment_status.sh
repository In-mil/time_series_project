#!/bin/bash
# Check deployment status in a loop

echo "Monitoring Cloud Run deployment..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
  clear
  echo "==========================================="
  echo "Cloud Run Deployment Status"
  echo "Time: $(date '+%H:%M:%S')"
  echo "==========================================="
  echo ""

  # Get latest revision
  gcloud run revisions list \
    --service=time-series-api \
    --region=europe-west3 \
    --limit=1 \
    --format="table(metadata.name,status.conditions[0].status,metadata.creationTimestamp)"

  echo ""

  # Check if service is ready
  STATUS=$(gcloud run services describe time-series-api \
    --region=europe-west3 \
    --format="value(status.conditions[0].status)" 2>/dev/null)

  if [ "$STATUS" = "True" ]; then
    echo "✓ Service is READY!"
    URL=$(gcloud run services describe time-series-api \
      --region=europe-west3 \
      --format="value(status.url)")
    echo "  URL: $URL"
    echo ""
    echo "Run tests with:"
    echo "  python3 test_api.py $URL"
    break
  else
    echo "⏳ Service not ready yet..."
    echo "   Checking again in 30 seconds..."
  fi

  sleep 30
done
