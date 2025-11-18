#!/bin/bash
# Get Cloud Run service URL

echo "Fetching Cloud Run service URL..."
URL=$(gcloud run services describe time-series-api \
  --region=europe-west3 \
  --format="value(status.url)")

if [ -z "$URL" ]; then
  echo "ERROR: Service not found or not deployed yet"
  exit 1
fi

echo ""
echo "=========================================="
echo "Service URL: $URL"
echo "=========================================="
echo ""
echo "Test the service:"
echo "  curl $URL/"
echo ""
echo "Run test suite:"
echo "  python3 scripts/test_deployed_api.py $URL"
echo ""
