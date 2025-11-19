#!/bin/bash
# Trigger GitHub Actions monitoring workflow from command line

set -e

echo "=========================================="
echo "Trigger Monitoring Workflow"
echo "=========================================="
echo ""

# Install GitHub CLI if not available
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) not found."
    echo ""
    echo "Install with:"
    echo "  brew install gh"
    echo ""
    echo "Or go to: https://github.com/In-mil/time_series_project/actions"
    echo "  → Comprehensive Production Monitoring"
    echo "  → Run workflow"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Not authenticated with GitHub."
    echo "Run: gh auth login"
    exit 1
fi

# Trigger the workflow
echo "Triggering workflow: Comprehensive Production Monitoring"
gh workflow run daily-monitoring.yml

echo ""
echo "✅ Workflow triggered!"
echo ""
echo "To view the run:"
echo "  gh run list --workflow=daily-monitoring.yml"
echo ""
echo "To watch it live:"
echo "  gh run watch"
echo ""
echo "Or open in browser:"
echo "  https://github.com/In-mil/time_series_project/actions"
echo ""
