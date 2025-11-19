#!/bin/bash
# Disable DVC tracking - work with local files only

echo "Disabling DVC for local development..."
echo ""

# Remove .dvc directory (keeps .dvcignore)
if [ -d ".dvc" ]; then
    echo "1. Moving .dvc to .dvc.backup"
    mv .dvc .dvc.backup
fi

# Clear any DVC cache
echo "2. Clearing DVC cache"
rm -rf .dvc/cache 2>/dev/null

# Add all model files to git (but not the actual large files - already gitignored)
echo "3. Ensuring files are git-ignored but locally present"

# Update .gitignore to ignore all large files
cat >> .gitignore << 'GITIGNORE'

# Large files - kept local only
/models/*.keras
/artifacts/drift_detection/reference_data.csv
/data/final_data/*.csv
GITIGNORE

echo ""
echo "✓ DVC disabled!"
echo "✓ All files are local - no cloud dependency"
echo ""
echo "You can now:"
echo "  1. Build Docker: docker-compose -f docker-compose.monitoring.yml build api"
echo "  2. Deploy without DVC"
echo ""
echo "To re-enable DVC later:"
echo "  mv .dvc.backup .dvc"
