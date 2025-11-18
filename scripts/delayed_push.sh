#!/bin/bash
echo "⏰ Waiting 30 minutes before pushing to GitHub..."
echo "Started at: $(date)"
sleep 1800
echo "🚀 Attempting git push at: $(date)"
git push --set-upstream origin main

if [ $? -eq 0 ]; then
    echo "✅ Push successful!"
else
    echo "❌ Push failed - GitHub might still be down"
fi
