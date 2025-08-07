#!/bin/bash
cd "$(dirname "$0")"
git add .
git commit -m "Auto-update: $(date)"
git pull origin main --rebase
git push origin main
echo "Repository successfully updated!"
