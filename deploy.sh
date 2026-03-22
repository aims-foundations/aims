#!/bin/bash
set -e

# Derive paths from script location (works on any machine)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DIR="$(dirname "$PROJECT_DIR")"

# Set HuggingFace cache to writable location
export HF_HOME=/lfs/local/0/sttruong/.cache/huggingface

# Activate conda environment
source "$USER_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate aims

# Navigate to project
cd "$PROJECT_DIR"

# Pull latest changes
# git pull

# Note: Each quarto render clears _book/, so we need to preserve outputs
# between renders by syncing to a temporary location

# Build PDF first (this populates _book/ with PDF)
echo "Building PDF..."
quarto render --to pdf --profile pdf

# Build HTML (this adds HTML to _book/ without clearing PDF)
echo "Building HTML..."
quarto render --to html --profile html --no-clean

# Deploy textbook to www/textbook/
echo "Deploying textbook to www/textbook/..."
rsync -av --delete --no-perms --no-owner --no-group _book/ /afs/cs/group/aimslab/www/textbook/

echo "Textbook deployed successfully!"
