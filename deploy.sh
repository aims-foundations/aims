#!/bin/bash
set -e

# Set HuggingFace cache to writable location
export HF_HOME=/lfs/local/0/sttruong/.cache/huggingface

# Activate conda environment
source /lfs/skampere2/0/sttruong/miniconda3/etc/profile.d/conda.sh
conda activate aims

# Navigate to project
cd /lfs/skampere2/0/sttruong/aimslab/aims_textbook

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
