#!/usr/bin/env bash
set -euo pipefail

# Derive paths from script location (works on any machine)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DIR="$(dirname "$PROJECT_DIR")"
CONDA_ENV_NAME="${AIMS_CONDA_ENV:-aims}"
DEPLOY_TARGET="${AIMS_DEPLOY_TARGET:-/afs/cs/group/aimslab/www/textbook/}"
BUILD_PDF="${AIMS_BUILD_PDF:-1}"

# Set HuggingFace cache to writable location
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if ! conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1; then
      echo "Warning: conda environment '$CONDA_ENV_NAME' was not found; using the current shell environment." >&2
    fi
    return
  fi

  for conda_sh in \
    "$USER_DIR/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck source=/dev/null
      source "$conda_sh"
      if ! conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1; then
        echo "Warning: conda environment '$CONDA_ENV_NAME' was not found; using the current shell environment." >&2
      fi
      return
    fi
  done

  echo "Conda not found; using the current shell environment."
}

activate_conda

# Navigate to project
cd "$PROJECT_DIR"

for cmd in quarto rsync; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: $cmd is required but not installed." >&2
    exit 1
  fi
done

# Pull latest changes
# git pull

# Note: Each quarto render clears _book/, so we need to preserve outputs
# between renders by syncing to a temporary location

if [[ "$BUILD_PDF" == "1" ]]; then
  # Build PDF first so the deploy-profile HTML render can include the download.
  echo "Building PDF..."
  quarto render --to pdf
else
  echo "Skipping PDF build (AIMS_BUILD_PDF=$BUILD_PDF)."
  if [[ ! -f "_book/AI-Measurement-Science.pdf" ]]; then
    echo "Warning: no existing PDF found in _book/; deploy-profile HTML will omit the PDF download link." >&2
  fi
fi

# Build HTML (this adds HTML to _book/ without clearing PDF)
echo "Building HTML..."
quarto render --to html --profile deploy --no-clean

# Deploy textbook to www/textbook/
echo "Deploying textbook to $DEPLOY_TARGET..."
rsync -av --delete --no-perms --no-owner --no-group _book/ "$DEPLOY_TARGET"

echo "Textbook deployed successfully!"
