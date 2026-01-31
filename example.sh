#!/bin/bash

# ==============================================================================
# This code need to be run inside the container.
# ==============================================================================

set -e

# The setup script created a subfolder 'TRELLIS_Workspace' to keep things tidy
PROJECT_ROOT="/root/t2"
ENV_NAME="venv"
ENV_DIR="$PROJECT_ROOT/$ENV_NAME"

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export ROCM_HOME=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1151
export FORCE_CUDA=1 # Forces HIP compilation even if detection is wonky

# Force AMD Serialization to prevent hangs (Strix Halo Specific)
export AMD_SERIALIZE_KERNEL=1 

# Activate Environment
if [ -f "$ENV_DIR/bin/activate" ]; then
    source "$ENV_DIR/bin/activate"
else
    echo "❌ Error: Virtual environment not found at $ENV_DIR"
    echo "   Did you run scripts/setup.sh?"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Starting TRELLIS generation..."

python example.py