#!/bin/bash

# ==============================================================================
# This code need to be run inside the container.
# ==============================================================================

set -e

export TZ=Etc/GMT
export DEBIAN_FRONTEND=noninteractive

# Paths
PROJECT_ROOT="/root/t2"
BUILD_ROOT="$PROJECT_ROOT/build"
ENV_NAME="venv"
REPO_URL="https://github.com/Microsoft/TRELLIS.2.git"

# Locate Patches: SCRIPT_DIR is inside 'scripts', so we go up one level to find 'patches'
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PATCHES_DIR="$SCRIPT_DIR/../patches"

# Global Compiler Settings (Default to hipcc)
export CC=hipcc
export CXX=hipcc
export CUDACXX=/opt/rocm/bin/hipcc
export USE_ROCM=1
export CUDA=/opt/rocm
# CRITICAL: Set Arch for Strix Halo (gfx1151)
export PYTORCH_ROCM_ARCH="gfx1151"

echo "=== CHECKING SYSTEM TOOLS ==="
apt update
apt install -y wget git vim python3-full python3-pip build-essential libgl1 libglib2.0-0 libeigen3-dev libjpeg-dev libpng-dev curl

# --- HUGGING FACE AUTHENTICATION START ---
echo "⚡ Installing Hugging Face CLI..."
curl -LsSf https://hf.co/cli/install.sh | bash

HF_TOKEN_FILE="$PROJECT_ROOT/hf.txt"

if [ -f "$HF_TOKEN_FILE" ]; then
    echo "⚡ Found hf.txt. Authenticating automatically..."
    # Read token and strip any whitespace/newlines
    HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    
    # Login using the token (Standard flag for the 'hf' binary)
    /root/.local/bin/hf auth login --token "$HF_TOKEN"
else
    echo "⚠️  hf.txt not found at $HF_TOKEN_FILE"
    echo "   Please paste your token manually below:"
    /root/.local/bin/hf auth login
fi
# --- HUGGING FACE AUTHENTICATION END ---

# Fix Eigen3 include path
if [ -d "/usr/include/eigen3/Eigen" ] && [ ! -d "/usr/include/Eigen" ]; then
    ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
fi

echo "=== CHECKING ROCM STATUS ==="
if dpkg -s amdgpu-install >/dev/null 2>&1; then
    echo "✅ ROCm installer (amdgpu-install) is already installed."
else
    echo "⚡ Installing ROCm installer..."
    WGET_URL="https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb"
    DEB_FILE="/tmp/amdgpu-install.deb"
    wget -O "$DEB_FILE" "$WGET_URL"
    apt install -y "$DEB_FILE"
    rm -f "$DEB_FILE"
fi

echo "⚡ Verifying ROCm libraries..."
amdgpu-install --usecase=rocm --no-dkms -y

echo "=== CHECKING PYTHON ENVIRONMENT ==="
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment ($ENV_NAME)..."
    python3 -m venv "$ENV_NAME"
else
    echo "✅ Virtual environment exists."
fi

source "$ENV_NAME/bin/activate"

pip install --upgrade pip setuptools wheel
pip install jupyter addict matplotlib plyfile open3d easydict trimesh zstandard
pip install opencv-python-headless imageio rembg onnxruntime-gpu kornia timm transformers
pip install utils3d --no-deps
pip check

# --- PATCHING TRANSFORMERS (Keeping SED here as it patches the venv lib, not the repo) ---
echo "⚡ Patching transformers library for BiRefNet & DINOv3..."
TRANSFORMERS_LIB="$ENV_NAME/lib/python3.12/site-packages/transformers/modeling_utils.py"

# Fix A: mark_tied_weights_as_initialized
sed -i 's/self.all_tied_weights_keys.keys()/set(getattr(self, "all_tied_weights_keys", None) or getattr(self, "_tied_weights_keys", None) or [])/g' "$TRANSFORMERS_LIB"
# Fix B: tie_weights
sed -i 's/tied_keys = self.all_tied_weights_keys/tied_keys = getattr(self, "all_tied_weights_keys", None) or getattr(self, "_tied_weights_keys", None) or []/g' "$TRANSFORMERS_LIB"
# Fix C: DINOv3 List vs Dict crash
sed -i 's/tied_keys = list(tied_keys.items())/tied_keys = list(tied_keys.items()) if isinstance(tied_keys, dict) else tied_keys/g' "$TRANSFORMERS_LIB"

if [ -d "/root/.cache/huggingface/modules" ]; then
    find /root/.cache/huggingface/modules -name "birefnet.py" -exec sed -i 's/torch.linspace(0, drop_path_rate, sum(depths))/torch.linspace(0, drop_path_rate, sum(depths), device="cpu")/g' {} +
fi

# Create Workspace
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"
echo "Building in $BUILD_ROOT..."

echo "=== REBUILDING TORCHVISION (Source) ==="
if ! python3 -c "import torchvision; print(torchvision.ops.nms)" >/dev/null 2>&1; then
    echo "⚡ 'nms' operator missing. Building TorchVision from source..."
    pip uninstall -y torchvision
    
    if [ ! -d "vision" ]; then
        git clone --recursive https://github.com/pytorch/vision.git
    fi
    cd vision
    git fetch --tags
    git checkout v0.25.0 2>/dev/null || git checkout main
    
    cat > clang_cleaner.sh << 'EOF'
#!/bin/bash
ARGS=()
for arg in "$@"; do
    if [ "$arg" != "-std=c++17" ]; then
        ARGS+=("$arg")
    fi
done
exec /opt/rocm/llvm/bin/clang "${ARGS[@]}"
EOF
    chmod +x clang_cleaner.sh

    export CC=$(pwd)/clang_cleaner.sh
    export CXX=/opt/rocm/bin/hipcc
    export FORCE_CUDA=1
    
    python3 setup.py clean
    pip install . --no-build-isolation --no-cache-dir
    
    rm clang_cleaner.sh
    export CC=hipcc
    export CXX=hipcc
    cd ..
else
    echo "✅ TorchVision 'nms' operator is present."
fi
pip check

echo "=== CHECKING PYTORCH (ROCm 7.2 Manual Install) ==="
if ! python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "rocm7.2"; then
    echo "⚡ Installing PyTorch 2.9.1 (ROCm 7.2) for Strix Halo..."
    
    # Clean, if needed
    #pip uninstall -y torch torchvision torchaudio
    
    pip install --no-cache-dir \
       https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
       https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
       https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
else
    echo "✅ PyTorch (ROCm 7.2) is already installed."
fi
pip check

echo "=== CHECKING TRELLIS ==="
if [ ! -d "TRELLIS.2/o-voxel" ]; then
    echo "⚡ Cloning TRELLIS.2..."
    rm -rf TRELLIS.2
    git clone "$REPO_URL" TRELLIS.2
    cd TRELLIS.2
    git submodule update --init --recursive
    
    if [ -f "requirements.txt" ]; then 
        echo "⚡ Installing requirements (safely)..."
        grep -vE "torch|torchvision|torchaudio" requirements.txt > requirements_safe.txt
        pip install -r requirements_safe.txt
        rm requirements_safe.txt
    fi
    cd ..
else
    echo "✅ TRELLIS.2 repo exists."
fi
pip check

# ==============================================================================
# ⚡ UNIVERSAL PATCHING STEP
# This copies everything from 'patches/' directly into 'BUILD_ROOT'
# It will overwrite TRELLIS files with the structure you defined.
# ==============================================================================
if [ -d "$PATCHES_DIR" ]; then
    echo "⚡ Applying custom patches from $PATCHES_DIR..."
    # Copy recursively (-r), verbose (-v), forcing overwrite
    cp -rv "$PATCHES_DIR/." .
else
    echo "⚠️  WARNING: Patches directory not found at $PATCHES_DIR"
fi
# ==============================================================================

echo "=== CHECKING DEPENDENCIES ==="
if [ ! -d "CuMesh" ]; then
    git clone https://github.com/Lamothe/CuMesh_rocm.git CuMesh
fi

echo "⚡ Building CuMesh..."
cd CuMesh
git submodule update --init --recursive
if pip install . --no-build-isolation --no-cache-dir; then
    echo "✅ CuMesh build successful."
else
    echo "❌ CuMesh build failed."
    exit 1
fi
cd ..

if [ ! -d "FlexGEMM" ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM
fi

echo "=== PATCHING FLEXGEMM ==="
cd FlexGEMM
find . -type f -name "*.py" -exec sed -i 's/input_precision="tf32"/input_precision="ieee"/g' {} +
find . -type f -name "*.py" -exec sed -i "s/input_precision='tf32'/input_precision='ieee'/g" {} +
if pip install . --no-build-isolation --no-cache-dir; then
    echo "✅ FlexGEMM build successful."
else
    echo "❌ FlexGEMM build failed."
    exit 1
fi
cd ..

echo "=== BUILDING O-VOXEL ==="
# NOTE: The patch for postprocess.py was already applied by the Universal Patching step above!

cd TRELLIS.2/o-voxel
rm -rf build/ dist/ *.egg-info

echo "⚡ Patching O-Voxel C++ sources for ROCm..."
# 1. Reset file to be safe
git checkout src/convert/flexible_dual_grid.cpp 2>/dev/null || echo "⚠️ Could not reset file"

# 2. Apply C++ Patches (Suffixes, Types, Narrowing)
sed -i 's/1e-6d/1e-6/g' src/convert/flexible_dual_grid.cpp
sed -i 's/0\.0d/0.0/g' src/convert/flexible_dual_grid.cpp
sed -i 's/\bfloat3\b/local_float3/g' src/convert/flexible_dual_grid.cpp
sed -i 's/\bint3\b/local_int3/g' src/convert/flexible_dual_grid.cpp
sed -i 's/\bint4\b/local_int4/g' src/convert/flexible_dual_grid.cpp
sed -i 's/neigh_indices\[/(int)neigh_indices\[/g' src/convert/flexible_dual_grid.cpp
sed -i 's/size_t (int)neigh_indices/size_t neigh_indices/g' src/convert/flexible_dual_grid.cpp
sed -i 's/torch::zeros({N, C}/torch::zeros({(long)N, (long)C}/g' src/io/filter_neighbor.cpp
sed -i 's/torch::zeros({N_leaf, C}/torch::zeros({(long)N_leaf, (long)C}/g' src/io/filter_parent.cpp
sed -i 's/{svo.size()}/{(long)svo.size()}/g' src/io/svo.cpp
sed -i 's/{codes.size()}/{(long)codes.size()}/g' src/io/svo.cpp

pip install . --no-build-isolation --no-cache-dir --no-deps
cd ../..

echo "=== FINAL CONFIGURATION ==="
# 4. Enable Experimental Kernels for Strix Halo
if ! grep -q "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" ~/.bashrc; then
    echo 'export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1' >> ~/.bashrc
fi
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

echo "=== SETUP COMPLETE ==="