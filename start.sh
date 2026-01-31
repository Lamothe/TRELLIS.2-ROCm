#!/bin/bash

# ==============================================================================
# This code runs on the host, it creates the container.
# ==============================================================================

CTX_DIR="$PWD/build/container"
CONTAINER_NAME="rocm72"
IMAGE="docker.io/library/ubuntu:24.04"

# Function to enforce local storage
function podlocal() {
    mkdir -p "$CTX_DIR/data" "$CTX_DIR/run"
    podman --root "$CTX_DIR/data" --runroot "$CTX_DIR/run" "$@"
}

# Create container if missing
if ! podlocal ps -a --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
    echo "Creating new isolated environment in $CTX_DIR..."
    
    podlocal run -d \
        --name "$CONTAINER_NAME" \
        --device /dev/kfd \
        --device /dev/dri \
        --security-opt=label=disable \
        -v "$PWD:/root/t2" \
        -w /root/t2 \
        "$IMAGE" \
        sleep infinity
fi

# 4. Run Setup AND THEN Enter Shell
echo "Entering $CONTAINER_NAME..."

podlocal exec -it "$CONTAINER_NAME" "/bin/bash"
