#!/bin/bash

# Model Proxy Server Startup Script
echo "Starting Model Proxy Server..."

# Set environment variables
export TRANSFORMERS_CACHE=/mnt/models/huggingface_cache
export HF_HOME=/mnt/models/huggingface_cache
export CUDA_VISIBLE_DEVICES=""

# Create necessary directories
mkdir -p /mnt/models/huggingface_cache
mkdir -p /mnt/models/user_models

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if TPU is available
echo "Checking TPU availability..."
python3 -c "import jax; print('TPU devices:', jax.devices())" || echo "Warning: TPU not detected"

# Kill any existing servers
pkill -f "python.*model_proxy_server.py" || true

# Start the server
echo "Starting server on port 8000..."
python3 model_proxy_server.py > proxy_server.log 2>&1 &

echo "Server started!"
echo "Web Interface: http://localhost:8000"
echo "Log file: proxy_server.log"
