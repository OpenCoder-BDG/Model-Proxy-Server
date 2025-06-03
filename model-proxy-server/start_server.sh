#!/bin/bash

# Production Model Proxy Server Startup Script
# Enhanced with better logging, error handling, and monitoring

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SERVER_PORT=${PORT:-8000}
LOG_DIR="/mnt/models/logs"
CACHE_DIR="/mnt/models/huggingface_cache"
USER_MODELS_DIR="/mnt/models/user_models"
VENV_DIR="venv"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
  ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗          
  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║          
  ██╔████╔██║██║   ██║██║  ██║█████╗  ██║          
  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║          
  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗     
  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝     
                                                   
  ██████╗ ██████╗  ██████╗ ██╗  ██╗██╗   ██╗       
  ██╔══██╗██╔══██╗██╔═══██╗╚██╗██╔╝╚██╗ ██╔╝       
  ██████╔╝██████╔╝██║   ██║ ╚███╔╝  ╚████╔╝        
  ██╔═══╝ ██╔══██╗██║   ██║ ██╔██╗   ╚██╔╝         
  ██║     ██║  ██║╚██████╔╝██╔╝ ██╗   ██║          
  ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝          
                                                   
  Production-Grade HuggingFace Model Proxy Server
EOF
    echo -e "${NC}"
}

# Check dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check Python version
    if ! python3 --version | grep -q "Python 3"; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    log_success "System dependencies check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment variables..."
    
    # Core environment variables
    export TRANSFORMERS_CACHE="$CACHE_DIR"
    export HF_HOME="$CACHE_DIR"
    export HF_HUB_CACHE="$CACHE_DIR"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    
    # GPU settings (disable by default, can be overridden)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}
    
    # Production settings
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    log_success "Environment variables configured"
}

# Create directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$CACHE_DIR"
    mkdir -p "$USER_MODELS_DIR"
    mkdir -p "/tmp/model_proxy"
    
    # Set permissions
    chmod 755 "$LOG_DIR" "$CACHE_DIR" "$USER_MODELS_DIR" || log_warning "Could not set directory permissions"
    
    log_success "Directories created successfully"
}

# Setup virtual environment
setup_venv() {
    if [ -d "$VENV_DIR" ]; then
        log_info "Activating existing virtual environment..."
        source "$VENV_DIR/bin/activate"
        log_success "Virtual environment activated"
    else
        log_warning "No virtual environment found at $VENV_DIR"
        log_info "Creating new virtual environment..."
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        log_success "Virtual environment created and activated"
        
        # Upgrade pip and install requirements
        log_info "Installing requirements..."
        pip install --upgrade pip
        pip install -r requirements.txt
        log_success "Requirements installed"
    fi
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check Python imports
    python3 -c "
import sys
try:
    import torch
    import transformers
    import fastapi
    import jax
    print('✓ All core dependencies imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" || {
        log_error "Health check failed - missing dependencies"
        exit 1
    }
    
    # Check hardware
    log_info "Hardware detection:"
    python3 -c "
import torch
import jax
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
print(f'JAX Devices: {len(jax.devices())}')
for device in jax.devices():
    print(f'  {device}')
"
    
    log_success "Health checks completed"
}

# Kill existing processes
cleanup_existing() {
    log_info "Cleaning up existing processes..."
    
    # Kill existing server processes
    pkill -f "model_proxy_server.py" || log_info "No existing server processes found"
    pkill -f "uvicorn.*model_proxy_server" || log_info "No existing uvicorn processes found"
    
    # Wait a moment for processes to terminate
    sleep 2
    
    log_success "Cleanup completed"
}

# Start server
start_server() {
    log_info "Starting Model Proxy Server..."
    
    # Create log file with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/proxy_server_$TIMESTAMP.log"
    LATEST_LOG="$LOG_DIR/proxy_server_latest.log"
    
    # Start server with proper logging
    nohup python3 model_proxy_server.py > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    
    # Create symlink to latest log
    ln -sf "$LOG_FILE" "$LATEST_LOG"
    
    # Save PID for management
    echo $SERVER_PID > /tmp/model_proxy_server.pid
    
    log_success "Server started with PID: $SERVER_PID"
    log_info "Log file: $LOG_FILE"
    log_info "Latest log: $LATEST_LOG"
}

# Verify server is running
verify_server() {
    log_info "Verifying server startup..."
    
    # Wait for server to start
    sleep 3
    
    # Check if process is still running
    if kill -0 $(cat /tmp/model_proxy_server.pid 2>/dev/null) 2>/dev/null; then
        log_success "Server process is running"
    else
        log_error "Server process failed to start"
        log_info "Check the log file for details:"
        tail -20 "$LOG_DIR/proxy_server_latest.log"
        exit 1
    fi
    
    # Test HTTP endpoint
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            log_success "Server is responding to HTTP requests"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Server failed to respond after $max_attempts attempts"
            exit 1
        fi
        
        log_info "Waiting for server to respond... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
}

# Display server info
show_server_info() {
    local external_ip
    external_ip=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")
    
    echo ""
    log_success "🚀 Model Proxy Server is running!"
    echo ""
    echo -e "${GREEN}📍 Access URLs:${NC}"
    echo -e "   Local:    http://localhost:$SERVER_PORT"
    echo -e "   Network:  http://$external_ip:$SERVER_PORT"
    echo ""
    echo -e "${GREEN}📊 Management:${NC}"
    echo -e "   Status:   http://localhost:$SERVER_PORT/status"
    echo -e "   Health:   http://localhost:$SERVER_PORT/health"
    echo -e "   Logs:     tail -f $LOG_DIR/proxy_server_latest.log"
    echo ""
    echo -e "${GREEN}🛠️  Control:${NC}"
    echo -e "   Stop:     kill \$(cat /tmp/model_proxy_server.pid)"
    echo -e "   Restart:  $0"
    echo ""
}

# Main execution
main() {
    banner
    
    log_info "Starting production Model Proxy Server deployment..."
    
    check_dependencies
    setup_environment
    create_directories
    setup_venv
    run_health_checks
    cleanup_existing
    start_server
    verify_server
    show_server_info
    
    log_success "Deployment completed successfully!"
}

# Handle script arguments
case "$1" in
    --dev|dev)
        export NODE_ENV=development
        log_info "Running in development mode"
        ;;
    --prod|prod)
        export NODE_ENV=production
        log_info "Running in production mode"
        ;;
    --help|help|-h)
        echo "Usage: $0 [--dev|--prod|--help]"
        echo "  --dev   : Development mode"
        echo "  --prod  : Production mode (default)"
        echo "  --help  : Show this help"
        exit 0
        ;;
    *)
        export NODE_ENV=production
        ;;
esac

# Run main function
main "$@"
