#!/usr/bin/env python3
"""
Production-Grade Model Proxy Server
High-performance HuggingFace model deployment platform with OpenAI API compatibility
"""

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

import requests
import torch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import jax
import jax.numpy as jnp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/mnt/models/proxy_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Production-grade global state with thread safety
USER_DEPLOYMENTS: Dict[str, Dict] = {}
ACTIVE_MODELS: Dict[str, Any] = {}
WEBSOCKET_CONNECTIONS: Dict[str, WebSocket] = {}
_deployment_lock = threading.Lock()


def is_model_compatible(model_info) -> tuple[bool, str]:
    """Check if a model is compatible with our setup"""
    
    # Get model tags
    tags = getattr(model_info, 'tags', [])
    library_name = getattr(model_info, 'library_name', '')
    
    # Filter out incompatible models
    incompatible_tags = ['mlx', 'gguf', 'onnx', 'openvino', 'tensorrt']
    incompatible_libraries = ['mlx', 'gguf', 'onnx']
    
    for tag in tags:
        if any(incomp in tag.lower() for incomp in incompatible_tags):
            return False, f"Incompatible format: {tag}"
    
    if library_name.lower() in incompatible_libraries:
        return False, f"Incompatible library: {library_name}"
    
    # Check for quantized models that might not work
    if any('bit' in tag for tag in tags) and 'mlx' in tags:
        return False, "MLX quantized model not supported"
    
    return True, "Compatible"


app = FastAPI(title="Model Proxy Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic Models for Production
class ModelSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100, description="Search query for models")
    limit: int = Field(default=12, ge=1, le=50, description="Maximum number of results")
    filter_type: Optional[str] = Field(default="text-generation", description="Model type filter")

class ModelDeployRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=200, description="HuggingFace model name")
    backend: str = Field(default="transformers", regex="^(transformers|jax)$", description="Model backend")
    api_key_enabled: bool = Field(default=True, description="Enable API key authentication")
    user_id: Optional[str] = Field(default=None, description="Optional user ID")
    max_memory_gb: Optional[int] = Field(default=8, ge=1, le=32, description="Maximum memory allocation")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier")
    messages: List[Dict[str, str]] = Field(..., min_items=1, description="Chat messages")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(default=False, description="Stream response")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")

class DeploymentStatus(BaseModel):
    user_id: str
    model_name: str
    status: str = Field(..., regex="^(deploying|ready|error|stopped)$")
    progress: int = Field(default=0, ge=0, le=100)
    message: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    created_at: datetime
    ready_at: Optional[datetime] = None

# Enhanced Utility Functions
def generate_api_key() -> str:
    """Generate a secure API key with proper prefix"""
    return f"mp-{secrets.token_urlsafe(32)}"

def generate_user_id() -> str:
    """Generate a unique user ID with timestamp"""
    timestamp = int(time.time())
    return f"user-{timestamp}-{str(uuid.uuid4())[:8]}"

def get_external_ip() -> str:
    """Get external IP with multiple fallback methods"""
    methods = [
        # Google Cloud metadata
        lambda: requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
            headers={"Metadata-Flavor": "Google"}, timeout=3
        ).text.strip(),
        # AWS metadata
        lambda: requests.get("http://169.254.169.254/latest/meta-data/public-ipv4", timeout=3).text.strip(),
        # External services
        lambda: requests.get("https://api.ipify.org", timeout=3).text.strip(),
        lambda: requests.get("https://icanhazip.com", timeout=3).text.strip(),
    ]
    
    for method in methods:
        try:
            ip = method()
            if ip and ip != "":
                logger.info(f"External IP detected: {ip}")
                return ip
        except Exception as e:
            logger.debug(f"IP detection method failed: {e}")
            continue
    
    # Final fallback
    logger.warning("Could not detect external IP, using localhost")
    return "localhost"

async def broadcast_status_update(user_id: str, status_data: dict):
    """Broadcast status updates via WebSocket"""
    if user_id in WEBSOCKET_CONNECTIONS:
        try:
            websocket = WEBSOCKET_CONNECTIONS[user_id]
            await websocket.send_json({
                "type": "deployment_status",
                "user_id": user_id,
                "data": status_data
            })
        except Exception as e:
            logger.error(f"Failed to send WebSocket update: {e}")
            # Remove dead connection
            WEBSOCKET_CONNECTIONS.pop(user_id, None)

def validate_model_compatibility(model_name: str) -> tuple[bool, str]:
    """Enhanced model compatibility checking"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model_info = api.model_info(model_name)
        
        # Check model tags
        tags = getattr(model_info, 'tags', [])
        library_name = getattr(model_info, 'library_name', '')
        
        # Incompatible formats
        incompatible_patterns = [
            'mlx', 'gguf', 'onnx', 'openvino', 'tensorrt', 
            'tflite', 'coreml', 'paddle'
        ]
        
        for tag in tags:
            for pattern in incompatible_patterns:
                if pattern in tag.lower():
                    return False, f"Incompatible format: {tag}"
        
        if library_name.lower() in incompatible_patterns:
            return False, f"Incompatible library: {library_name}"
        
        # Check model size (basic heuristic)
        if hasattr(model_info, 'safetensors') and model_info.safetensors:
            total_size = sum(
                file_info.get('size', 0) 
                for file_info in model_info.safetensors.get('parameters', {}).values()
            )
            # Warn if model is very large (>16GB)
            if total_size > 16 * 1024**3:
                return False, f"Model too large: {total_size / (1024**3):.1f}GB"
        
        return True, "Compatible"
        
    except Exception as e:
        logger.error(f"Error validating model {model_name}: {e}")
        return False, f"Validation error: {str(e)}"

async def search_huggingface_models(query: str, limit: int = 12, filter_type: str = "text-generation") -> List[Dict]:
    """Enhanced HuggingFace model search with filtering and validation"""
    try:
        url = "https://huggingface.co/api/models"
        params = {
            "search": query,
            "limit": min(limit * 2, 50),  # Get more to filter
            "filter": filter_type,
            "sort": "downloads",
            "direction": -1
        }
        
        logger.info(f"Searching for models: {query}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        models = response.json()
        formatted_models = []
        
        for model in models:
            model_id = model.get("id", "")
            if not model_id:
                continue
                
            # Validate model compatibility
            is_compatible, reason = validate_model_compatibility(model_id)
            
            # Get model size estimate
            size_info = "Unknown size"
            try:
                if 'safetensors' in model.get('tags', []):
                    size_info = "Optimized (safetensors)"
                elif any(tag in model.get('tags', []) for tag in ['pytorch', 'tf']):
                    size_info = "Standard format"
            except:
                pass
            
            formatted_model = {
                "id": model_id,
                "name": model_id,
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "tags": model.get("tags", [])[:5],  # Limit tags
                "description": (model.get("description", "") or "No description available")[:180] + "...",
                "library_name": model.get("library_name", "transformers"),
                "pipeline_tag": model.get("pipeline_tag", "text-generation"),
                "size_info": size_info,
                "compatible": is_compatible,
                "compatibility_reason": reason,
                "model_index": model.get("modelIndex", {}),
                "created_at": model.get("createdAt", ""),
                "updated_at": model.get("lastModified", "")
            }
            
            # Only include compatible models in main results
            if is_compatible:
                formatted_models.append(formatted_model)
                
            if len(formatted_models) >= limit:
                break
        
        logger.info(f"Found {len(formatted_models)} compatible models")
        return formatted_models
        
    except Exception as e:
        logger.error(f"Error searching HuggingFace models: {e}")
        # Enhanced fallback with popular, tested models
        return [
            {
                "id": "microsoft/DialoGPT-medium", "name": "microsoft/DialoGPT-medium",
                "downloads": 2500000, "likes": 350, "tags": ["conversational", "pytorch"],
                "description": "A large-scale pretrained dialogue response generation model optimized for conversational AI applications",
                "library_name": "transformers", "pipeline_tag": "conversational", "size_info": "~1.2GB",
                "compatible": True, "compatibility_reason": "Compatible"
            },
            {
                "id": "gpt2", "name": "gpt2",
                "downloads": 5000000, "likes": 1200, "tags": ["text-generation", "pytorch"],
                "description": "GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion",
                "library_name": "transformers", "pipeline_tag": "text-generation", "size_info": "~500MB",
                "compatible": True, "compatibility_reason": "Compatible"
            },
            {
                "id": "google/flan-t5-base", "name": "google/flan-t5-base",
                "downloads": 1800000, "likes": 280, "tags": ["text2text-generation", "pytorch"],
                "description": "FLAN-T5 base model fine-tuned on a collection of datasets phrased as instructions",
                "library_name": "transformers", "pipeline_tag": "text2text-generation", "size_info": "~900MB",
                "compatible": True, "compatibility_reason": "Compatible"
            },
            {
                "id": "microsoft/DialoGPT-small", "name": "microsoft/DialoGPT-small",
                "downloads": 1200000, "likes": 180, "tags": ["conversational", "pytorch"],
                "description": "A smaller, faster version of DialoGPT optimized for resource-constrained environments",
                "library_name": "transformers", "pipeline_tag": "conversational", "size_info": "~350MB",
                "compatible": True, "compatibility_reason": "Compatible"
            },
            {
                "id": "distilgpt2", "name": "distilgpt2",
                "downloads": 800000, "likes": 150, "tags": ["text-generation", "pytorch"],
                "description": "Distilled version of GPT-2: smaller, faster, yet preserving 95% of GPT-2 performance",
                "library_name": "transformers", "pipeline_tag": "text-generation", "size_info": "~300MB",
                "compatible": True, "compatibility_reason": "Compatible"
            }
        ]

async def load_model_async(user_id: str, model_name: str, backend: str = "transformers", max_memory_gb: int = 8) -> Dict:
    """Enhanced model loading with progress tracking and error handling"""
    try:
        logger.info(f"Loading model {model_name} for user {user_id} with {backend} backend")
        
        # Update status to starting
        await update_deployment_status(user_id, "deploying", 10, "Initializing model loading...")

        # Create model directory
        model_dir = f"/mnt/models/user_models/{model_name.replace('/', '_')}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs("/mnt/models/huggingface_cache", exist_ok=True)

        if backend == "transformers":
            # Get model info and validate
            await update_deployment_status(user_id, "deploying", 20, "Fetching model information...")
            
            from huggingface_hub import HfApi
            api = HfApi()
            model_info = api.model_info(model_name)
            pipeline_tag = getattr(model_info, 'pipeline_tag', 'text-generation')
            
            logger.info(f"Model {model_name} has pipeline tag: {pipeline_tag}")
            
            # Load tokenizer first
            await update_deployment_status(user_id, "deploying", 40, "Loading tokenizer...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="/mnt/models/huggingface_cache",
                trust_remote_code=True,
                use_fast=True
            )

            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Determine optimal torch dtype and device configuration
            device_config = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            await update_deployment_status(user_id, "deploying", 60, "Loading model weights...")

            # Load appropriate model based on pipeline tag
            if pipeline_tag == "text2text-generation":
                from transformers import AutoModelForSeq2SeqLM
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir="/mnt/models/huggingface_cache",
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device_config,
                    low_cpu_mem_usage=True,
                    max_memory={0: f"{max_memory_gb}GB"} if torch.cuda.is_available() else None
                )
                task = "text2text-generation"
            else:
                # Default to causal LM for text-generation
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir="/mnt/models/huggingface_cache",
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device_config,
                    low_cpu_mem_usage=True,
                    max_memory={0: f"{max_memory_gb}GB"} if torch.cuda.is_available() else None
                )
                task = "text-generation"

            await update_deployment_status(user_id, "deploying", 80, "Creating inference pipeline...")

            # Create optimized pipeline
            pipe = pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                device_map=device_config,
                torch_dtype=torch_dtype,
                return_full_text=False if task == "text-generation" else True
            )

            await update_deployment_status(user_id, "deploying", 95, "Finalizing deployment...")

            model_data = {
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": pipe,
                "backend": backend,
                "task": task,
                "model_name": model_name,
                "device": str(model.device) if hasattr(model, 'device') else 'cpu',
                "dtype": str(torch_dtype),
                "memory_usage": f"{max_memory_gb}GB limit",
                "loaded_at": datetime.now().isoformat(),
                "status": "ready"
            }
            
            logger.info(f"Model {model_name} loaded successfully for user {user_id}")
            return model_data

        elif backend == "jax":
            # JAX implementation placeholder
            await update_deployment_status(user_id, "deploying", 50, "JAX backend not fully implemented, falling back to transformers...")
            return await load_model_async(user_id, model_name, "transformers", max_memory_gb)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        logger.error(f"Error loading model for user {user_id}: {e}")
        await update_deployment_status(user_id, "error", 0, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

async def update_deployment_status(user_id: str, status: str, progress: int, message: str):
    """Update deployment status and broadcast via WebSocket"""
    with _deployment_lock:
        if user_id in USER_DEPLOYMENTS:
            USER_DEPLOYMENTS[user_id].update({
                "status": status,
                "progress": progress,
                "message": message,
                "updated_at": datetime.now().isoformat()
            })
            
            if status == "ready":
                USER_DEPLOYMENTS[user_id]["ready_at"] = datetime.now().isoformat()
    
    # Broadcast update
    status_data = {
        "status": status,
        "progress": progress,
        "message": message,
        "user_id": user_id
    }
    
    await broadcast_status_update(user_id, status_data)

# Enhanced API Routes with WebSocket Support
@app.get("/")
async def redirect_to_frontend():
    """Redirect to main frontend"""
    return HTMLResponse(content="""
    <script>window.location.href = '/frontend';</script>
    <p>Redirecting to frontend...</p>
    """)

@app.get("/frontend", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    try:
        with open("frontend.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found. Please ensure frontend.html exists.</h1>", status_code=404)

@app.get("/deployment/{user_id}", response_class=HTMLResponse)
async def serve_deployment_page(user_id: str):
    """Serve user-specific deployment page"""
    try:
        with open("deployment.html", "r") as f:
            content = f.read().replace("{{USER_ID}}", user_id)
            return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Deployment page not found.</h1>", status_code=404)

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    WEBSOCKET_CONNECTIONS[user_id] = websocket
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for connection testing
            await websocket.send_json({"type": "pong", "message": "Connection active"})
    except WebSocketDisconnect:
        WEBSOCKET_CONNECTIONS.pop(user_id, None)
        logger.info(f"WebSocket connection closed for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        WEBSOCKET_CONNECTIONS.pop(user_id, None)

@app.post("/search-models")
async def search_models(request: ModelSearchRequest):
    """Enhanced HuggingFace model search with validation"""
    try:
        logger.info(f"Search request: query='{request.query}', limit={request.limit}")
        models = await search_huggingface_models(
            request.query, 
            request.limit, 
            request.filter_type or "text-generation"
        )
        return {
            "models": models,
            "total": len(models),
            "query": request.query,
            "filter_type": request.filter_type
        }
    except Exception as e:
        logger.error(f"Error in search_models: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/deploy-model")
async def deploy_model(request: ModelDeployRequest, background_tasks: BackgroundTasks):
    """Enhanced model deployment with validation and progress tracking"""
    try:
        # Validate model compatibility first
        is_compatible, reason = validate_model_compatibility(request.model_name)
        if not is_compatible:
            raise HTTPException(status_code=400, detail=f"Model incompatible: {reason}")
        
        # Generate user ID if not provided
        user_id = request.user_id or generate_user_id()
        
        # Generate API key if enabled
        api_key = generate_api_key() if request.api_key_enabled else None
        
        external_ip = get_external_ip()
        base_url = f"http://{external_ip}:8000/user/{user_id}/v1"
        
        # Create comprehensive user deployment entry
        with _deployment_lock:
            USER_DEPLOYMENTS[user_id] = {
                "model_name": request.model_name,
                "backend": request.backend,
                "api_key": api_key,
                "api_key_enabled": request.api_key_enabled,
                "max_memory_gb": request.max_memory_gb,
                "status": "initializing",
                "progress": 0,
                "message": "Deployment queued",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "base_url": base_url,
                "external_ip": external_ip,
                "deployment_url": f"http://{external_ip}:8000/deployment/{user_id}"
            }
        
        # Start model loading in background
        background_tasks.add_task(
            load_user_model, 
            user_id, 
            request.model_name, 
            request.backend,
            request.max_memory_gb
        )
        
        logger.info(f"Started deployment for user {user_id}: {request.model_name}")
        
        return {
            "message": f"Deploying model {request.model_name}",
            "user_id": user_id,
            "model_name": request.model_name,
            "backend": request.backend,
            "api_key_enabled": request.api_key_enabled,
            "max_memory_gb": request.max_memory_gb,
            "base_url": base_url,
            "deployment_url": f"http://{external_ip}:8000/deployment/{user_id}",
            "websocket_url": f"ws://{external_ip}:8000/ws/{user_id}",
            "status": "initializing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in deploy_model: {e}")
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

async def load_user_model(user_id: str, model_name: str, backend: str, max_memory_gb: int = 8):
    """Enhanced model loading for a specific user with progress tracking"""
    try:
        logger.info(f"Starting model loading for user {user_id}: {model_name}")
        
        # Load the model with progress tracking
        model_data = await load_model_async(user_id, model_name, backend, max_memory_gb)
        
        # Store in active models
        ACTIVE_MODELS[user_id] = model_data
        
        # Final status update
        await update_deployment_status(user_id, "ready", 100, "Model deployment completed successfully!")
        
        logger.info(f"Model {model_name} loaded successfully for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error loading model for user {user_id}: {e}")
        await update_deployment_status(user_id, "error", 0, f"Deployment failed: {str(e)}")
        
        # Clean up any partial model data
        ACTIVE_MODELS.pop(user_id, None)

@app.get("/deployment-status/{user_id}")
async def get_deployment_status(user_id: str):
    """Enhanced deployment status with comprehensive information"""
    if user_id not in USER_DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="User deployment not found")
    
    deployment = USER_DEPLOYMENTS[user_id]
    
    # Add model information if available
    model_info = {}
    if user_id in ACTIVE_MODELS:
        model_data = ACTIVE_MODELS[user_id]
        model_info = {
            "device": model_data.get("device", "unknown"),
            "dtype": model_data.get("dtype", "unknown"),
            "memory_usage": model_data.get("memory_usage", "unknown"),
            "task": model_data.get("task", "unknown"),
            "loaded_at": model_data.get("loaded_at")
        }
    
    return {
        "user_id": user_id,
        "model_name": deployment["model_name"],
        "backend": deployment["backend"],
        "status": deployment["status"],
        "progress": deployment.get("progress", 0),
        "message": deployment.get("message", ""),
        "api_key": deployment.get("api_key"),
        "api_key_enabled": deployment["api_key_enabled"],
        "base_url": deployment["base_url"],
        "deployment_url": deployment.get("deployment_url"),
        "websocket_url": f"ws://{deployment.get('external_ip', 'localhost')}:8000/ws/{user_id}",
        "created_at": deployment["created_at"],
        "updated_at": deployment.get("updated_at"),
        "ready_at": deployment.get("ready_at"),
        "model_info": model_info,
        "error": deployment.get("error")
    }

@app.get("/user/{user_id}/v1/models")
async def get_user_models(user_id: str):
    """Enhanced OpenAI-compatible models endpoint"""
    if user_id not in USER_DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="User deployment not found")
    
    deployment = USER_DEPLOYMENTS[user_id]
    
    if deployment["status"] != "ready":
        raise HTTPException(
            status_code=503, 
            detail=f"Model not ready. Status: {deployment['status']}"
        )
    
    # Get additional model information
    model_info = {}
    if user_id in ACTIVE_MODELS:
        model_data = ACTIVE_MODELS[user_id]
        model_info = {
            "task": model_data.get("task", "text-generation"),
            "backend": model_data.get("backend", "transformers"),
            "device": model_data.get("device", "cpu"),
            "dtype": model_data.get("dtype", "float32")
        }
    
    created_timestamp = int(datetime.fromisoformat(deployment["created_at"]).timestamp())
    
    return {
        "object": "list",
        "data": [{
            "id": deployment["model_name"],
            "object": "model",
            "created": created_timestamp,
            "owned_by": f"user-{user_id}",
            "permission": [],
            "root": deployment["model_name"],
            "parent": None,
            **model_info
        }]
    }

@app.post("/user/{user_id}/v1/chat/completions")
async def user_chat_completions(user_id: str, request: ChatCompletionRequest, req: Request):
    """Enhanced OpenAI-compatible chat completions endpoint"""
    if user_id not in USER_DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="User deployment not found")
    
    deployment = USER_DEPLOYMENTS[user_id]
    
    # Check API key if enabled
    if deployment["api_key_enabled"]:
        auth_header = req.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        
        provided_key = auth_header.split(" ", 1)[1]
        if provided_key != deployment.get("api_key"):
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    if deployment["status"] != "ready":
        raise HTTPException(
            status_code=503, 
            detail=f"Model not ready. Status: {deployment['status']}"
        )
    
    if user_id not in ACTIVE_MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_data = ACTIVE_MODELS[user_id]
        pipeline = model_data["pipeline"]
        task = model_data["task"]
        
        # Enhanced prompt formatting based on model task
        if task == "text2text-generation":
            # For T5-style models, format differently
            prompt = ""
            for message in request.messages:
                if message["role"] == "user":
                    prompt += f"Question: {message['content']}\nAnswer: "
                elif message["role"] == "system":
                    prompt = f"{message['content']}\n" + prompt
        else:
            # For causal LM models
            prompt = ""
            for message in request.messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\nAssistant: "
                elif role == "assistant":
                    prompt += f"{content}\n"
        
        # Generate response with enhanced parameters
        generation_kwargs = {
            "max_length": len(prompt.split()) + request.max_tokens,
            "temperature": request.temperature,
            "do_sample": request.temperature > 0,
            "pad_token_id": pipeline.tokenizer.eos_token_id,
            "eos_token_id": pipeline.tokenizer.eos_token_id,
        }
        
        # Add optional parameters if provided
        if hasattr(request, 'top_p') and request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if hasattr(request, 'frequency_penalty') and request.frequency_penalty is not None:
            generation_kwargs["repetition_penalty"] = 1.0 + request.frequency_penalty
            
        response = pipeline(prompt, **generation_kwargs)
        
        # Extract generated text
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
        else:
            generated_text = str(response)
            
        # Clean up the response
        if task == "text2text-generation":
            assistant_response = generated_text.strip()
        else:
            assistant_response = generated_text[len(prompt):].strip()
        
        # Handle empty responses
        if not assistant_response:
            assistant_response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        completion_id = f"chatcmpl-{int(time.time())}-{secrets.token_hex(6)}"
        created_time = int(time.time())
        
        # Calculate token usage (approximate)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(assistant_response.split())
        
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_time,
            "model": deployment["model_name"],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "system_fingerprint": f"mp-{user_id[:8]}"
        }
        
    except Exception as e:
        logger.error(f"Error in chat completion for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Generation failed: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Enhanced server status with detailed information"""
    try:
        # Get deployment statistics
        deployment_stats = {"ready": 0, "deploying": 0, "error": 0, "initializing": 0}
        for deployment in USER_DEPLOYMENTS.values():
            status = deployment.get("status", "unknown")
            deployment_stats[status] = deployment_stats.get(status, 0) + 1
        
        # System information
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "server_info": {
                "external_ip": get_external_ip(),
                "port": 8000,
                "version": "2.0.0-production"
            },
            "deployments": {
                "total": len(USER_DEPLOYMENTS),
                "active_models": len(ACTIVE_MODELS),
                "statistics": deployment_stats
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "hardware": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "jax_devices": len(jax.devices()) if jax.devices() else 0
            },
            "websocket_connections": len(WEBSOCKET_CONNECTIONS)
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "active_deployments": len(USER_DEPLOYMENTS),
            "active_models": len(ACTIVE_MODELS),
            "external_ip": get_external_ip(),
            "error": str(e)
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check for load balancers"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Model Proxy Server...")
    logger.info(f"External IP: {get_external_ip()}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"JAX Devices: {len(jax.devices())}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        access_log=True,
        log_level="info"
    )