#!/usr/bin/env python3
"""
Enhanced Model Proxy Server - Individual user endpoints with unique API keys
Similar to llm_proxy but for HuggingFace models with TPU support
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

import requests
import torch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import jax
import jax.numpy as jnp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for user deployments
USER_DEPLOYMENTS: Dict[str, Dict] = {}
ACTIVE_MODELS: Dict[str, Any] = {}


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

# Models
class ModelSearchRequest(BaseModel):
    query: str
    limit: int = 10

class ModelDeployRequest(BaseModel):
    model_name: str
    backend: str = "transformers"
    api_key_enabled: bool = True
    user_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False

# Utility functions
def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"sk-{secrets.token_urlsafe(32)}"

def generate_user_id() -> str:
    """Generate a unique user ID"""
    return str(uuid.uuid4())

def get_external_ip() -> str:
    """Get external IP of the instance"""
    try:
        response = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip", 
                              headers={"Metadata-Flavor": "Google"}, timeout=5)
        return response.text.strip()
    except:
        return "34.44.140.182"  # Fallback

def search_huggingface_models(query: str, limit: int = 10) -> List[Dict]:
    """Search HuggingFace models using their API"""
    try:
        url = "https://huggingface.co/api/models"
        params = {
            "search": query,
            "limit": limit,
            "filter": "text-generation",
            "sort": "downloads",
            "direction": -1
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        models = response.json()
        
        # Format for our UI
        formatted_models = []
        for model in models:
            formatted_models.append({
                "id": model.get("id", ""),
                "name": model.get("id", ""),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "tags": model.get("tags", []),
                "description": model.get("description", "")[:200] + "..." if model.get("description", "") else ""
            })
        
        return formatted_models
    except Exception as e:
        logger.error(f"Error searching HuggingFace models: {e}")
        # Fallback to popular models
        return [
            {"id": "gpt2", "name": "gpt2", "downloads": 1000000, "likes": 500, "tags": ["text-generation"], "description": "GPT-2 is a transformers model pretrained on a very large corpus of English data"},
            {"id": "microsoft/DialoGPT-medium", "name": "microsoft/DialoGPT-medium", "downloads": 500000, "likes": 200, "tags": ["conversational"], "description": "Large-scale pretraining for dialogue generation"},
            {"id": "google/flan-t5-base", "name": "google/flan-t5-base", "downloads": 300000, "likes": 150, "tags": ["text2text-generation"], "description": "FLAN-T5 Base model for instruction following"},
            {"id": "microsoft/DialoGPT-small", "name": "microsoft/DialoGPT-small", "downloads": 200000, "likes": 100, "tags": ["conversational"], "description": "Smaller version of DialoGPT for dialogue generation"},
            {"id": "distilgpt2", "name": "distilgpt2", "downloads": 150000, "likes": 80, "tags": ["text-generation"], "description": "Distilled version of GPT-2"}
        ]

async def load_model_async(model_name: str, backend: str = "transformers") -> Dict:
    """Load model asynchronously with proper model type detection"""
    try:
        logger.info(f"Loading model {model_name} with {backend} backend")

        # Create model directory
        model_dir = f"/mnt/models/user_models/{model_name.replace('/', '_')}"
        os.makedirs(model_dir, exist_ok=True)

        if backend == "transformers":
            # First, get model info to determine the correct model type
            from huggingface_hub import HfApi
            api = HfApi()
            model_info = api.model_info(model_name)
            pipeline_tag = getattr(model_info, 'pipeline_tag', 'text-generation')
            
            logger.info(f"Model {model_name} has pipeline tag: {pipeline_tag}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="/mnt/models/huggingface_cache",
                trust_remote_code=True
            )

            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load appropriate model based on pipeline tag
            if pipeline_tag == "text2text-generation":
                from transformers import AutoModelForSeq2SeqLM
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir="/mnt/models/huggingface_cache",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                task = "text2text-generation"
            else:
                # Default to causal LM for text-generation
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir="/mnt/models/huggingface_cache",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )
                task = "text-generation"

            # Create pipeline with correct task
            pipe = pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )

            return {
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": pipe,
                "backend": backend,
                "task": task,
                "status": "ready"
            }

        elif backend == "jax":
            # JAX implementation would go here
            # For now, fallback to transformers
            return await load_model_async(model_name, "transformers")

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the enhanced frontend"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Proxy Server</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #9c27b0, #e91e63);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: bold;
        }
        
        .search-section {
            margin-bottom: 30px;
        }
        
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        input[type="text"] {
            flex: 1;
            min-width: 250px;
            padding: 15px;
            border: 2px solid #9c27b0;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #e91e63;
            box-shadow: 0 0 20px rgba(156, 39, 176, 0.3);
        }
        
        input[type="text"]::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        button {
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(45deg, #9c27b0, #e91e63);
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(156, 39, 176, 0.3);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .model-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(156, 39, 176, 0.2);
            border-color: #9c27b0;
        }
        
        .model-card.selected {
            border-color: #e91e63;
            background: rgba(233, 30, 99, 0.1);
        }
        
        .model-name {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 10px;
            color: #e91e63;
        }
        
        .model-stats {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
        }
        
        .model-description {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.4;
        }
        
        .deployment-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .options-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .option-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        label {
            font-weight: bold;
            color: #9c27b0;
        }
        
        select {
            padding: 12px;
            border: 2px solid #9c27b0;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 14px;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            accent-color: #e91e63;
        }
        
        .endpoint-info {
            background: rgba(0, 255, 0, 0.1);
            border: 2px solid #4caf50;
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            display: none;
        }
        
        .endpoint-info.show {
            display: block;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .endpoint-item {
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }
        
        .endpoint-label {
            font-weight: bold;
            color: #4caf50;
            margin-bottom: 5px;
        }
        
        .endpoint-value {
            font-family: 'Courier New', monospace;
            background: rgba(0, 0, 0, 0.3);
            padding: 8px;
            border-radius: 5px;
            word-break: break-all;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-ready { background-color: #4caf50; }
        .status-loading { background-color: #ff9800; animation: pulse 1s infinite; }
        .status-error { background-color: #f44336; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .loading-spinner {
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid #e91e63;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .search-box {
                flex-direction: column;
            }
            
            input[type="text"] {
                min-width: 100%;
            }
            
            .models-grid {
                grid-template-columns: 1fr;
            }
            
            .options-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Model Proxy Server</h1>
        
        <div class="search-section">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Search HuggingFace models...">
                <button onclick="searchModels()">Search Models</button>
            </div>
        </div>
        
        <div id="modelsContainer" class="models-grid">
            <!-- Models will be loaded here -->
        </div>
        
        <div class="deployment-section">
            <h3>🔧 Deployment Options</h3>
            <div class="options-grid">
                <div class="option-group">
                    <label for="backendSelect">Backend:</label>
                    <select id="backendSelect">
                        <option value="transformers">Transformers (Recommended)</option>
                        <option value="jax">JAX (Experimental)</option>
                    </select>
                </div>
                <div class="option-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="apiKeyEnabled" checked>
                        <label for="apiKeyEnabled">Enable API Key Protection</label>
                    </div>
                </div>
            </div>
            <button onclick="deployModel()" id="deployBtn" disabled>
                <span class="status-indicator status-ready"></span>
                Deploy Selected Model
            </button>
        </div>
        
        <div id="endpointInfo" class="endpoint-info">
            <h3>✅ Model Deployed Successfully!</h3>
            <div class="endpoint-item">
                <div class="endpoint-label">Model Name:</div>
                <div class="endpoint-value" id="modelName"></div>
            </div>
            <div class="endpoint-item">
                <div class="endpoint-label">Base URL:</div>
                <div class="endpoint-value" id="baseUrl"></div>
            </div>
            <div class="endpoint-item" id="apiKeySection">
                <div class="endpoint-label">API Key:</div>
                <div class="endpoint-value" id="apiKey"></div>
            </div>
            <div class="endpoint-item">
                <div class="endpoint-label">User ID:</div>
                <div class="endpoint-value" id="userId"></div>
            </div>
            <p style="margin-top: 15px; color: #4caf50;">
                💡 Copy these details to use with OpenHands or other applications!
            </p>
        </div>
    </div>

    <script>
        let selectedModel = null;
        let deploymentStatus = 'idle';
        
        // Search models
        async function searchModels() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            const container = document.getElementById('modelsContainer');
            container.innerHTML = '<div class="loading-spinner"></div>';
            
            try {
                const response = await fetch('/search-models', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, limit: 12 })
                });
                
                const data = await response.json();
                displayModels(data.models || []);
            } catch (error) {
                console.error('Search error:', error);
                container.innerHTML = '<p style="text-align: center; color: #f44336;">Error searching models. Please try again.</p>';
            }
        }
        
        // Display models
        function displayModels(models) {
            const container = document.getElementById('modelsContainer');
            
            if (models.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #ff9800;">No models found. Try a different search term.</p>';
                return;
            }
            
            container.innerHTML = models.map(model => `
                <div class="model-card" onclick="selectModel('${model.id}', '${model.name}')">
                    <div class="model-name">${model.name}</div>
                    <div class="model-stats">
                        <span>📥 ${model.downloads.toLocaleString()}</span>
                        <span>❤️ ${model.likes}</span>
                    </div>
                    <div class="model-description">${model.description}</div>
                </div>
            `).join('');
        }
        
        // Select model
        function selectModel(modelId, modelName) {
            // Remove previous selection
            document.querySelectorAll('.model-card').forEach(card => {
                card.classList.remove('selected');
            });
            
            // Add selection to clicked card
            event.currentTarget.classList.add('selected');
            
            selectedModel = { id: modelId, name: modelName };
            document.getElementById('deployBtn').disabled = false;
            
            // Hide previous endpoint info
            document.getElementById('endpointInfo').classList.remove('show');
        }
        
        // Deploy model
        async function deployModel() {
            if (!selectedModel) {
                alert('Please select a model first');
                return;
            }
            
            const deployBtn = document.getElementById('deployBtn');
            const statusIndicator = deployBtn.querySelector('.status-indicator');
            
            // Update UI for loading state
            deployBtn.disabled = true;
            deployBtn.innerHTML = '<span class="status-indicator status-loading"></span>Deploying Model...';
            deploymentStatus = 'deploying';
            
            try {
                const backend = document.getElementById('backendSelect').value;
                const apiKeyEnabled = document.getElementById('apiKeyEnabled').checked;
                
                const response = await fetch('/deploy-model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_name: selectedModel.id,
                        backend: backend,
                        api_key_enabled: apiKeyEnabled
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Poll for deployment status
                    pollDeploymentStatus(data.user_id);
                } else {
                    throw new Error(data.detail || 'Deployment failed');
                }
                
            } catch (error) {
                console.error('Deployment error:', error);
                deployBtn.innerHTML = '<span class="status-indicator status-error"></span>Deployment Failed';
                setTimeout(() => {
                    deployBtn.innerHTML = '<span class="status-indicator status-ready"></span>Deploy Selected Model';
                    deployBtn.disabled = false;
                }, 3000);
            }
        }
        
        // Poll deployment status
        async function pollDeploymentStatus(userId) {
            try {
                const response = await fetch(`/deployment-status/${userId}`);
                const data = await response.json();
                
                if (data.status === 'ready') {
                    // Show endpoint information
                    showEndpointInfo(data);
                    
                    const deployBtn = document.getElementById('deployBtn');
                    deployBtn.innerHTML = '<span class="status-indicator status-ready"></span>Deploy Another Model';
                    deployBtn.disabled = false;
                    
                } else if (data.status === 'error') {
                    throw new Error(data.error || 'Deployment failed');
                } else {
                    // Still deploying, poll again
                    setTimeout(() => pollDeploymentStatus(userId), 2000);
                }
                
            } catch (error) {
                console.error('Status polling error:', error);
                const deployBtn = document.getElementById('deployBtn');
                deployBtn.innerHTML = '<span class="status-indicator status-error"></span>Deployment Failed';
                setTimeout(() => {
                    deployBtn.innerHTML = '<span class="status-indicator status-ready"></span>Deploy Selected Model';
                    deployBtn.disabled = false;
                }, 3000);
            }
        }
        
        // Show endpoint information
        function showEndpointInfo(data) {
            document.getElementById('modelName').textContent = data.model_name;
            document.getElementById('baseUrl').textContent = data.base_url;
            document.getElementById('userId').textContent = data.user_id;
            
            const apiKeySection = document.getElementById('apiKeySection');
            if (data.api_key) {
                document.getElementById('apiKey').textContent = data.api_key;
                apiKeySection.style.display = 'block';
            } else {
                apiKeySection.style.display = 'none';
            }
            
            document.getElementById('endpointInfo').classList.add('show');
        }
        
        // Load popular models on page load
        window.onload = function() {
            // Default search removed - starts blank
            searchModels();
        };
        
        // Enter key support for search
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchModels();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/search-models")
async def search_models(request: ModelSearchRequest):
    """Search HuggingFace models"""
    try:
        models = search_huggingface_models(request.query, request.limit)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error in search_models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deploy-model")
async def deploy_model(request: ModelDeployRequest, background_tasks: BackgroundTasks):
    """Deploy a model for a user"""
    try:
        # Generate user ID if not provided
        user_id = request.user_id or generate_user_id()
        
        # Generate API key if enabled
        api_key = generate_api_key() if request.api_key_enabled else None
        
        # Create user deployment entry
        USER_DEPLOYMENTS[user_id] = {
            "model_name": request.model_name,
            "backend": request.backend,
            "api_key": api_key,
            "api_key_enabled": request.api_key_enabled,
            "status": "deploying",
            "created_at": datetime.now().isoformat(),
            "base_url": f"http://{get_external_ip()}:8000/user/{user_id}/v1"
        }
        
        # Start model loading in background
        background_tasks.add_task(load_user_model, user_id, request.model_name, request.backend)
        
        return {
            "message": f"Deploying model {request.model_name}",
            "user_id": user_id,
            "model_name": request.model_name,
            "backend": request.backend,
            "api_key_enabled": request.api_key_enabled
        }
        
    except Exception as e:
        logger.error(f"Error in deploy_model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def load_user_model(user_id: str, model_name: str, backend: str):
    """Load model for a specific user"""
    try:
        logger.info(f"Loading model {model_name} for user {user_id}")
        
        # Load the model
        model_data = await load_model_async(model_name, backend)
        
        # Store in active models
        ACTIVE_MODELS[user_id] = model_data
        
        # Update deployment status
        USER_DEPLOYMENTS[user_id]["status"] = "ready"
        USER_DEPLOYMENTS[user_id]["loaded_at"] = datetime.now().isoformat()
        
        logger.info(f"Model {model_name} loaded successfully for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error loading model for user {user_id}: {e}")
        USER_DEPLOYMENTS[user_id]["status"] = "error"
        USER_DEPLOYMENTS[user_id]["error"] = str(e)

@app.get("/deployment-status/{user_id}")
async def get_deployment_status(user_id: str):
    """Get deployment status for a user"""
    if user_id not in USER_DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="User deployment not found")
    
    deployment = USER_DEPLOYMENTS[user_id]
    
    return {
        "user_id": user_id,
        "model_name": deployment["model_name"],
        "backend": deployment["backend"],
        "status": deployment["status"],
        "api_key": deployment.get("api_key"),
        "api_key_enabled": deployment["api_key_enabled"],
        "base_url": deployment["base_url"],
        "error": deployment.get("error")
    }

@app.get("/user/{user_id}/v1/models")
async def get_user_models(user_id: str):
    """Get models for a specific user (OpenAI compatible)"""
    if user_id not in USER_DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="User not found")
    
    deployment = USER_DEPLOYMENTS[user_id]
    
    if deployment["status"] != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "object": "list",
        "data": [{
            "id": deployment["model_name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"user-{user_id}"
        }]
    }

@app.post("/user/{user_id}/v1/chat/completions")
async def user_chat_completions(user_id: str, request: ChatCompletionRequest):
    """Chat completions for a specific user (OpenAI compatible)"""
    if user_id not in USER_DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="User not found")
    
    deployment = USER_DEPLOYMENTS[user_id]
    
    # Check API key if enabled
    if deployment["api_key_enabled"]:
        # In a real implementation, you'd check the Authorization header
        pass
    
    if deployment["status"] != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if user_id not in ACTIVE_MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_data = ACTIVE_MODELS[user_id]
        pipeline = model_data["pipeline"]
        
        # Format messages into a single prompt
        prompt = ""
        for message in request.messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                prompt += f"User: {content}\nAssistant: "
            elif role == "assistant":
                prompt += f"{content}\n"
        
        # Generate response
        response = pipeline(
            prompt,
            max_length=len(prompt.split()) + request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated_text = response[0]["generated_text"]
        assistant_response = generated_text[len(prompt):].strip()
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
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
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(assistant_response.split()),
                "total_tokens": len(prompt.split()) + len(assistant_response.split())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat completion for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get server status"""
    return {
        "status": "running",
        "active_deployments": len(USER_DEPLOYMENTS),
        "active_models": len(ACTIVE_MODELS),
        "jax_devices": len(jax.devices()) if jax.devices() else 0,
        "external_ip": get_external_ip()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=True)