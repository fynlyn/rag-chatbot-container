from __future__ import annotations

import asyncio
import json
from typing import List, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from ..config import settings
from ..llm import ollama

router = APIRouter(prefix="/models", tags=["models"])


class ModelInfo(BaseModel):
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None
    status: str = "available"  # available, installed, downloading, error
    description: Optional[str] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    tags: Optional[List[str]] = None
    is_official: Optional[bool] = None
    pulls: Optional[int] = None
    updated_at: Optional[str] = None


class ModelDownloadRequest(BaseModel):
    model_name: str


class ModelSearchRequest(BaseModel):
    query: str
    limit: int = 20


class ModelDownloadStatus(BaseModel):
    model_name: str
    status: str  # downloading, completed, error
    progress: Optional[str] = None
    error: Optional[str] = None


# Store download statuses
download_statuses: Dict[str, ModelDownloadStatus] = {}


@router.get("/search")
async def search_models(q: str = Query(..., description="Search query for models"), 
                       limit: int = Query(20, description="Maximum number of results")):
    """Search for models in the Ollama library"""
    try:
        # Search Ollama library using their search API
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Note: This uses the Ollama library search API
            # If this doesn't work, we'll fall back to a local search
            search_url = f"https://ollama.com/api/search"
            params = {"q": q, "limit": limit}
            
            try:
                r = await client.get(search_url, params=params)
                r.raise_for_status()
                data = r.json()
                
                # Process the search results
                models = []
                for model_data in data.get("models", []):
                    model_info = ModelInfo(
                        name=model_data.get("name", ""),
                        description=model_data.get("description", ""),
                        tags=model_data.get("tags", []),
                        is_official=model_data.get("is_official", False),
                        pulls=model_data.get("pulls", 0),
                        updated_at=model_data.get("updated_at"),
                        status="available"
                    )
                    models.append(model_info)
                
                # Check which ones are installed
                installed_models = await get_installed_models()
                installed_names = {m.name for m in installed_models}
                
                for model in models:
                    if model.name in installed_names:
                        model.status = "installed"
                    elif model.name in download_statuses:
                        status = download_statuses[model.name]
                        model.status = status.status
                
                return models
                
            except Exception as e:
                # Fallback: search our curated list + installed models
                return await fallback_search(q, limit)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def fallback_search(query: str, limit: int) -> List[ModelInfo]:
    """Fallback search in curated models + installed models when API search fails"""
    query_lower = query.lower()
    
    # Extended curated list with more models including DeepSeek
    all_models = [
        {
            "name": "llama3.2:3b",
            "description": "Latest Llama 3.2 model, 3B parameters - good balance of speed and quality",
            "family": "llama",
            "parameter_size": "3B",
            "size": "~2GB"
        },
        {
            "name": "llama3.2:1b",
            "description": "Smallest Llama 3.2 model, 1B parameters - fastest option",
            "family": "llama", 
            "parameter_size": "1B",
            "size": "~1GB"
        },
        {
            "name": "llama3.1:8b",
            "description": "Llama 3.1 - 8B parameters, high quality responses",
            "family": "llama",
            "parameter_size": "8B",
            "size": "~4.9GB"
        },
        {
            "name": "phi3:mini",
            "description": "Microsoft Phi-3 Mini - efficient and fast for chat",
            "family": "phi3",
            "parameter_size": "3.8B",
            "size": "~2.2GB"
        },
        {
            "name": "gemma2:2b",
            "description": "Google Gemma 2 - 2B parameters, optimized for efficiency",
            "family": "gemma2",
            "parameter_size": "2B", 
            "size": "~1.6GB"
        },
        {
            "name": "qwen2:1.5b",
            "description": "Qwen 2 - 1.5B parameters, multilingual support",
            "family": "qwen2",
            "parameter_size": "1.5B",
            "size": "~934MB"
        },
        {
            "name": "qwen2:7b",
            "description": "Qwen 2 - 7B parameters, larger multilingual model",
            "family": "qwen2",
            "parameter_size": "7B",
            "size": "~4.4GB"
        },
        {
            "name": "deepseek-r1:latest",
            "description": "DeepSeek R1 - Advanced reasoning model with chain of thought",
            "family": "deepseek",
            "parameter_size": "Unknown",
            "size": "~Variable",
            "tags": ["reasoning", "chain-of-thought"]
        },
        {
            "name": "deepseek-r1:1.5b",
            "description": "DeepSeek R1 - 1.5B parameters, compact reasoning model",
            "family": "deepseek",
            "parameter_size": "1.5B",
            "size": "~1.5GB",
            "tags": ["reasoning", "small"]
        },
        {
            "name": "deepseek-r1:7b",
            "description": "DeepSeek R1 - 7B parameters, powerful reasoning capabilities",
            "family": "deepseek", 
            "parameter_size": "7B",
            "size": "~4.3GB",
            "tags": ["reasoning", "large"]
        },
        {
            "name": "deepseek-coder:1.3b",
            "description": "DeepSeek Coder - Specialized for code generation and understanding",
            "family": "deepseek",
            "parameter_size": "1.3B",
            "size": "~1.3GB",
            "tags": ["coding", "programming"]
        },
        {
            "name": "codellama:7b",
            "description": "Code Llama - Meta's code-specialized language model",
            "family": "llama",
            "parameter_size": "7B", 
            "size": "~3.8GB",
            "tags": ["coding", "programming"]
        },
        {
            "name": "mistral:7b",
            "description": "Mistral 7B - Efficient and powerful general-purpose model",
            "family": "mistral",
            "parameter_size": "7B",
            "size": "~4.1GB"
        },
        {
            "name": "neural-chat:7b",
            "description": "Intel's neural chat model optimized for conversations",
            "family": "neural-chat",
            "parameter_size": "7B",
            "size": "~4.1GB",
            "tags": ["chat", "conversation"]
        }
    ]
    
    # Filter models based on query
    filtered_models = []
    for model in all_models:
        if (query_lower in model["name"].lower() or 
            query_lower in model.get("description", "").lower() or
            query_lower in model.get("family", "").lower() or
            any(query_lower in tag for tag in model.get("tags", []))):
            filtered_models.append(model)
    
    # Get installed models to check status
    try:
        installed_models = await get_installed_models()
        installed_names = {m.name for m in installed_models}
    except:
        installed_names = set()
    
    # Convert to ModelInfo objects
    result = []
    for model in filtered_models[:limit]:
        model_info = ModelInfo(
            name=model["name"],
            description=model["description"],
            family=model["family"],
            parameter_size=model["parameter_size"],
            size=model["size"],
            tags=model.get("tags"),
            status="installed" if model["name"] in installed_names else "available"
        )
        
        # Check if currently downloading
        if model["name"] in download_statuses:
            status = download_statuses[model["name"]]
            model_info.status = status.status
            
        result.append(model_info)
    
    return result


@router.get("/available", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of popular available models from Ollama library"""
    # Popular models that work well for RAG
    popular_models = [
        {
            "name": "llama3.2:3b",
            "description": "Latest Llama 3.2 model, 3B parameters - good balance of speed and quality",
            "family": "llama",
            "parameter_size": "3B",
            "size": "~2GB"
        },
        {
            "name": "llama3.2:1b",
            "description": "Smallest Llama 3.2 model, 1B parameters - fastest option",
            "family": "llama", 
            "parameter_size": "1B",
            "size": "~1GB"
        },
        {
            "name": "phi3:mini",
            "description": "Microsoft Phi-3 Mini - efficient and fast for chat",
            "family": "phi3",
            "parameter_size": "3.8B",
            "size": "~2.2GB"
        },
        {
            "name": "gemma2:2b",
            "description": "Google Gemma 2 - 2B parameters, optimized for efficiency",
            "family": "gemma2",
            "parameter_size": "2B", 
            "size": "~1.6GB"
        },
        {
            "name": "qwen2:1.5b",
            "description": "Qwen 2 - 1.5B parameters, multilingual support",
            "family": "qwen2",
            "parameter_size": "1.5B",
            "size": "~934MB"
        },
        {
            "name": "llama3.1:8b",
            "description": "Llama 3.1 - 8B parameters, high quality responses",
            "family": "llama",
            "parameter_size": "8B",
            "size": "~4.9GB"
        }
    ]
    
    # Get currently installed models
    try:
        installed_models = await get_installed_models()
        installed_names = {m.name for m in installed_models}
        
        result = []
        for model in popular_models:
            model_info = ModelInfo(
                name=model["name"],
                description=model["description"],
                family=model["family"],
                parameter_size=model["parameter_size"],
                size=model["size"],
                status="installed" if model["name"] in installed_names else "available"
            )
            
            # Check if currently downloading
            if model["name"] in download_statuses:
                status = download_statuses[model["name"]]
                model_info.status = status.status
                
            result.append(model_info)
            
        return result
        
    except Exception as e:
        # Fallback to basic list if Ollama is not available
        return [ModelInfo(**model, status="available") for model in popular_models]


@router.get("/installed", response_model=List[ModelInfo])
async def get_installed_models():
    """Get list of currently installed models"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{settings.llm_base_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            
            models = []
            for model in data.get("models", []):
                details = model.get("details", {})
                models.append(ModelInfo(
                    name=model["name"],
                    size=f"{model.get('size', 0) / (1024*1024*1024):.1f}GB" if model.get('size') else None,
                    modified_at=model.get("modified_at"),
                    status="installed",
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size")
                ))
                
            return models
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get installed models: {str(e)}")


@router.post("/download")
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading a model in the background"""
    model_name = request.model_name
    
    # Check if already downloading
    if model_name in download_statuses:
        status = download_statuses[model_name]
        if status.status == "downloading":
            return {"message": f"Model {model_name} is already downloading", "status": status}
    
    # Initialize download status
    download_statuses[model_name] = ModelDownloadStatus(
        model_name=model_name,
        status="downloading",
        progress="Starting download..."
    )
    
    # Start download in background
    background_tasks.add_task(download_model_task, model_name)
    
    return {
        "message": f"Started downloading {model_name}",
        "status": download_statuses[model_name]
    }


@router.get("/download/status/{model_name}")
async def get_download_status(model_name: str):
    """Get download status for a specific model"""
    if model_name in download_statuses:
        return download_statuses[model_name]
    else:
        return {"model_name": model_name, "status": "not_found", "error": "No download in progress"}


@router.post("/set-active")
async def set_active_model(request: ModelDownloadRequest):
    """Set the active model for chat"""
    model_name = request.model_name
    
    # Verify model is installed
    try:
        installed = await get_installed_models()
        installed_names = {m.name for m in installed}
        
        if model_name not in installed_names:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not installed")
        
        # Update configuration (in production, this would update the config file)
        settings.llm_model = model_name
        
        return {
            "message": f"Active model set to {model_name}",
            "model": model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set active model: {str(e)}")


async def download_model_task(model_name: str):
    """Background task to download a model"""
    try:
        download_statuses[model_name].progress = "Connecting to Ollama..."
        
        # Use ollama client to pull the model
        await ollama.ensure_model(model_name)
        
        download_statuses[model_name].status = "completed"
        download_statuses[model_name].progress = "Download completed successfully"
        
    except Exception as e:
        download_statuses[model_name].status = "error"
        download_statuses[model_name].error = str(e)
        download_statuses[model_name].progress = f"Download failed: {str(e)}"


@router.delete("/remove/{model_name}")
async def remove_model(model_name: str):
    """Remove an installed model"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.delete(f"{settings.llm_base_url}/api/delete", json={"name": model_name})
            r.raise_for_status()
            
        return {"message": f"Model {model_name} removed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove model: {str(e)}")
