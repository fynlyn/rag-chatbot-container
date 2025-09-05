# Model Management Features

## Overview
The RAG chatbot now includes comprehensive model management capabilities that allow users to browse, download, install, and manage LLM models through both the web UI and REST API.

## Web UI Features

### 🤖 Models Tab
The web interface now includes a "Models" tab with the following functionality:

1. **Current Model Display**: Shows which model is currently active
2. **Available Models**: Browse popular models optimized for RAG
3. **Installed Models**: View and manage currently installed models
4. **One-Click Operations**: Download, activate, and remove models with simple clicks

### Model Cards
Each model is displayed in a card format showing:
- **Model Name**: e.g., `llama3.2:3b`
- **Status Badge**: Available, Installed, Downloading, Error
- **Model Info**: Parameter size (e.g., 3B), Download size (e.g., ~2GB)
- **Description**: Brief description of model capabilities
- **Action Buttons**: Download, Use This Model, Remove

### Real-Time Updates
- Download progress is displayed in real-time
- Status updates automatically without page refresh
- Visual indicators for model states (colors, badges)

## REST API Endpoints

### GET /models/available
Returns list of popular models available for download.

```json
[
  {
    "name": "llama3.2:3b",
    "size": "~2GB",
    "status": "installed",
    "description": "Latest Llama 3.2 model, 3B parameters - good balance of speed and quality",
    "family": "llama",
    "parameter_size": "3B"
  }
]
```

### GET /models/installed
Returns list of currently installed models.

### POST /models/download
Start downloading a model in background.
```json
{"model_name": "qwen2:1.5b"}
```

### GET /models/download/status/{model_name}
Check download progress for a specific model.

### POST /models/set-active
Set the active model for chat.
```json
{"model_name": "phi3:mini"}
```

### DELETE /models/remove/{model_name}
Remove an installed model.

## Supported Models

The system includes a curated list of models optimized for RAG applications:

### Lightweight Models (< 2GB)
- **llama3.2:1b** - Fastest option, 1B parameters (~1GB)
- **qwen2:1.5b** - Multilingual support, 1.5B parameters (~934MB)
- **gemma2:2b** - Google's efficient model, 2B parameters (~1.6GB)

### Balanced Models (2-3GB)
- **llama3.2:3b** - Best balance of speed and quality, 3B parameters (~2GB)
- **phi3:mini** - Microsoft's efficient model, 3.8B parameters (~2.2GB)

### High-Quality Models (4GB+)
- **llama3.1:8b** - High-quality responses, 8B parameters (~4.9GB)

## Usage Examples

### Via Web UI
1. Open the web interface
2. Click the "🤖 Models" tab
3. Browse available models
4. Click "Download" on desired model
5. Wait for download to complete
6. Click "Use This Model" to activate

### Via API
```bash
# List available models
curl http://localhost:8000/models/available

# Download a model
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen2:1.5b"}'

# Check download status
curl http://localhost:8000/models/download/status/qwen2:1.5b

# Set active model
curl -X POST http://localhost:8000/models/set-active \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen2:1.5b"}'
```

## Background Processing
- Model downloads happen in the background
- Users can continue using the system while models download
- Download status is tracked and can be queried
- Automatic cleanup of download status after completion

## Error Handling
- Network failures are handled gracefully
- Download errors are reported to the user
- Invalid model names are rejected
- Proper HTTP status codes for all operations

## Integration
The model management system is fully integrated with:
- The existing chat functionality
- Configuration system (updates active model)
- Status endpoint (shows current model info)
- Web UI (seamless tab switching)

This provides users with complete control over their RAG system's language models without needing command-line access or Docker knowledge.
