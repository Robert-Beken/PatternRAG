# API Reference

PatternRAG provides a RESTful API that is compatible with OpenAI's chat completions API. This allows it to be easily integrated with existing tools and interfaces that support the OpenAI API format.

## Base URL

The API is available at:

```
http://localhost:8000
```

You can change the host and port using the `--host` and `--port` options when starting the service.

## Authentication

By default, the API does not require authentication. To enable authentication, set the `auth.enabled` option to `true` in your configuration file and specify an API key:

```yaml
auth:
  enabled: true
  api_key: "your-secure-api-key"
```

When authentication is enabled, include the API key in the request header:

```
Authorization: Bearer your-secure-api-key
```

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
POST /chat/completions
```

This endpoint accepts chat messages and returns a response. It is compatible with the OpenAI Chat API format.

#### Request Body

```json
{
  "model": "pattern-rag",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Find connections between quantum physics and consciousness."}
  ],
  "stream": false
}
```

Parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model to use. Use "pattern-rag" for PatternRAG or any other model supported by your LLM provider. |
| `messages` | array | Yes | Array of message objects. Each message has a role ("system", "user", "assistant") and content. |
| `stream` | boolean | No | Whether to stream the response. Default is false. |

#### Response

```json
{
  "id": "chatcmpl-123456",
  "object": "chat.completion",
  "model": "pattern-rag",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I've analyzed several connections between quantum physics and consciousness...",
        "context": {
          "sources": [
            {
              "source": "/path/to/document.pdf",
              "content": "Quantum mechanics has often been invoked in discussions of consciousness...",
              "title": "Quantum Mind",
              "author": "John Smith",
              "chunk_type": "paragraph"
            }
          ],
          "connections": [
            "Both quantum physics and consciousness involve observer effects",
            "The concept of non-locality appears in both quantum entanglement and certain theories of consciousness",
            "Quantum coherence has been proposed as a mechanism for neural processes"
          ],
          "related_entities": ["David Bohm", "Roger Penrose", "quantum entanglement"],
          "metadata": {
            "processing_time": "2.35s",
            "search_mode": "pattern",
            "expanded_queries": [
              "Find connections between quantum physics and consciousness.",
              "How do quantum mechanical principles relate to theories of mind?",
              "What parallels exist between quantum phenomena and conscious experience?"
            ]
          }
        }
      },
      "finish_reason": "stop"
    }
  ]
}
```

The response follows the OpenAI format with an additional `context` field that contains information about the sources, connections, and other metadata.

### Models

```
GET /models
```

Returns a list of available models.

#### Response

```json
{
  "models": ["pattern-rag", "llama2", "llama3", "mistral"]
}
```

### OpenAI-compatible Models

```
GET /v1/models
```

Returns a list of available models in OpenAI format.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "pattern-rag",
      "object": "model",
      "created": 1678999999,
      "owned_by": "user"
    },
    {
      "id": "llama2",
      "object": "model",
      "created": 1678999999,
      "owned_by": "llm-service"
    }
  ]
}
```

### Health Check

```
GET /health
```

Returns the status of the service.

#### Response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "service": "PatternRAG",
  "timestamp": 1678999999.123
}
```

## Using the API

### Python Example

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "pattern-rag",
    "messages": [
        {"role": "user", "content": "What patterns connect ancient architectural techniques across different civilizations?"}
    ]
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(json.dumps(result, indent=2))
```

### Curl Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pattern-rag",
    "messages": [
      {"role": "user", "content": "What connections exist between ancient Egyptian and Mayan mathematics?"}
    ]
  }'
```

### Streaming Example

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "pattern-rag",
    "messages": [
        {"role": "user", "content": "What patterns connect ancient architectural techniques across different civilizations?"}
    ],
    "stream": true
}

response = requests.post(url, headers=headers, json=data, stream=True)

for line in response.iter_lines():
    if line:
        line_text = line.decode('utf-8')
        if line_text.startswith('data: '):
            json_str = line_text[6:]  # Remove 'data: ' prefix
            try:
                chunk = json.loads(json_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
            except json.JSONDecodeError:
                continue
```

## Integration with OpenWebUI

PatternRAG is compatible with [OpenWebUI](https://github.com/open-webui/open-webui) and similar interfaces. To use PatternRAG with OpenWebUI:

1. Start the PatternRAG service:
   ```bash
   python -m patternrag.service
   ```

2. In OpenWebUI, add a new model provider with:
   - Provider URL: `http://localhost:8000`
   - API Key: (leave empty unless you've enabled authentication)

3. Select "pattern-rag" from the model list in the interface.

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication failed
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses follow this format:

```json
{
  "error": "Detailed error message"
}
```

## Rate Limiting

By default, PatternRAG does not implement rate limiting. For production deployments, we recommend using a reverse proxy like Nginx or a Kubernetes ingress controller that supports rate limiting.
