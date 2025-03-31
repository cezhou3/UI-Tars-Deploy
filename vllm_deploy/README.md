# UI-TARS-7B-DPO Deployment

This project provides scripts to deploy the UI-TARS-7B multimodal model using vLLM with advanced high-performance optimizations, supporting both local and Slurm deployment options. UI-TARS-7B is based on QWenVL2-7B, a state-of-the-art model designed for efficient text and image processing.

## About QWenVL2-7B

QWenVL2-7B is a powerful multimodal model capable of understanding both text and images. This deployment leverages vLLM's advanced features for maximum throughput and minimum latency.

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- At least 24GB GPU memory (recommended)

## Installation

```bash
pip install -r requirements.txt
```

## Advanced High-Performance Features

- AWQ quantization for memory efficiency
- Optimized KV cache management
- 97% GPU memory utilization
- Paged attention mechanism
- Block-wise tensor operations
- Increased batch sizes and reduced wait times
- Optimized token processing
- Tensor parallelism support for multi-GPU deployment

## Usage

### Local Deployment

```bash
bash run_local.sh
```

### Slurm Deployment

```bash
bash run_slurm.sh
```

After submitting the Slurm job, the system will:
1. Generate a unique API key for secure access
2. Start the vLLM server on the allocated node
3. Create a `server_connection_info.txt` file with connection details

## Remote Access

### Finding Connection Information

After the Slurm job starts, check the `server_connection_info.txt` file which contains:
- Server URL (typically http://[node-ip]:8000)
- Generated API key
- Example curl command

### Example: Connecting from Local Machine

1. Note the server URL and API key from `server_connection_info.txt`
2. Use the query_model.py script:

```bash
python examples/query_model.py --host [node-ip] --port 8000 --api-key [your-api-key] --prompt "Hello, world!"
```

### API Key Authentication

All requests to the server require authentication using the generated API key:

```bash
curl -X POST "http://[node-ip]:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer [your-api-key]" \
  -d '{"model":"UI-TARS-7B-DPO", "prompt":"Hello, world", "max_tokens":100}'
```

### Multimodal Requests

To send an image along with text:

```bash
python examples/query_model.py --host [node-ip] --port 8000 --api-key [your-api-key] --prompt "Describe this image" --image path/to/image.jpg
```

## Security Recommendations

1. Do not share API keys with unauthorized users
2. Consider setting up HTTPS for production deployments
3. Restrict CORS origins in config.py for production use
4. Use a firewall to limit access to the server port

## Performance Monitoring

To monitor server performance:
```bash
python examples/monitor_performance.py --host [node-ip] --port 8000 --api-key [your-api-key]
```

## Configuration

Edit `config.py` to customize deployment options and further optimize performance.
