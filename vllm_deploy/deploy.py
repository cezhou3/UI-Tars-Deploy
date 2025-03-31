"""
Main deployment script for UI-TARS-7B-DPO using vLLM
"""

import argparse
import os
import psutil
import torch
import json
import socket
from typing import List, Dict, Optional, Union

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import serve_openai_api
from vllm.entrypoints.api_server import serve_api
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.middleware.cors import CORSMiddleware

from config import MODEL_CONFIG, SERVER_CONFIG, LOCAL_CONFIG, API_KEY_PATH, generate_api_key


# API key security
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def optimize_system():
    """Apply system-level optimizations."""
    # Set process priority higher
    try:
        import os
        os.nice(-10)  # Higher priority (Unix only)
    except:
        pass
    
    # Optimize CPU affinity if possible
    try:
        process = psutil.Process()
        cores = list(range(psutil.cpu_count()))
        process.cpu_affinity(cores)
    except:
        pass


def get_host_ip() -> str:
    """Get the IP address of the host machine."""
    try:
        # Get the hostname first
        hostname = socket.gethostname()
        # Get the IP address for the hostname
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        print(f"Error getting host IP: {e}")
        return "127.0.0.1"  # Default to localhost


def setup_api_keys(app: FastAPI, api_keys: List[str]) -> None:
    """Set up API key authentication."""
    if not api_keys:
        # Generate a default API key if none provided
        default_key = generate_api_key()
        api_keys.append(default_key)
        # Save the API key to a file
        with open(API_KEY_PATH, "w") as f:
            f.write(f"Default API Key: {default_key}\n")
        print(f"Generated default API key and saved to {API_KEY_PATH}")
    
    # Function to validate API key
    async def get_api_key(api_key: str = Security(api_key_header)) -> str:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is missing",
            )
        
        # Strip "Bearer " prefix if present
        if api_key.startswith("Bearer "):
            api_key = api_key[7:]
            
        if api_key not in api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        return api_key
    
    # Add the dependency to the app
    app.dependency_overrides[APIKey] = get_api_key
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SERVER_CONFIG.get("cors_allow_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deploy UI-TARS-7B-DPO using vLLM")
    parser.add_argument(
        "--model", type=str, default=MODEL_CONFIG["model_name"], help="Model name or path"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=MODEL_CONFIG["tensor_parallel_size"],
        help="Tensor parallelism degree",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=MODEL_CONFIG["dtype"],
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=MODEL_CONFIG.get("quantization", "none"),
        choices=["none", "awq", "squeezellm", "gptq"],
        help="Quantization method",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MODEL_CONFIG["max_model_len"],
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--host", type=str, default=SERVER_CONFIG["host"], help="Host to bind"
    )
    parser.add_argument(
        "--port", type=int, default=SERVER_CONFIG["port"], help="Port to bind"
    )
    parser.add_argument(
        "--api-style",
        type=str,
        default="openai",
        choices=["vllm", "openai"],
        help="API style (vLLM or OpenAI compatible)",
    )
    parser.add_argument(
        "--paged-attention",
        action="store_true",
        default=LOCAL_CONFIG.get("paged_attention", True),
        help="Enable paged attention for memory efficiency",
    )
    parser.add_argument(
        "--api-keys", 
        type=str, 
        default=None,
        help="Comma-separated list of API keys for authentication",
    )
    parser.add_argument(
        "--enable-auth",
        action="store_true",
        default=SERVER_CONFIG.get("enable_auth", True),
        help="Enable API key authentication",
    )
    parser.add_argument(
        "--public-hostname",
        type=str,
        default=SERVER_CONFIG.get("public_hostname", ""),
        help="Public hostname for the server",
    )
    return parser.parse_args()


def main():
    """Main function to deploy the model with high performance."""
    # Apply system optimizations
    optimize_system()
    
    args = parse_args()
    
    # Print deployment information
    print(f"Deploying {args.model} with ultra high-performance vLLM configuration")
    print(f"Tensor parallelism: {args.tensor_parallel_size}")
    print(f"Data type: {args.dtype}, Quantization: {args.quantization}")
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    print(f"Memory utilization target: {MODEL_CONFIG.get('gpu_memory_utilization', 0.97)*100}%")
    
    # Get host IP if public hostname is not specified
    public_hostname = args.public_hostname
    if not public_hostname:
        public_hostname = get_host_ip()
        print(f"Using detected IP address: {public_hostname}")
    
    # Set up API keys
    api_keys = []
    if args.enable_auth:
        if args.api_keys:
            api_keys = args.api_keys.split(",")
        else:
            api_keys = SERVER_CONFIG.get("api_keys", [])
    
    # Prepare engine arguments with advanced high-performance settings
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=MODEL_CONFIG["trust_remote_code"],
        gpu_memory_utilization=MODEL_CONFIG.get("gpu_memory_utilization", 0.97),
        enforce_eager=MODEL_CONFIG.get("enforce_eager", False),
        quantization=args.quantization,
        
        # Advanced high-performance settings
        swap_space=8,  # Use 8GB of swap space
        max_num_seqs=LOCAL_CONFIG.get("concurrent_requests", 32),
        disable_log_stats=True,  # Disable logging for better performance
        disable_log_requests=True,  # Disable request logging for performance
        enable_chunked_prefill=True,  # Enable chunked prefills
        kv_cache_dtype=MODEL_CONFIG.get("kv_cache_dtype", "auto"),
        block_size=MODEL_CONFIG.get("block_size", 16),
        gpu_block_size=MODEL_CONFIG.get("block_size", 16),
        enable_delayed_outputs=True,
        paged_attention=args.paged_attention,
    )
    
    # Set CUDA visible devices if running in a distributed setting
    if "CUDA_VISIBLE_DEVICES" not in os.environ and args.tensor_parallel_size > 1:
        gpu_ids = list(range(args.tensor_parallel_size))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        print(f"Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Print connection information
    print(f"\nServer will be accessible at: http://{public_hostname}:{args.port}")
    if args.enable_auth:
        print(f"API authentication is ENABLED. API keys stored in: {API_KEY_PATH}")
    else:
        print("WARNING: API authentication is DISABLED. This is not recommended for production.")
    
    # Start the server with optimized settings
    if args.api_style == "openai":
        app = FastAPI(title="vLLM API Server")
        if args.enable_auth:
            setup_api_keys(app, api_keys)
        
        serve_openai_api(
            engine_args=engine_args,
            served_model=args.model,
            host=args.host,
            port=args.port,
            max_num_batched_tokens=SERVER_CONFIG["max_num_batched_tokens"],
            max_batch_size=SERVER_CONFIG.get("max_batch_size", 64),
            max_wait_time=SERVER_CONFIG.get("max_wait_time", 0.05),
            app=app,  # Pass our modified FastAPI app
        )
    else:
        serve_api(
            engine_args=engine_args,
            host=args.host,
            port=args.port,
            max_num_batched_tokens=SERVER_CONFIG["max_num_batched_tokens"],
            max_batch_size=SERVER_CONFIG.get("max_batch_size", 64),
            max_wait_time=SERVER_CONFIG.get("max_wait_time", 0.05),
        )


if __name__ == "__main__":
    main()
