"""
Configuration for UI-TARS-7B-DPO deployment
"""

# Model configuration
MODEL_CONFIG = {
    "model_name": "bytedance-research/UI-TARS-7B-DPO",  # Model name or path
    "tensor_parallel_size": 1,  # Number of GPUs to use for tensor parallelism
    "dtype": "bfloat16",  # Data type for model weights
    "max_model_len": 8192,  # Maximum sequence length
    "trust_remote_code": True,  # Allow loading of remote code
    "gpu_memory_utilization": 0.97,  # Higher utilization for better performance
    "enforce_eager": False,  # Disable eager execution for better performance
    "quantization": "awq",  # Use AWQ quantization for memory efficiency
    "kv_cache_dtype": "auto",  # Optimize KV cache data type
    "block_size": 16,  # Optimize block size for better memory efficiency
}

# Server configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_num_batched_tokens": 16384,  # Further increased for better throughput
    "max_batch_size": 64,  # Increased for better throughput
    "max_wait_time": 0.05,  # Reduced wait time for faster response
    "distributed_setup": False,  # Set to True for multi-node deployment
}

# Local deployment configuration
LOCAL_CONFIG = {
    "num_gpus": 1,  # Number of GPUs to use for local deployment
    "paged_attention": True,  # Enable paged attention for memory efficiency
    "concurrent_requests": 32,  # Handle more concurrent requests
}

# Slurm configuration
SLURM_CONFIG = {
    "partition": "gpu",  # Slurm partition to use
    "job_name": "vllm-ui-tars",  # Slurm job name
    "num_nodes": 1,  # Number of nodes to use
    "gpus_per_node": 1,  # Number of GPUs per node
    "cpus_per_task": 8,  # Increased CPU allocation for better throughput
    "memory": "64G",  # Increased memory allocation
    "time": "120:00:00",  # Time limit for the job
}
