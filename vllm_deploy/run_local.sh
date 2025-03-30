#!/bin/bash

# Script for local deployment of UI-TARS-7B-DPO using vLLM

# Source configuration
source config.py 2>/dev/null || echo "Using default configuration"

# Set performance-optimized environment variables
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1  # Improve performance for single-GPU setup
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Optimize CUDA connections
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  # Optimize memory allocator
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism for better stability

# Clear GPU cache if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "Clearing GPU cache before deployment..."
    nvidia-smi --gpu-reset
fi

# Enable TF32 if supported
python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True" 2>/dev/null || echo "TF32 optimization not available"

# Run the deployment script with optimized parameters
echo "Starting high-performance UI-TARS-7B-DP deployment with vLLM..."
python deploy.py \
  --tensor-parallel-size ${LOCAL_CONFIG[num_gpus]:-1} \
  --api-style openai \
  --quantization awq \
  --paged-attention

echo "Deployment complete. Server running at ${SERVER_CONFIG[host]:-0.0.0.0}:${SERVER_CONFIG[port]:-8000}"
