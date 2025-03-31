#!/bin/bash

# Script for SLURM-based deployment of UI-TARS-7B-DPO using vLLM

# Parse configuration from Python config
PARTITION=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG['partition'])")
JOB_NAME=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG['job_name'])")
NUM_NODES=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG['num_nodes'])")
GPUS_PER_NODE=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG['gpus_per_node'])")
CPUS_PER_TASK=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG['cpus_per_task'])")
MEMORY=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG['memory'])")
TIME=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG['time'])")
TP_SIZE=$(python -c "from config import MODEL_CONFIG; print(MODEL_CONFIG['tensor_parallel_size'])")
NETWORK_INTERFACE=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG.get('network_interface', 'eth0'))")
PORT=$(python -c "from config import SLURM_CONFIG; print(SLURM_CONFIG.get('port', 8000))")

# Generate a random API key for this deployment
API_KEY=$(python -c "from config import generate_api_key; print(generate_api_key())")
echo "Generated API Key: ${API_KEY}"
echo "This API key will be required to access the server"
echo "Keep this key secure and share it only with authorized users"

# Create a submission script with advanced network configuration
cat > slurm_submit.sh << EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=vllm_server_%j.out
#SBATCH --error=vllm_server_%j.err

echo "Job started at \$(date)"
echo "Running on \$(hostname)"
echo "Available GPUs: \$(nvidia-smi -L)"

# Get the node's IP address for external access
NODE_IP=\$(hostname -I | awk '{print \$1}')
echo "Server will be accessible at: http://\${NODE_IP}:${PORT}"

# Save connection info to a file for reference
cat > server_connection_info.txt << EOT
Server URL: http://\${NODE_IP}:${PORT}
API Key: ${API_KEY}
Example curl command:
curl -X POST "http://\${NODE_IP}:${PORT}/v1/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${API_KEY}" \\
  -d '{"model":"UI-TARS-7B-DPO", "prompt":"Hello, world", "max_tokens":100}'
EOT

echo "Connection information saved to server_connection_info.txt"

# Load necessary modules
module purge
module load cuda python

# Set advanced high-performance environment variables
export CUDA_AUTO_BOOST=0  # Disable auto boost for consistent performance
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all GPUs on the node
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_DEBUG=INFO    # Debug info in case of issues
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  # Optimize memory allocator
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Optimize CUDA connections
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism
export OMP_NUM_THREADS=${CPUS_PER_TASK}  # Set optimal OpenMP threads

# Configure networking for external access
export VLLM_HOST="0.0.0.0"  # Listen on all network interfaces

# Show networking info
echo "Network information:"
ip addr show ${NETWORK_INTERFACE}
netstat -tulpn | grep LISTEN

# Enable TF32 if supported
python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True" 2>/dev/null

# Run the deployment with network configuration
python deploy.py \
  --tensor-parallel-size ${TP_SIZE} \
  --api-style openai \
  --quantization awq \
  --paged-attention \
  --host 0.0.0.0 \
  --port ${PORT} \
  --api-keys "${API_KEY}" \
  --enable-auth \
  --public-hostname "\${NODE_IP}"

echo "Job completed at \$(date)"
EOF

# Submit the job
echo "Submitting UI-TARS-7B-DPO deployment job to SLURM..."
sbatch slurm_submit.sh
echo "Job submitted. Check status with 'squeue -u \$USER'"
echo "Once the job starts, connection details will be available in server_connection_info.txt"
