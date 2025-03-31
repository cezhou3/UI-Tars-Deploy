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

# Create a submission script with advanced high-performance optimizations
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
#SBATCH --constraint=a100
#SBATCH --output=vllm_uitars_%j.out
#SBATCH --error=vllm_uitars_%j.err

source ~/miniconda3/bin/activate agentS

echo "Job started at \$(date)"
echo "Running on \$(hostname)"
echo "Available GPUs: \$(nvidia-smi -L)"

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

# Optimize system for inference
echo "Setting system optimizations..."
echo never > /sys/kernel/mm/transparent_hugepage/enabled || echo "Could not set transparent hugepages"
echo 1 > /proc/sys/vm/overcommit_memory || echo "Could not set overcommit_memory"
sysctl -w vm.swappiness=1 || echo "Could not set swappiness"

# Enable TF32 if supported
python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True" 2>/dev/null

# Run the deployment with advanced options
python deploy.py \
  --tensor-parallel-size ${TP_SIZE} \
  --api-style openai \
  --quantization ${QUANTIZATION} \
  --paged-attention

echo "Job completed at \$(date)"
EOF

# Submit the job
echo "Submitting UI-TARS-7B-DPO deployment job to SLURM..."
sbatch slurm_submit.sh
echo "Job submitted. Check status with 'squeue -u \$USER'"
