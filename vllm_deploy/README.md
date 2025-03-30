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

### Example Query (Text only)

```bash
python examples/query_model.py --prompt "Explain quantum computing in simple terms."
```

### Example Query (Image + Text)

```bash
python examples/query_model.py --prompt "Describe this image in detail." --image path/to/your/image.jpg
```

## Performance Monitoring

To monitor server performance:
```bash
python examples/monitor_performance.py
```

## Configuration

Edit `config.py` to customize deployment options and further optimize performance.
