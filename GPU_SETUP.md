# Phase 3: GPU Setup Guide for AI-Powered SOC System

## Prerequisites

- NVIDIA GPU (RTX 3060+ or A100/H100 for production)
- CUDA 12.0+ installed
- cuDNN 8.9+ installed
- Ubuntu 22.04 LTS (recommended) or Windows Server 2022

## Step 1: Install NVIDIA CUDA Toolkit

### Linux (Ubuntu)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-0
sudo apt-get -y install nvidia-cuda-toolkit
```

### Windows
Download and install from: https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvidia-smi
```

## Step 2: Create Python Virtual Environment with GPU Support

```bash
# Create venv for GPU environment
python -m venv venv_gpu
source venv_gpu/bin/activate  # Linux/Mac
# OR
venv_gpu\Scripts\activate  # Windows
```

## Step 3: Install ML/AI Dependencies with GPU Support

```bash
pip install --upgrade pip setuptools wheel

# Core ML libraries with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
pip install xgboost[gpu] --upgrade
pip install lightgbm --upgrade
pip install rapids-singlenode --upgrade

# CPU fallbacks for compatibility
pip install scikit-learn pandas shap imbalanced-learn

# Other requirements
pip install -r requirements.txt
```

## Step 4: Verify GPU Availability

```bash
python -c "
import torch
import xgboost as xgb
import lightgbm as lgb

print('PyTorch GPU available:', torch.cuda.is_available())
print('XGBoost GPU support:', xgb.__version__)
print('LightGBM GPU support:', lgb.__version__)
"
```

## Step 5: Configure Models for GPU Execution

Update [detection/ml_classifier.py](detection/ml_classifier.py) to enable GPU:

```python
# In MLClassifier.__init__():
self.xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='gpu_hist',        # GPU histogram
    gpu_id=0,                        # Primary GPU
)

self.lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    device='gpu',                    # GPU device
    gpu_platform_id=0,
    gpu_device_id=0,
)
```

## Step 6: Optimize for Batch Processing

For high-throughput inference, use batching:

```python
# In detection pipeline
import numpy as np

def batch_predict(classifier, alerts: list[dict], batch_size: int = 128):
    """Batch prediction for GPU efficiency."""
    results = []
    for i in range(0, len(alerts), batch_size):
        batch = alerts[i:i+batch_size]
        batch_results = [classifier.predict(alert) for alert in batch]
        results.extend(batch_results)
    return results
```

## Step 7: Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# In Python
import psutil
print('GPU Memory:', psutil.virtual_memory().percent)
```

## Performance Benchmarks

Expected inference times (per alert):
- CPU (Intel i7): 80-150ms
- GPU (RTX 3080): 5-15ms
- GPU (H100): 1-3ms

Expected throughput:
- CPU: ~10 alerts/sec
- GPU (RTX): ~50-100 alerts/sec
- GPU (H100): ~200-500 alerts/sec

## Multi-GPU Setup (Optional)

For distributed detection:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Use GPUs 0-3

# Split work across GPUs
from torch.nn.parallel import DataParallel
model = DataParallel(model, device_ids=[0,1,2,3])
```

## Common Issues and Fixes

### Issue: CUDA out of memory
```
Solution: Reduce batch size or use gradient checkpointing
ml_classifier = MLClassifier()
# Add to fit(): gc.collect() between batches
```

### Issue: GPU not detected
```
Solution: Check driver
nvidia-smi
# Update driver if needed
```

### Issue: XGBoost tree_method not supported
```
Solution: Verify CUDA compatibility
python -c "import xgboost; print(xgboost.__version__)"
# Reinstall with: pip install xgboost[gpu] --upgrade
```

## Production Deployment

### Using Docker with GPU
```dockerfile
FROM nvidia/cuda:12.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "pipeline/kafka_consumer.py"]
```

Build and run:
```bash
docker build -t soc-gpu .
docker run --gpus all -e KAFKA_BROKERS=kafka:9092 soc-gpu
```

### Performance Tuning
- Use mixed precision (float16) for 2-3x speedup: `ml_classifier.enable_amp = True`
- Enable TensorRT for inference optimization (for NVIDIA GPUs)
- Profile with `nvidia-smi` and `nsys profile`

## Next Steps

After GPU setup complete:
1. Run `python -m pytest tests/test_detection.py -v` to validate
2. Train models with `python detection/ml_classifier.py --train`
3. Monitor performance with GPU telemetry
4. Deploy detection pipeline to GPU cluster

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/)
- [LightGBM GPU Support](https://lightgbm.readthedocs.io/en/latest/GPU-Tuning.html)
- [PyTorch GPU Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
