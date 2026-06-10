# Transfer Instructions: Local Development → GPU PC

## Project Transfer Checklist

### Phase 1: Code Preparation

Before transferring, ensure all code is committed and tested:

```bash
# 1. Run all tests to ensure no failures
python -m pytest tests/ -v --tb=short

# 2. Validate all Python files
python -m py_compile $(find . -name "*.py" -type f)

# 3. Check for any uncommitted changes
git status

# 4. Create a clean export
git archive --format=tar.gz -o soc_system_phase3.tar.gz HEAD
# OR without git:
tar -czf soc_system_phase3.tar.gz \
  --exclude=.venv \
  --exclude=.venv-1 \
  --exclude=__pycache__ \
  --exclude=.pytest_cache \
  --exclude=detection/models/*.json \
  --exclude=detection/models/*.txt \
  --exclude=docker-compose.yml \
  .
```

### Phase 2: Pre-Transfer Checklist

**Files to verify exist:**
```
detection/
  ├── __init__.py
  ├── rule_engine.py
  ├── ml_classifier.py
  ├── anomaly_detector.py
  ├── aggregator.py
  ├── sigma_rules/
  │   ├── ssh_brute_force.yml
  │   ├── port_scan.yml
  │   ├── privilege_escalation.yml
  │   ├── web_shell_upload.yml
  │   └── lateral_movement.yml
  └── models/           (created on first training)

tests/
  ├── test_detection.py (Phase 3 tests)
  ├── test_enrichment.py
  └── test_wazuh_bridge.py

backend/schemas/
  └── alert.py

pipeline/
  ├── kafka_producer.py
  ├── kafka_consumer.py
  └── enrichment/
      ├── geoip.py
      ├── threat_intel.py
      ├── mitre_mapper.py
      └── sessionizer.py

.env.example        (includes KAFKA_BROKERS, REDIS_HOST, ABUSEIPDB_API_KEY)
requirements.txt    (includes Phase 3 dependencies)
GPU_SETUP.md        (this guide)
PHASE2.md
README.md
```

**Files NOT to transfer:**
```
.venv/              (CPU virtual environment)
.venv-1/            (alternate environment)
__pycache__/        (compiled Python cache)
.pytest_cache/      (pytest cache)
docker-compose.yml  (use docker-compose.kafka.yml instead)
*.json (models)     (retrain on GPU)
*.txt (models)      (retrain on GPU)
```

### Phase 3: Transfer Methods

#### Option 1: USB/External Drive
```bash
# On source machine (local PC):
tar -czf soc_system_phase3.tar.gz \
  --exclude=.venv --exclude=.venv-1 --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude=detection/models \
  /path/to/Doom_deez_attacks

# Copy to USB drive
# Transfer to GPU PC
# Extract on GPU PC:
tar -xzf soc_system_phase3.tar.gz
```

#### Option 2: Git Repository
```bash
# On source machine:
git init
git add .
git commit -m "Phase 3 initial commit"
git remote add origin <GPU_PC_REPO_URL>
git push -u origin main

# On GPU PC:
git clone <GPU_PC_REPO_URL>
cd Doom_deez_attacks
```

#### Option 3: Network Transfer (SCP/SFTP)
```bash
# On GPU PC (destination):
# Create directory
mkdir -p ~/soc_system
cd ~/soc_system

# On source machine (local PC):
scp -r --exclude=.venv --exclude=__pycache__ \
  /path/to/Doom_deez_attacks \
  gpu_user@gpu_host:~/soc_system/
```

#### Option 4: Rsync (Most Efficient)
```bash
rsync -av --delete \
  --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache \
  --exclude=detection/models \
  /path/to/Doom_deez_attacks/ \
  gpu_user@gpu_host:~/soc_system/Doom_deez_attacks/
```

### Phase 4: GPU PC Setup

**On the GPU PC, after transfer:**

```bash
# 1. Navigate to project
cd ~/soc_system/Doom_deez_attacks

# 2. Verify file structure
ls -la  # Check for detection/ pipeline/ tests/ directories

# 3. Create GPU-optimized virtual environment
python3.11 -m venv venv_gpu
source venv_gpu/bin/activate

# 4. Upgrade pip
pip install --upgrade pip setuptools wheel

# 5. Install GPU-specific dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# 6. Install requirements
pip install -r requirements.txt

# 7. Verify GPU setup
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"

# 8. Run Phase 3 detection tests
python -m pytest tests/test_detection.py -v
```

### Phase 5: Model Training on GPU

```bash
# 1. Prepare training data
python -c "
import pandas as pd
from detection.ml_classifier import MLClassifier
import numpy as np

# Generate synthetic training dataset
alerts = [...]  # Load from Kafka or synthetic
y = np.array([...])  # Labels: 0=benign, 1=suspicious, 2=malicious

# Train on GPU
classifier = MLClassifier()
results = classifier.train(pd.DataFrame(alerts), y)
print('Training results:', results)
"

# 2. Verify model performance
python -c "
from detection.ml_classifier import MLClassifier

classifier = MLClassifier()
classifier.load()  # Load trained model

# Test inference speed
import time
start = time.time()
for _ in range(100):
    classifier.predict({'rule_level': 5, 'session_event_count': 10})
elapsed = (time.time() - start) * 1000 / 100
print(f'Average inference time: {elapsed:.2f}ms')
"
```

### Phase 6: Infrastructure Setup on GPU PC

```bash
# Option A: Docker Compose (Recommended)
cd detection/sigma_rules  # Verify rules exist
cd infra
docker-compose -f docker-compose.kafka.yml up -d

# Option B: Manual Services
# Install Kafka, Zookeeper, Redis locally or use managed services

# Verify connectivity
redis-cli ping         # Should return PONG
kafka-topics --list    # Should list topics
```

### Phase 7: Run Full Detection Pipeline

```bash
# Terminal 1: Kafka Consumer (Enrichment Pipeline)
python pipeline/kafka_consumer.py

# Terminal 2: Detection Engine (new in Phase 3)
python detection/run_detector.py  # (see below for script)

# Terminal 3: Monitor Detections
python -c "
from confluent_kafka import Consumer
import json

consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'detection-monitor',
    'auto.offset.reset': 'latest'
})
consumer.subscribe(['detections'])

while True:
    msg = consumer.poll(timeout=1.0)
    if msg:
        detection = json.loads(msg.value().decode())
        print(f\"Detection: {detection['alert_id']} - {detection['attack_type']} ({detection['confidence_score']:.2%})\")
"
```

### Phase 8: Performance Validation

```bash
# Run benchmarks
python -m pytest tests/test_detection.py::test_detection_performance_under_100ms -v

# Expected results (GPU):
# CPU: 80-150ms per alert
# GPU: 5-15ms per alert

# Monitor GPU usage
nvidia-smi -l 1

# Profile detection pipeline
python -m cProfile -s cumulative detection/run_detector.py | head -20
```

### Phase 9: Environment Variables Setup

Create `.env` on GPU PC:
```bash
# Copy and customize .env.example
cp .env.example .env

# Edit with GPU-specific settings
cat > .env << EOF
# Wazuh
WAZUH_HOST=wazuh-manager
WAZUH_PORT=55000
WAZUH_USERNAME=wazuh
WAZUH_PASSWORD=<SECURE_PASSWORD>
WAZUH_VERIFY_SSL=false

# Kafka
KAFKA_BROKERS=localhost:9092

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# APIs
ABUSEIPDB_API_KEY=<API_KEY>
GEOIP_DATABASE_PATH=data/GeoLite2-City.mmdb

# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # or "0,1,2,3" for multi-GPU
GPU_BATCH_SIZE=512
GPU_WORKERS=4

# Detection
ML_MODEL_PATH=detection/models/xgb_model.json
ANOMALY_MODEL_THRESHOLD=0.05
DETECTION_BATCH_SIZE=128
EOF
```

### Phase 10: Continuous Integration

```bash
# Set up automated testing on GPU PC
crontab -e  # Add:
# Run tests every 6 hours
0 */6 * * * cd ~/soc_system/Doom_deez_attacks && python -m pytest tests/test_detection.py -v >> /var/log/soc_tests.log

# Monitor GPU memory
0 * * * * nvidia-smi >> /var/log/gpu_memory.log
```

## Troubleshooting Transfer Issues

### Issue: Import errors after transfer
```bash
Solution:
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
```

### Issue: File permissions
```bash
Solution:
chmod -R u+rwx ~/soc_system/Doom_deez_attacks
chmod +x detection/*.py
```

### Issue: CUDA version mismatch
```bash
Solution:
# Check CUDA version
nvcc --version

# Reinstall torch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
```

### Issue: Kafka not connecting
```bash
Solution:
# Update KAFKA_BROKERS in .env
# For local: KAFKA_BROKERS=localhost:9092
# For Docker: KAFKA_BROKERS=kafka:9092
# For remote: KAFKA_BROKERS=gpu_host:9092
```

## Post-Transfer Validation

```bash
#!/bin/bash
# Validate GPU SOC system

echo "=== File Structure ==="
ls -la detection/ pipeline/ tests/ backend/

echo "=== Python Version ==="
python --version

echo "=== GPU Availability ==="
nvidia-smi

echo "=== Python Imports ==="
python -c "
import torch, xgboost, lightgbm, shap, sklearn
print('✓ All ML libraries available')
"

echo "=== Detection Engine Tests ==="
python -m pytest tests/test_detection.py -v

echo "=== GPU Performance Test ==="
python -c "
from detection.ml_classifier import MLClassifier
import time

classifier = MLClassifier()
# Train or load model
# Measure inference time on GPU
"

echo "=== Transfer Complete ==="
```

## Support & Monitoring

- GPU memory usage: `nvidia-smi`
- Inference latency: `pytorch/torch.cuda.Event` for profiling
- Model accuracy: `detection/ml_classifier.py` test suite
- Alert pipeline: Monitor Kafka consumer lag

For issues, check:
1. CUDA drivers: `nvidia-smi`
2. PyTorch GPU: `python -c "import torch; torch.cuda.is_available()"`
3. Model training logs: `detection/models/training.log`
4. Kafka connectivity: `kafka-topics --list`
