# Reasoning-vs-InfernceScaling
This repository contains code to test reasoning capabilities using the Quiet Star model and inference scaling laws with Rebase. Experiments were conducted using:
- 1 A100 GPU
- 8 CPUs

# The repository includes:
- `base.py`: Evaluates the base Mistral-7B model for:
    - Accuracy
    - Inference time
    - FLOPS per inference
- `quietstar.py`: Same metrics for Quiet Star model
- `rebase.py`: Same metrics using Rebase for inference optimization
    - Note: To run rebase.py, modify the hyperparameters in rebase.yaml first

# Install required libraries
```
pip install -r requirements.txt
```

