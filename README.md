# Reasoning-vs-InfernceScaling
This repository contains code to test reasoning capabilities using the Quiet Star model and inference scaling laws with Rebase. Experiments were conducted using:
- 1 A100 GPU
- 8 CPUs

The evaluation was performed on 128 questions from the GSM8K dataset. 

For rebase we conducted different experiments with varying width sizes of the Rebase tree: 3, 6, and 16 (16 was the original width size from the paper).

## Repository Contents
The repository includes:

- `base.py`: Evaluates the base Mistral-7B model for:
    - Accuracy
    - Inference time
    - FLOPS per inference
- `rebase.py`: Same metrics using Rebase for inference optimization
    - Note: To run rebase.py, modify the hyperparameters in rebase.yaml first
- `quietstar eval/quietstar.py`: Same metrics for Quiet Star model
    - Note: To run quietstar you need to have the custom modeling_mistral and config files in the same directory as the evaluation script
- `quietstar eval/quiet-rebase.py`: Same metrics using rebase inference with quiet-star model
    - Note: To run quietstar you need to have the custom modeling_mistral and config files in the same directory as the evaluation script

# Results
The results contain results for basline Mistral7b model, quietstar, rebase with 3,6,and 16 width tree

# To install required libraries
```
pip install -r requirements.txt
```


