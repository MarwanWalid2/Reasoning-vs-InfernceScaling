# Reasoning-vs-InfernceScaling
This repository contains code to test reasoning capabilities using the Quiet Star model and inference scaling laws with Rebase. Experiments were conducted using:
- 1 A100 GPU
- 8 CPUs

# To install required libraries
```
pip install -r requirements.txt
```

# Configuration Settings

## General Configuration
| Parameter | Value |
|-----------|-------|
| Temperature | 1.0 |
| Max Tokens | 384 |
| Random Seed | 42 |

## REBASE Configuration
| Parameter | Value |
|-----------|-------|
| Softmax Temperature | 0.2 |
| Tree Width Options | 3, 6, 16 |

Note: Random seed 42 was maintained from the original QuietSTaR implementation to ensure comparable results.

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

# Results folder
The results contain results for basline Mistral7b model, quietstar, rebase with 3,6,and 16 width tree

The evaluation was performed on 128 questions from the GSM8K dataset. 

For rebase we conducted different experiments with varying width sizes of the Rebase tree: 3, 6, and 16 (16 was the original width size from the paper).


# Efficiency Metrics & Rankings

## Equations

1. Accuracy per TFLOP = Accuracy (%) / FLOPs (Trillion)
2. Accuracy per Second = Accuracy (%) / Time (seconds)
3. Efficiency Score = (Accuracy per TFLOP × Accuracy per Second) × 100

## Results

| Model | Accuracy (%) | FLOPs (T) | Time (s) | Acc/TFLOP | Acc/Second | Efficiency Score |
|-------|-------------|-----------|-----------|------------|------------|------------------|
| Baseline | 10.16 | 11.22 | 52.47 | 0.91 | 0.19 | 17.29 |  
| QuietSTaR | 32.03 | 12.73 | 554.66 | 2.52 | 0.06 | 15.12 |
| REBASE+QuietSTaR | 9.38 | 4.25 | 143.66 | 2.21 | 0.07 | 15.47 |
| REBASE (w=3) | 10.94 | 2.35 | 8.47 | 4.66 | 1.29 | 601.14 |
| REBASE (w=6) | 10.16 | 4.96 | 17.82 | 2.05 | 0.57 | 116.85 |
| REBASE (w=16) | 12.50 | 13.57 | 46.90 | 0.92 | 0.27 | 24.84 |

## Example Calculation (REBASE w=3)

1. Accuracy per TFLOP = 10.94% / 2.35T = 4.66
2. Accuracy per Second = 10.94% / 8.47s = 1.29
3. Efficiency Score = (4.66 × 1.29) × 100 = 601.14



