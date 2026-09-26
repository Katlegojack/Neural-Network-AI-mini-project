# MNIST Continual-Learning Results

Reference run using the current notebook configuration.

## Experimental setup

- Dataset: MNIST
- Tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
- Model: 784 -> 256 -> 128 -> 10 MLP with ReLU
- Batch size: 128
- Epochs per task: 5
- Optimizer: Adam
- Learning rate: 0.001
- Random seed: 42
- EWC lambda: 1000
- Fisher batches: 50
- Replay buffer size: 1000
- Replay batch size: 64

## Final measurements

| Method | Final average accuracy | Average forgetting |
|---|---:|---:|
| Naive Sequential | 19.71% | 99.41 percentage points |
| EWC | 19.77% | 99.53 percentage points |
| Experience Replay | 90.25% | 10.62 percentage points |

In this run, Naive Sequential Training and the current EWC configuration both lost almost all accuracy on older tasks as new digit pairs were learned. Experience Replay retained substantially more old-task performance.

These values are experimental results from this specific configuration, not assumed outcomes. Re-running with different seeds or hyperparameters may produce different exact values.
