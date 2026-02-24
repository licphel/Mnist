# Mnist

**A minimal MNIST classifier with manual training loop implementation using TensorFlow.** 

Built to understand the fundamentals of deep learning training pipelines. (The Mnist dataset is partially taken for faster training, which is the cause of relatively low accurary.)

## Features

- **Manual training loop**
- **Custom loss functions** (L1, L2, Exponential, Cross-entropy)
- **Clean modular design**: data, model, training, visualization

## Model Architecture

```
Input (28x28x1)
    ↓
Conv2D(32) + ReLU
    ↓
MaxPooling(2x2)
    ↓
Conv2D(64) + ReLU
    ↓
MaxPooling(2x2)
    ↓
Conv2D(64) + ReLU
    ↓
Flatten
    ↓
Dense(64) + ReLU
    ↓
Dense(10) + Softmax
```

## Aims

Built to understand:
- Gradient computation & backpropagation
- Manual batch processing
- Training/validation loops
- Custom loss functions