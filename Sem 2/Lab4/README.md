# Lab 4 — Multilayer Perceptron

## Task

1. Take a well-known dataset (Fisher's Iris).
2. Build a multilayer perceptron (MLP) with one hidden layer and softmax activation on the output layer.
3. Train using the ADAM optimizer.
4. Compare generalization for 4 hidden-layer activation functions: ReLU, Sigmoid, Bipolar Sigmoid, GELU.

## Algorithm

### MLP Architecture

```
Input (4) ──→ Hidden (16, activation) ──→ Output (3, softmax)
```

- **Dataset**: Iris — 150 samples, 4 features, 3 classes (embedded, no external files)
- **Split**: 80% train (120) / 20% test (30), shuffled with seed
- **Normalization**: Z-score (mean=0, std=1) computed on training set
- **Weight init**: Xavier/Glorot uniform
- **Loss**: Cross-entropy

### ADAM Optimizer

```
m = β₁·m + (1−β₁)·g
v = β₂·v + (1−β₂)·g²
m̂ = m / (1−β₁ᵗ)
v̂ = v / (1−β₂ᵗ)
θ = θ − lr · m̂ / (√v̂ + ε)
```

Parameters: lr=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸, epochs=500

### Activation Functions

| Function | f(z) | f'(z) |
|----------|-------|-------|
| ReLU | max(0, z) | 1 if z > 0, else 0 |
| Sigmoid | 1/(1+e⁻ᶻ) | σ(z)·(1−σ(z)) |
| Bipolar Sigmoid | 2σ(z)−1 | 2σ(z)(1−σ(z)) |
| GELU | 0.5z(1+tanh(√(2/π)(z+0.044715z³))) | see code |

## Structure

```
Lab4/
├── Makefile
├── README.md
├── task.md
├── cpp/
│   ├── mlp.h            # Header-only core logic
│   ├── solution.cpp     # Entry point
│   └── tests.cpp        # 121 assertions
└── python/
    ├── solution.py      # Complete solution
    └── test_solution.py # 49 test methods
```

## Usage

```bash
make              # Build C++ and set up Python venv
make test         # Run all tests (C++ + Python)
make test-cpp     # Run C++ tests only
make test-python  # Run Python tests only
make run-cpp      # Run C++ solution
make run-python   # Run Python solution
make clean        # Remove build artifacts
```

## Results

### Comparison Table (Python, seed=42)

```
Датасет: Iris (150 образцов, 4 признака, 3 класса)
Скрытый слой: 16 нейронов
Оптимизатор: ADAM (lr=0.001, beta1=0.9, beta2=0.999)
Эпохи: 500

+-------------------+------------------+-----------------+----------------+
| Функция активации | Точность (обуч.) | Точность (тест) | Потери (обуч.) |
+-------------------+------------------+-----------------+----------------+
| ReLU              |          98.33%  |         93.33%  |        0.0314  |
| Sigmoid           |          98.33%  |         96.67%  |        0.0428  |
| Bipolar Sigmoid   |          98.33%  |         96.67%  |        0.0428  |
| GELU              |          98.33%  |         96.67%  |        0.0423  |
+-------------------+------------------+-----------------+----------------+
```

All four activation functions achieve >93% test accuracy on the Iris dataset. Sigmoid, Bipolar Sigmoid, and GELU show slightly better generalization (96.67% test) compared to ReLU (93.33%), while ReLU achieves the lowest training loss.

### Tests

| Language | Tests | Status |
|----------|-------|--------|
| C++ | 121 | Pass |
| Python | 49 | Pass |
