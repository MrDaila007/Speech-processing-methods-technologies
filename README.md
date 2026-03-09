# Speech Processing Methods & Technologies

Lab assignments for the "Speech Processing Methods & Technologies" course (Master's program).

## Labs

| # | Topic | Description |
|---|-------|-------------|
| [Lab1](Lab1/) | Discretization | Audio signal conversion to a target sample rate with frequency quantization |
| [Lab2](Lab2/) | K-Means Algorithm | Plane point clustering with classification quality estimation |

## Implementation Languages

Each lab is implemented in three languages:

- **C++** (C++11)
- **Python** (3.8+)
- **R** (4.0+)

## Project Structure

```
├── Lab1/
│   ├── task.md           # Task description
│   ├── README.md         # Overview & results
│   ├── Makefile
│   ├── cpp/
│   ├── python/
│   └── r/
├── Lab2/
│   ├── task.md
│   ├── README.md
│   ├── Makefile
│   ├── cpp/
│   ├── python/
│   └── r/
└── README.md
```

## Quick Start

```bash
# Lab1 — build and test
cd Lab1 && make test

# Lab2 — build and test
cd Lab2 && make test
```

### Available Make Commands

| Command | Action |
|---------|--------|
| `make` | Build C++ and Python |
| `make test` | Run all tests (C++ + Python + R) |
| `make test-cpp` | C++ tests only |
| `make test-python` | Python tests only |
| `make test-r` | R tests only |
| `make run-cpp` | Run C++ solution |
| `make run-python` | Run Python solution |
| `make run-r` | Run R solution |
| `make clean` | Remove build artifacts |

## Requirements

- `g++` with C++11 support
- `python3` (3.8+)
- `Rscript` (4.0+)
