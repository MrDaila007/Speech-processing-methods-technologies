# Lab 1 — Discretization

## Task

Given **n** seconds of audio with **10⁶·n** signal frequency values in the range **20 Hz — 20,000 Hz**.

The goal is to:
1. Convert the original values to match a target **sample rate** (48, 96, 128, 176, 256 KHz).
2. Average multiple values falling into the same time interval (**arithmetic mean**).
3. Round the result to the nearest **quantization level** — evenly spaced within 20–20,000 Hz, determined by bitrate: `step = (20000 - 20) / (2^bitrate - 1)`.

### Input Parameters

| Parameter | Description |
|-----------|-------------|
| `n` | Number of audio seconds |
| Bitrate | Quantization depth (bits), determines the number of levels |
| Sample rate | 48, 96, 128, 176, or 256 KHz |

Input data is randomly generated in the 20–20,000 Hz range.

## Structure

```
Lab1/
├── task.md              # Task description
├── Makefile             # Build, run, test
├── cpp/
│   ├── discretization.h # Core logic
│   ├── solution.cpp     # Entry point
│   └── tests.cpp        # Tests
├── python/
│   ├── solution.py      # Solution
│   └── test_solution.py # Tests
└── r/
    ├── solution.R       # Solution
    └── tests.R          # Tests
```

## Usage

```bash
make run-cpp       # C++
make run-python    # Python
make run-r         # R
make test          # All tests
```

## Results

### Example Output (n=1, bitrate=8, 48 KHz)

```
=== Discretization Results ===
Sample rate:      48 KHz
Bitrate:          8 bit
Quant. levels:    256
Quant. step:      78.3529 Hz
Total samples:    48000

First 20 samples:
    Sample    Frequency (Hz)
--------------------------
         0       9814.1176
         1       8873.8824
         2      10205.8824
         3      11616.2353
         4       9657.4118
         5      10440.9412
         6      10205.8824
         7       8638.8235
         8       8795.5294
         9      10049.1765
        10       6523.2941
        11      11067.7647
        12       7306.8235
        13      10049.1765
        14       9265.6471
        15      11459.5294
        16       9892.4706
        17      10127.5294
        18       9030.5882
        19      10440.9412

Statistics:
  Min: 4564.4706 Hz
  Max: 15063.7647 Hz
  Avg: 10019.1461 Hz
```

The average value of ~10,010 Hz is close to the expected mean of a uniform distribution on [20, 20000], confirming correctness.

### Tests

| Language | Tests | Status |
|----------|-------|--------|
| C++ | 49 | Passed |
| Python | 32 | Passed |
| R | 36 | Passed |

Test coverage:
- **Unit**: `quantize`, `compute_step`, `generate_signal`, `validate_params`
- **Integration**: `discretize` — output sample count, value range, quantization levels, averaging, multi-second signal
- **Edge cases**: 1-bit quantization (2 levels), 8-bit vs 16-bit precision comparison
