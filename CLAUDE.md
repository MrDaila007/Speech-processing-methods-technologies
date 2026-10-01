# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's-level lab assignments for a "Speech Processing Methods & Technologies" course. Each lab implements the same algorithm in two languages (C++11, Python 3) using only standard libraries — no external dependencies.

- **Lab1 (Discretization)**: Converts audio signals to a target sample rate with frequency quantization (bitrate 1–24, sample rates: 48/96/128/176/256 KHz, frequency range 20–20,000 Hz)
- **Lab2 (K-Means)**: Clusters 2D points and computes misclassification error rate via majority voting
- **Lab3 (Hidden Markov Model)**: Initializes a random HMM for 27 observation symbols (a-z + space), computes phrase probabilities via the Forward algorithm with Rabiner scaling

## Build & Test Commands

All commands run from the respective `Lab{N}/` directory using GNU Make:

```bash
make              # Build C++ and set up Python venv
make test         # Run all tests (C++ + Python)
make test-cpp     # Run C++ tests only
make test-python  # Run Python tests only
make run-cpp      # Run C++ solution
make run-python   # Run Python solution
make clean        # Remove build artifacts
```

C++ compiles with `g++ -O2`. Python tests use `unittest` (`python3 -m unittest test_solution -v`).

## Architecture

Each lab follows an identical structure across both languages:

| Role | C++ | Python |
|------|-----|--------|
| Core logic | `cpp/{discretization,kmeans,hmm,mlp}.h` (header-only) | `python/solution.py` |
| Entry point | `cpp/solution.cpp` | `python/solution.py` |
| Tests | `cpp/tests.cpp` | `python/test_solution.py` |

**Key patterns across all languages:**
- Core logic is separated from I/O and testing
- Input validation throws early with descriptive messages (`std::invalid_argument` / `ValueError`)
- Deterministic random generation via explicit seeds (`std::mt19937` / `random.Random(seed)`)
- C++ tests use custom assertion macros (`ASSERT_TRUE`, `ASSERT_NEAR`, `ASSERT_THROWS`); Python uses `unittest.TestCase`

## Documentation

Task descriptions (`task.md`) are bilingual (English + Russian). READMEs contain algorithm overviews, usage instructions, and expected output examples.
