# Lab 3 — Hidden Markov Model

## Task

Initialize a **Hidden Markov Model** (HMM) randomly for the English alphabet: **26 lowercase letters + space** (27 observation symbols).

Given several English phrases, compute the **probability of each phrase being produced** by the model using the **Forward algorithm** (with Rabiner scaling for numerical stability).

### HMM Parameters

| Parameter | Symbol | Size | Description |
|-----------|--------|------|-------------|
| Initial distribution | π | `n_states` | Probability of starting in each hidden state |
| Transition matrix | A | `n_states × n_states` | State-to-state transition probabilities |
| Emission matrix | B | `n_states × 27` | Observation probabilities per state |

All parameters are randomly initialized and row-normalized (each row sums to 1).

### Forward Algorithm

1. **Init**: `α₁(i) = πᵢ · bᵢ(O₁)`
2. **Induction**: `αₜ₊₁(j) = [Σᵢ αₜ(i) · aᵢⱼ] · bⱼ(Oₜ₊₁)`
3. **Scaling**: `cₜ = 1 / Σᵢ αₜ(i)` to prevent underflow
4. **Result**: `log P(O|λ) = −Σₜ log(cₜ)`

## Structure

```
Lab3/
├── task.md              # Task description
├── Makefile             # Build, run, test
├── cpp/
│   ├── hmm.h           # Core logic
│   ├── solution.cpp    # Entry point
│   └── tests.cpp       # Tests
└── python/
    ├── solution.py     # Solution
    └── test_solution.py # Tests
```

## Usage

```bash
make run-cpp       # C++
make run-python    # Python
make test          # All tests
```

## Results

### Example Output (n_states=3, seed=42)

```
=== Результаты СММ ===
Скрытых состояний:  3
Символов наблюдения: 27 (a-z + пробел)

Вероятности фраз:
                         Фраза     Длина    Лог-вероятность
------------------------------------------------------------
                 "hello world"        11            -35.7558
         "hidden markov model"        19            -62.9217
           "speech processing"        17            -54.1437
         "the quick brown fox"        19            -65.5313
```

Log-probabilities are large negative numbers because the model is randomly initialized (not trained on real text). Longer phrases yield lower log-probabilities, as expected.

### Tests

| Language | Tests | Status |
|----------|-------|--------|
| C++ | 94 | Passed |
| Python | 36 | Passed |

Test coverage:
- **Unit**: `char_to_index`, `text_to_indices`, `normalize`, `validate_params`, `init_hmm`, `forward`
- **Integration**: full pipeline with multiple phrases, many hidden states
- **Edge cases**: single state/observation, empty input, invalid characters, determinism
