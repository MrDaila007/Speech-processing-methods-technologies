# Lab 2 — K-Means Algorithm

## Task

Given **n** points on a plane belonging to **k** distinct classes.

The goal is to:
1. Split the points into **m** clusters, minimizing total distance to cluster centers.
2. For each cluster, determine its class via **majority voting**.
3. Compute the **fraction of misclassified points** — those whose class differs from their cluster's class.

### Input Parameters

| Parameter | Description |
|-----------|-------------|
| `n` | Number of points |
| `k` | Number of classes |
| `m` | Number of clusters |
| Array | `n` triples `(x, y, class)` — coordinates and class label |

### Algorithm

1. Randomly select `m` initial centroids from the input points.
2. Assign each point to the nearest centroid (Euclidean distance).
3. Recompute centroids as the mean of each cluster's points.
4. Repeat steps 2–3 until convergence (centroid shift < ε).
5. Cluster class = majority class among its points.
6. Error rate = misclassified points / total points.

## Structure

```
Lab2/
├── task.md              # Task description
├── Makefile             # Build, run, test
├── cpp/
│   ├── kmeans.h         # Core logic
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

Two input modes are supported:
- `gen` — automatic point generation (clusters around random centers with Gaussian noise)
- `manual` — manual input of `(x y class)` triples

## Results

### Example Output (n=300, k=3, m=3, gen)

```
=== Clustering Results ===
Iterations:        15
Total distance:    2075.04
Error rate:        0.0000%

Clusters:
   Cluster     Class    Points  Mismatched       Center X       Center Y
----------------------------------------------------------------------
         0         1        96           0        27.6272         9.6086
         1         2       102           0        -4.9647       -40.2977
         2         0       102           0        29.1706       -31.8084

First 20 points:
       #           X           Y     Class     Cluster
------------------------------------------------------
       0     31.0495    -26.6039         0           2
       1     27.2412    -30.8357         0           2
       2     -7.7641    -37.2897         2           1
       3     29.7654    -33.7955         0           2
       4     25.8289      5.9730         1           0
       5     -6.6719    -40.8218         2           1
       6     -5.0791    -47.1262         2           1
       7      0.2550    -40.5262         2           1
       8     24.9678     14.4222         1           0
       9     25.9089    -35.5484         0           2
```

With well-separated classes, the algorithm achieves **0% error** — all points are assigned to clusters matching their true class. The algorithm converged in **15 iterations**.

### Tests

| Language | Tests | Status |
|----------|-------|--------|
| C++ | 833 | Passed |
| Python | 28 | Passed |
| R | 34 | Passed |

Test coverage:
- **Unit**: `euclidean_distance`, `find_nearest_centroid`, `init_centroids`, `assign_clusters`, `update_centroids`, `determine_cluster_class`, `compute_total_distance`, `generate_points`
- **Integration**: `kmeans` — cluster count, assignment validity, error rate range, error count consistency
- **Edge cases**: well-separated clusters (error < 5%), single cluster (m=1), m=n, empty input, zero clusters
