# OverNaN Testing and Benchmarking Guide

This directory contains the test suite and benchmarking tools for validating OverNaN's NaN-aware oversampling algorithms.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Suite](#test-suite)
  - [Test Categories](#test-categories)
  - [Running Tests](#running-tests)
  - [Test Configuration](#test-configuration)
- [Benchmark Suite](#benchmark-suite)
  - [Benchmark Datasets](#benchmark-datasets)
  - [Metrics](#metrics)
  - [Running Benchmarks](#running-benchmarks)
  - [Interpreting Results](#interpreting-results)
- [Comparison with Impute-Then-Oversample](#comparison-with-impute-then-oversample)
- [Adding New Tests](#adding-new-tests)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

OverNaN provides two validation tools:

| File | Purpose |
|------|---------|
| `OverNaN_test.py` | Unit tests and integration tests for correctness validation |
| `OverNaN_bench.py` | Performance benchmarks on real-world datasets with missing values |

The test suite validates that all three algorithms (SMOTENaN, ADASYNNaN, ROSENaN) function correctly across various configurations, while the benchmark suite measures their practical effectiveness on classification tasks. The aim is to achieve similar or better classification performance, while preserving the missingness.

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scikit-learn joblib xgboost openml imbalanced-learn

# Run tests
python OverNaN_test.py

# Run benchmarks
python OverNaN_bench.py
```

---

## Test Suite

### Test Categories

The test suite (`OverNaN_test.py`) validates OverNaN through the following test categories:

#### 1. Basic Functionality (`test_basic_functionality`)

Verifies that each method produces valid output:
- Correct output shapes
- Balanced class distributions
- NaN values preserved appropriately

#### 2. Sampling Strategies (`test_sampling_strategies`)

Tests all supported sampling strategies:

| Strategy | Description |
|----------|-------------|
| `'auto'` | Balance all minority classes to match majority |
| `'minority'` | Oversample only the smallest class |
| `'not majority'` | Oversample all classes except the largest |
| `float` (e.g., `0.8`) | Target ratio relative to majority class |
| `dict` (e.g., `{0: 100, 1: 80}`) | Explicit target counts per class |

#### 3. NaN Handling Strategies (`test_nan_handling_strategies`)

Validates the three NaN handling modes:

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `'preserve_pattern'` | NaN if either parent has NaN | Conservative; maintains missingness structure |
| `'interpolate'` | Use available values when possible | Aggressive; minimizes NaN in output |
| `'random_pattern'` | Probabilistic NaN based on class rates | Balanced; preserves statistical properties |

#### 4. ROSE Shrinkage Parameter (`test_rose_shrinkage`)

Tests the bandwidth scaling parameter specific to ROSE:

| Shrinkage | Effect |
|-----------|--------|
| `0.0` | Exact copies (equivalent to random oversampling) |
| `0.5` | Samples closer to seeds |
| `1.0` | Standard Silverman bandwidth |
| `>1.0` | More dispersed samples |

#### 5. Pandas Compatibility (`test_pandas_compatibility`)

Ensures seamless integration with pandas:
- DataFrame input produces DataFrame output
- Column names preserved
- Series names preserved
- Index handling

#### 6. Parallel Processing (`test_parallel_processing`)

Validates joblib parallelization:
- `n_jobs=1` (sequential) vs `n_jobs=-1` (all cores)
- Shape consistency between modes
- Performance scaling

#### 7. Direct Class Usage (`test_direct_class_usage`)

Tests direct instantiation of algorithm classes:
- `SMOTENaN`
- `ADASYNNaN`
- `ROSENaN`

#### 8. Classifier Integration (`test_with_classifiers`)

End-to-end integration with XGBoost:
- Train/test split with stratification
- Performance comparison across methods
- Metric reporting (precision, recall, F1)

#### 9. Edge Cases (`test_edge_cases`)

Validates robustness under challenging conditions:
- Very few minority samples (k > n)
- Samples with all-NaN features
- High NaN percentage (50%)
- Single-feature datasets

#### 10. Reproducibility (`test_reproducibility`)

Confirms deterministic behavior:
- Same `random_state` produces identical results
- Shape consistency
- Value-level equality (with NaN handling)

#### 11. String Representation (`test_repr`)

Validates `__repr__` output for debugging and logging.

### Running Tests

```bash
# Run all tests
python OverNaN_test.py

# Run specific test (modify main block)
python -c "from OverNaN_test import test_basic_functionality; test_basic_functionality()"
```

### Test Configuration

The test suite uses a configurable data generator:

```python
def create_imbalanced_data_with_nan(
    n_majority=500,      # Majority class samples
    n_minority=50,       # Minority class samples
    n_features=10,       # Number of features
    nan_percentage=0.2,  # Proportion of NaN values
    random_state=42      # Reproducibility seed
)
```

Modify these parameters to test specific scenarios.

---

## Benchmark Suite

### Benchmark Datasets

The benchmark suite (`OverNaN_bench.py`) uses publicly available OpenML datasets with naturally occurring missing values. All datasets are non-biomedical to ensure broad applicability.

| Dataset | OpenML ID | Domain | Samples | Description |
|---------|-----------|--------|---------|-------------|
| adult | 1590 | Census/Economics | 48,842 | Income prediction (>50K vs <=50K) |
| cylinder-bands | 6332 | Engineering | 540 | Process delay prediction |
| Titanic | 40945 | Transportation | 1,309 | Survival prediction |
| soybean | 1023 | Environment | 683 | STEM profession prediction |
| car-evaluation | 1017 | Consumer | 1,728 | Car quality classification |

#### Dataset Selection Criteria

Datasets were selected based on:
1. **Natural missingness**: Missing values occur organically, not artificially introduced
2. **Class imbalance**: Minority class represents a meaningful challenge
3. **Domain diversity**: Non-biomedical to avoid domain-specific concerns
4. **Accessibility**: Freely available through OpenML
5. **Size variety**: Range from hundreds to tens of thousands of samples

### Metrics

The benchmark reports the following metrics:

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **Balanced Accuracy** | Average of recall for each class | [0, 1] | Accounts for class imbalance |
| **F1-Macro** | Unweighted mean of per-class F1 | [0, 1] | Equal weight to all classes |
| **F1-Minority** | F1-score for minority class only | [0, 1] | Focus on minority performance |
| **G-Mean** | Geometric mean of class recalls | [0, 1] | Sensitive to poor class performance |
| **ROC-AUC** | Area under ROC curve | [0, 1] | Ranking quality |

#### Why These Metrics?

Standard accuracy is misleading for imbalanced data. For example, with 95% majority class, predicting all samples as majority achieves 95% accuracy but 0% minority recall.

**Balanced Accuracy** and **G-Mean** specifically address this by weighting classes equally. **F1-Minority** directly measures the metric most practitioners care about: detecting the rare class.

### Running Benchmarks

```bash
# Full benchmark (all datasets)
python OverNaN_bench.py

# Results saved to: overnan_benchmark_results.csv
```

#### Programmatic Usage

```python
from OverNaN_bench import run_full_benchmark, print_summary_table

# Run with custom parameters
results = run_full_benchmark(
    datasets=None,      # Use default datasets, or pass custom list
    n_splits=5,         # Cross-validation folds
    random_state=42     # Reproducibility
)

# Display summary
print_summary_table(results)
```

#### Single Dataset Benchmark

```python
from OverNaN_bench import DatasetConfig, run_dataset_benchmark

# Define custom dataset
my_dataset = DatasetConfig(
    openml_id=989,
    name="anneal",
    description="Steel grade prediction",
    domain="Materials Science"
)

# Run benchmark
results = run_dataset_benchmark(my_dataset, n_splits=5, random_state=42)
```

### Interpreting Results

#### Sample Output

```
======================================================================
BENCHMARK SUMMARY
======================================================================

adult (NaN: 0.9%, IR: 3.2)
------------------------------------------------------------
Method       Bal.Acc    F1-Macro     F1-Min     G-Mean    ROC-AUC
------------------------------------------------------------
Baseline     0.742±0.01 0.721±0.01 0.612±0.02 0.738±0.01 0.891±0.01
SMOTENaN     0.768±0.01 0.749±0.01 0.658±0.02 0.765±0.01 0.898±0.01
ADASYNNaN    0.771±0.01 0.752±0.01 0.663±0.02 0.769±0.01 0.899±0.01
ROSENaN      0.759±0.01 0.738±0.01 0.641±0.02 0.756±0.01 0.894±0.01
```

#### Key Observations

1. **IR (Imbalance Ratio)**: Higher values indicate more severe imbalance
2. **NaN %**: Percentage of missing values in the dataset
3. **Mean ± Std**: Cross-validation mean and standard deviation
4. **Method Comparison**: Compare oversampling methods against baseline (no oversampling)

#### What to Look For

| Observation | Interpretation |
|-------------|----------------|
| Oversampling > Baseline | Method improves minority class detection |
| High std deviation | Unstable performance; may need more data or tuning |
| ROSE competitive with SMOTE | KDE approach viable; faster for high dimensions |
| ADASYN > SMOTE | Adaptive sampling helps on this dataset |

---

## Comparison with Impute-Then-Oversample

The benchmark includes an optional comparison between OverNaN and the naive approach of imputing missing values before applying standard oversampling.

### Approaches Compared

| Approach | Description |
|----------|-------------|
| Baseline | No oversampling |
| Impute+SMOTE | Mean imputation, then standard SMOTE |
| Impute+ADASYN | Mean imputation, then standard ADASYN |
| Impute+ROSE | Mean imputation, then standard ROSE |
| OverNaN-SMOTE | NaN-aware SMOTE (no imputation) |
| OverNaN-ADASYN | NaN-aware ADASYN (no imputation) |
| OverNaN-ROSE | NaN-aware ROSE (no imputation) |

### Running the Comparison

```python
from OverNaN_bench import run_impute_comparison, BENCHMARK_DATASETS

# Compare on adult dataset
results = run_impute_comparison(
    BENCHMARK_DATASETS[0],  # adult
    n_splits=5,
    random_state=42
)
```

### Why OverNaN May Outperform Impute-Then-Oversample

1. **Information preservation**: Imputation destroys the information encoded in missingness patterns
2. **Bias avoidance**: Mean imputation can shift feature distributions
3. **Synthetic sample quality**: OverNaN generates samples that respect the original data structure
4. **Downstream compatibility**: XGBoost and similar models handle NaN natively

---

## Adding New Tests

### Adding a Unit Test

```python
def test_my_new_feature():
    """
    Test description here.
    """
    print("\n" + "=" * 70)
    print("TESTING MY NEW FEATURE")
    print("=" * 70)
    
    # Setup
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20)
    
    # Test logic
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        oversampler = OverNaN(method=method, random_state=42)
        X_res, y_res = oversampler.fit_resample(X, y)
        
        # Assertions
        assert X_res.shape[0] > X.shape[0], f"{method} should increase samples"
        print(f"  {method}: PASSED")

# Add to run_all_tests():
def run_all_tests():
    # ... existing tests ...
    test_my_new_feature()  # Add here
```

### Adding a Benchmark Dataset

```python
from OverNaN_bench import DatasetConfig, BENCHMARK_DATASETS

# Add new dataset
new_dataset = DatasetConfig(
    openml_id=12345,          # OpenML dataset ID
    name="my-dataset",        # Short name
    description="Description", # What it predicts
    domain="Domain Category"  # e.g., Finance, Social, etc.
)

# Option 1: Add to default list
BENCHMARK_DATASETS.append(new_dataset)

# Option 2: Run separately
from OverNaN_bench import run_dataset_benchmark
results = run_dataset_benchmark(new_dataset)
```

---

## Troubleshooting

### Common Issues

#### 1. OpenML Connection Error

```
Error: HTTPSConnectionPool... Max retries exceeded
```

**Solution**: Check internet connection. OpenML may be temporarily unavailable.

```python
# Test connection
import openml
openml.config.server = "https://www.openml.org/api/v1/xml"
datasets = openml.datasets.list_datasets(size=1)
```

#### 2. Categorical Column Error

```
Error: Cannot setitem on a Categorical with a new category
```

**Solution**: This was fixed in v0.2.0. Ensure you have the latest `OverNaN_bench.py`.

#### 3. Int32 Overflow on Windows

```
ValueError: high is out of bounds for int32
```

**Solution**: Fixed in v0.2.0. Seed generation now uses `2**31 - 1` instead of `2**32 - 1`.

#### 4. Memory Error on Large Datasets

**Solution**: Reduce dataset size or use fewer cross-validation folds:

```python
results = run_full_benchmark(n_splits=3)  # Reduce from 5
```

#### 5. imbalanced-learn Import Error

```
ImportError: No module named 'imblearn'
```

**Solution**: Install imbalanced-learn for comparison benchmarks:

```bash
pip install imbalanced-learn
```

### Dependency Versions

Tested with:

| Package | Minimum Version |
|---------|-----------------|
| Python | 3.8+ |
| numpy | 1.19.0 |
| pandas | 1.1.0 |
| scikit-learn | 0.24.0 |
| joblib | 1.0.0 |
| xgboost | 1.4.0 |
| openml | 0.12.0 |
| imbalanced-learn | 0.8.0 (optional) |

---

## References

### Algorithms

1. **SMOTE**: Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357. [DOI: 10.1613/jair.953](https://doi.org/10.1613/jair.953)

2. **ADASYN**: He, H., Bai, Y., Garcia, E.A., Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. *IEEE International Joint Conference on Neural Networks*, 1322-1328. [DOI: 10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969)

3. **ROSE**: Menardi, G. and Torelli, N. (2014). Training and assessing classification rules with imbalanced data. *Data Mining and Knowledge Discovery*, 28, 92-122. [DOI: 10.1007/s10618-012-0295-5](https://doi.org/10.1007/s10618-012-0295-5)

### Data Source

4. **OpenML**: Vanschoren, J., van Rijn, J.N., Bischl, B., Torgo, L. (2014). OpenML: Networked Science in Machine Learning. *SIGKDD Explorations*, 15(2), 49-60. [DOI: 10.1145/2641190.2641198](https://doi.org/10.1145/2641190.2641198)

---

## Contributing

Contributions to the test and benchmark suites are welcome. Please ensure:

1. New tests follow the existing naming convention (`test_*`)
2. Tests include docstrings explaining purpose
3. Benchmarks use reproducible random states
4. All tests pass before submitting PR

```bash
# Verify all tests pass
python OverNaN_test.py

# Run benchmarks to check for regressions
python OverNaN_bench.py
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file in the repository root.

---

*OverNaN: Oversampling for Imbalanced Learning with Missing Values*

Repository: [https://github.com/amaxiom/OverNaN](https://github.com/amaxiom/OverNaN)