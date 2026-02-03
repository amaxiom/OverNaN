# OverNaN Computation Guide

Technical documentation for the OverNaN implementation, covering parameters, reproducibility, parallel processing, and memory management.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Class Hierarchy](#class-hierarchy)
- [Parameter Reference](#parameter-reference)
  - [OverNaN (Unified Interface)](#overnan-unified-interface)
  - [SMOTENaN](#smotenan)
  - [ADASYNNaN](#adasynnan)
  - [ROSENaN](#rosenan)
- [Reproducibility](#reproducibility)
  - [Random State Management](#random-state-management)
  - [Deterministic Execution](#deterministic-execution)
  - [Cross-Platform Consistency](#cross-platform-consistency)
- [Parallel Processing](#parallel-processing)
  - [Joblib Backend](#joblib-backend)
  - [Parallelization Strategy](#parallelization-strategy)
  - [Thread Safety](#thread-safety)
  - [Performance Considerations](#performance-considerations)
- [Memory Management](#memory-management)
  - [Allocation Patterns](#allocation-patterns)
  - [Garbage Collection](#garbage-collection)
  - [Large Dataset Handling](#large-dataset-handling)
- [Data Type Handling](#data-type-handling)
  - [Input Validation](#input-validation)
  - [NaN Representation](#nan-representation)
  - [Pandas Compatibility](#pandas-compatibility)
- [Numerical Considerations](#numerical-considerations)
- [Error Handling](#error-handling)

---

## Architecture Overview

OverNaN follows a modular architecture with abstract base classes defining common interfaces and concrete implementations for each algorithm.

```
BaseOverSamplerNaN (ABC)
├── _determine_sampling_strategy()
├── fit_resample()
└── _resample_class() [abstract]

BaseNeighborOverSamplerNaN (BaseOverSamplerNaN)
├── _calculate_distance_with_nan()
├── _find_neighbours()
└── _generate_synthetic_sample()

SMOTENaN (BaseNeighborOverSamplerNaN)
├── _generate_samples_for_instance()
└── _resample_class()

ADASYNNaN (BaseNeighborOverSamplerNaN)
├── _calculate_density_distribution()
└── _resample_class()

ROSENaN (BaseOverSamplerNaN)
├── _compute_bandwidth()
├── _compute_class_nan_rates()
├── _compute_class_means()
├── _generate_synthetic_sample_rose()
├── _generate_samples_batch()
└── _resample_class()

OverNaN (BaseOverSamplerNaN)
├── _oversampler [composition]
└── fit_resample() [delegation]
```

---

## Class Hierarchy

### Inheritance Structure

| Class | Inherits From | Purpose |
|-------|---------------|---------|
| `BaseOverSamplerNaN` | `BaseEstimator`, `TransformerMixin`, `ABC` | Common sampling strategy and pandas handling |
| `BaseNeighborOverSamplerNaN` | `BaseOverSamplerNaN` | NaN-aware distance and neighbor finding |
| `SMOTENaN` | `BaseNeighborOverSamplerNaN` | SMOTE implementation |
| `ADASYNNaN` | `BaseNeighborOverSamplerNaN` | ADASYN implementation |
| `ROSENaN` | `BaseOverSamplerNaN` | ROSE implementation (no neighbors) |
| `OverNaN` | `BaseOverSamplerNaN` | Unified interface via composition |

### Design Patterns

- **Template Method**: `fit_resample()` defines the skeleton; `_resample_class()` is overridden
- **Strategy**: `OverNaN` delegates to concrete implementations via `_oversampler`
- **Factory**: `OverNaN.__init__()` instantiates the appropriate algorithm class

---

## Parameter Reference

### OverNaN (Unified Interface)

```python
OverNaN(
    method='SMOTE',
    neighbours=5,
    sampling_strategy='auto',
    random_state=None,
    nan_handling='preserve_pattern',
    n_jobs=None,
    **kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `'SMOTE'` | Algorithm: `'SMOTE'`, `'ADASYN'`, or `'ROSE'` |
| `neighbours` | `int` | `5` | k-nearest neighbors (SMOTE/ADASYN only) |
| `sampling_strategy` | `str`, `float`, `dict` | `'auto'` | Target class distribution |
| `random_state` | `int`, `None` | `None` | Seed for reproducibility |
| `nan_handling` | `str` | `'preserve_pattern'` | NaN synthesis strategy |
| `n_jobs` | `int`, `None` | `None` | Parallel jobs (`None`=1, `-1`=all cores) |
| `**kwargs` | | | Method-specific parameters |

#### Method-Specific kwargs

| Parameter | Methods | Type | Default | Description |
|-----------|---------|------|---------|-------------|
| `beta` | ADASYN | `float` | `1.0` | Balance level multiplier |
| `learning_rate` | ADASYN | `float` | `1.0` | Density weight exponent |
| `shrinkage` | ROSE | `float` | `1.0` | Bandwidth scaling factor |

### SMOTENaN

```python
SMOTENaN(
    neighbours=5,
    sampling_strategy='auto',
    random_state=None,
    nan_handling='preserve_pattern',
    n_jobs=None
)
```

| Parameter | Type | Default | Valid Values |
|-----------|------|---------|--------------|
| `neighbours` | `int` | `5` | `>= 1` |
| `sampling_strategy` | `str`, `float`, `dict` | `'auto'` | See [Sampling Strategy](#sampling-strategy-parameter) |
| `random_state` | `int`, `RandomState`, `None` | `None` | Any integer or None |
| `nan_handling` | `str` | `'preserve_pattern'` | `'preserve_pattern'`, `'interpolate'`, `'random_pattern'` |
| `n_jobs` | `int`, `None` | `None` | `None`, `-1`, or positive integer |

### ADASYNNaN

```python
ADASYNNaN(
    neighbours=5,
    sampling_strategy='auto',
    random_state=None,
    nan_handling='preserve_pattern',
    n_jobs=None,
    beta=1.0,
    learning_rate=1.0
)
```

| Parameter | Type | Default | Valid Values |
|-----------|------|---------|--------------|
| `neighbours` | `int` | `5` | `>= 1` |
| `sampling_strategy` | `str`, `float`, `dict` | `'auto'` | See [Sampling Strategy](#sampling-strategy-parameter) |
| `random_state` | `int`, `RandomState`, `None` | `None` | Any integer or None |
| `nan_handling` | `str` | `'preserve_pattern'` | `'preserve_pattern'`, `'interpolate'`, `'random_pattern'` |
| `n_jobs` | `int`, `None` | `None` | `None`, `-1`, or positive integer |
| `beta` | `float` | `1.0` | `> 0` (typically `0.5` to `2.0`) |
| `learning_rate` | `float` | `1.0` | `> 0` (typically `0.5` to `2.0`) |

### ROSENaN

```python
ROSENaN(
    sampling_strategy='auto',
    random_state=None,
    nan_handling='preserve_pattern',
    n_jobs=None,
    shrinkage=1.0
)
```

| Parameter | Type | Default | Valid Values |
|-----------|------|---------|--------------|
| `sampling_strategy` | `str`, `float`, `dict` | `'auto'` | See [Sampling Strategy](#sampling-strategy-parameter) |
| `random_state` | `int`, `RandomState`, `None` | `None` | Any integer or None |
| `nan_handling` | `str` | `'preserve_pattern'` | `'preserve_pattern'`, `'interpolate'`, `'random_pattern'` |
| `n_jobs` | `int`, `None` | `None` | `None`, `-1`, or positive integer |
| `shrinkage` | `float` | `1.0` | `>= 0` |

### Sampling Strategy Parameter

The `sampling_strategy` parameter accepts multiple types:

| Type | Example | Behavior |
|------|---------|----------|
| `str` | `'auto'` | Predefined strategy (see below) |
| `float` | `0.8` | Target minority/majority ratio |
| `dict` | `{0: 100, 1: 80}` | Explicit target counts per class |

#### String Strategies

| Value | Description |
|-------|-------------|
| `'auto'` | Resample all classes except majority to match majority count |
| `'minority'` | Resample only the smallest class to match majority |
| `'not majority'` | Same as `'auto'` |
| `'not minority'` | Resample all except minority to match majority |
| `'all'` | Resample all classes to match majority |

### NaN Handling Parameter

| Value | Synthetic Sample Behavior |
|-------|---------------------------|
| `'preserve_pattern'` | Feature is NaN if either parent has NaN (conservative) |
| `'interpolate'` | Use available value if one parent has it; interpolate if both have values |
| `'random_pattern'` | Probabilistically assign NaN based on parent patterns |

---

## Reproducibility

### Random State Management

OverNaN uses scikit-learn's `check_random_state()` for consistent random number generation:

```python
from sklearn.utils import check_random_state

# In fit_resample():
random_state = check_random_state(self.random_state)
```

#### Acceptable Input Types

| Input | Behavior |
|-------|----------|
| `None` | Uses `np.random.mtrand._rand` (non-reproducible) |
| `int` | Creates `RandomState(seed)` |
| `RandomState` | Uses directly |

#### Example: Ensuring Reproducibility

```python
from overnan import OverNaN

# Reproducible
over1 = OverNaN(method='SMOTE', random_state=42)
over2 = OverNaN(method='SMOTE', random_state=42)

X_res1, y_res1 = over1.fit_resample(X, y)
X_res2, y_res2 = over2.fit_resample(X, y)

assert np.allclose(X_res1, X_res2, equal_nan=True)  # True
```

### Deterministic Execution

For fully deterministic results:

1. **Set `random_state`**: Pass an integer seed
2. **Use sequential processing**: Set `n_jobs=1` or `n_jobs=None`
3. **Control numpy global state** (if not setting `random_state`):

```python
import numpy as np
np.random.seed(42)
```

#### Parallel Processing and Reproducibility

When `n_jobs != 1`, each parallel worker receives an independent seed derived from the master random state:

```python
# In parallel methods:
seeds = random_state.randint(0, 2**31 - 1, size=n_workers)

# Each worker gets:
worker_rng = check_random_state(seeds[worker_id])
```

This ensures reproducibility across runs but results may differ from sequential execution due to different sample generation order.

### Cross-Platform Consistency

Seed generation uses `2**31 - 1` (not `2**32 - 1`) for Windows int32 compatibility:

```python
# Correct (cross-platform):
seeds = random_state.randint(0, 2**31 - 1, size=n)

# Incorrect (fails on Windows):
# seeds = random_state.randint(0, 2**32 - 1, size=n)
```

---

## Parallel Processing

### Joblib Backend

OverNaN uses joblib for parallel execution:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=self.n_jobs)(
    delayed(self._process_function)(args)
    for args in work_items
)
```

### Parallelization Strategy

#### SMOTENaN

Parallelizes across seed samples in the minority class:

```python
# Each minority sample processed independently
results = Parallel(n_jobs=self.n_jobs)(
    delayed(self._generate_samples_for_instance)(
        X_class[i], X_class, n_synthetic_list[i], 
        check_random_state(seeds[i])
    )
    for i in range(n_minority) if n_synthetic_list[i] > 0
)
```

**Granularity**: One task per minority sample

#### ADASYNNaN

Parallelizes density calculation across minority samples:

```python
# Density computation parallelized
densities = Parallel(n_jobs=self.n_jobs)(
    delayed(compute_density)(sample) for sample in X_minority
)
```

**Granularity**: One task per minority sample

#### ROSENaN

Parallelizes batch generation across workers:

```python
# Samples distributed across workers
job_sizes = distribute_samples(n_samples, n_jobs)

results = Parallel(n_jobs=self.n_jobs)(
    delayed(self._generate_samples_batch)(
        X_class, bandwidths, job_sizes[i], check_random_state(seeds[i]),
        nan_rates, class_means
    )
    for i in range(len(job_sizes))
)
```

**Granularity**: Batches of samples per worker

### Thread Safety

OverNaN is thread-safe under the following conditions:

1. **Separate instances**: Each thread uses its own `OverNaN` instance
2. **Immutable input**: Input arrays are not modified during processing
3. **Independent random states**: Each instance has its own `random_state`

#### Unsafe Pattern

```python
# UNSAFE: Shared instance across threads
shared_oversampler = OverNaN(method='SMOTE')

def worker(X, y):
    return shared_oversampler.fit_resample(X, y)  # Race condition
```

#### Safe Pattern

```python
# SAFE: Instance per thread
def worker(X, y, seed):
    oversampler = OverNaN(method='SMOTE', random_state=seed)
    return oversampler.fit_resample(X, y)
```

### Performance Considerations

#### n_jobs Selection

| `n_jobs` | Behavior |
|----------|----------|
| `None` | Sequential execution (1 worker) |
| `1` | Sequential execution (1 worker) |
| `-1` | Use all available CPU cores |
| `n` | Use exactly `n` workers |

#### Overhead Analysis

Parallel processing introduces overhead:

| Factor | Impact |
|--------|--------|
| Process spawning | Fixed cost per call (~100ms) |
| Data serialization | Proportional to data size |
| Result aggregation | Proportional to output size |

**Recommendation**: Use `n_jobs=-1` when:
- `n_minority > 100` (enough parallelizable work)
- `n_features > 10` (expensive distance calculations)
- Total synthetic samples > 1000

#### Memory Multiplication

With `n_jobs=k`, peak memory usage increases approximately `k`-fold due to data copying to worker processes.

---

## Memory Management

### Allocation Patterns

#### Pre-allocation vs Dynamic Growth

OverNaN uses list accumulation followed by array conversion:

```python
# Pattern used:
synthetic_samples = []
for ...:
    synthetic_samples.append(sample)
return np.array(synthetic_samples)
```

This avoids repeated array reallocation but requires final conversion overhead.

#### Memory Lifecycle

```
fit_resample() called
    ├── Input validation (copy created for float64 conversion)
    ├── Per-class processing
    │   ├── Extract class samples (view, not copy)
    │   ├── Generate synthetic samples (new allocations)
    │   └── Append to resampled arrays (vstack creates copies)
    ├── Pandas conversion (if applicable)
    └── gc.collect() called
```

### Garbage Collection

Explicit garbage collection is triggered after `fit_resample()`:

```python
def fit_resample(self, X, y):
    # ... processing ...
    
    # Memory cleanup
    gc.collect()
    
    return X_resampled, y_resampled
```

#### Manual Memory Management

For memory-constrained environments:

```python
import gc

# Process in batches
for batch_X, batch_y in data_batches:
    oversampler = OverNaN(method='ROSE', random_state=42)
    X_res, y_res = oversampler.fit_resample(batch_X, batch_y)
    
    # Process results
    process(X_res, y_res)
    
    # Explicit cleanup
    del X_res, y_res, oversampler
    gc.collect()
```

### Large Dataset Handling

#### Memory Estimation

Approximate peak memory for `fit_resample()`:

```
Peak Memory ≈ sizeof(X) * 4 + sizeof(synthetic_samples)

Where:
- sizeof(X) = n_samples * n_features * 8 bytes (float64)
- Factor of 4 accounts for: input, validation copy, class extraction, output
- synthetic_samples = n_synthetic * n_features * 8 bytes
```

#### Example Calculation

For a dataset with 100,000 samples, 50 features, generating 50,000 synthetic samples:

```
Input:      100,000 * 50 * 8 =  40 MB
Peak:       40 MB * 4        = 160 MB
Synthetic:  50,000 * 50 * 8  =  20 MB
Total Peak: ~180 MB
```

#### Strategies for Large Datasets

1. **Reduce parallel workers**: Lower `n_jobs` reduces memory multiplication
2. **Process subsets**: Split data and process incrementally
3. **Use ROSE**: No neighbor search reduces intermediate allocations

```python
# Memory-efficient approach for very large datasets
def memory_efficient_resample(X, y, chunk_size=10000):
    """Process large datasets in chunks."""
    from overnan import OverNaN
    
    results_X = []
    results_y = []
    
    # Get minority indices
    minority_class = np.argmin(np.bincount(y))
    minority_idx = np.where(y == minority_class)[0]
    
    # Process minority class in chunks
    for start in range(0, len(minority_idx), chunk_size):
        end = min(start + chunk_size, len(minority_idx))
        chunk_idx = minority_idx[start:end]
        
        # Include some majority samples for context
        majority_idx = np.where(y != minority_class)[0]
        sample_majority = np.random.choice(
            majority_idx, 
            size=min(len(majority_idx), chunk_size),
            replace=False
        )
        
        combined_idx = np.concatenate([chunk_idx, sample_majority])
        X_chunk = X[combined_idx]
        y_chunk = y[combined_idx]
        
        oversampler = OverNaN(method='ROSE', n_jobs=1, random_state=42)
        X_res, y_res = oversampler.fit_resample(X_chunk, y_chunk)
        
        results_X.append(X_res)
        results_y.append(y_res)
        
        gc.collect()
    
    return np.vstack(results_X), np.concatenate(results_y)
```

---

## Data Type Handling

### Input Validation

Input validation uses scikit-learn utilities:

```python
from sklearn.utils.validation import check_array

# Feature matrix validation
try:
    X = check_array(X, ensure_all_finite='allow-nan', dtype=np.float64)
except TypeError:
    # Fallback for older sklearn versions
    X = check_array(X, force_all_finite='allow-nan', dtype=np.float64)

# Target validation
y = check_array(y, ensure_2d=False, dtype=None)
```

#### Accepted Input Types

| Type | Handling |
|------|----------|
| `np.ndarray` | Direct validation |
| `pd.DataFrame` | Converted to array; metadata preserved |
| `pd.Series` | Converted to array; metadata preserved |
| `list` | Converted to array |
| `scipy.sparse` | **Not supported** (raises error) |

### NaN Representation

OverNaN uses numpy's `np.nan` (IEEE 754 NaN):

```python
# Internal NaN checks
np.isnan(value)  # Scalar
np.isnan(array)  # Vectorized

# NaN-aware array comparison
np.array_equal(a, b, equal_nan=True)
```

#### Special Values

| Value | Treatment |
|-------|-----------|
| `np.nan` | Recognized as missing |
| `float('nan')` | Recognized as missing |
| `None` | Converted to `np.nan` during validation |
| `np.inf` | **Not treated as missing** |
| `-np.inf` | **Not treated as missing** |

### Pandas Compatibility

#### Metadata Preservation

```python
def fit_resample(self, X, y):
    # Store pandas metadata
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_series = isinstance(y, pd.Series)
    X_columns = X.columns if X_is_df else None
    y_name = y.name if y_is_series else None
    
    # ... processing on numpy arrays ...
    
    # Restore pandas types
    if X_is_df:
        X_resampled = pd.DataFrame(X_resampled, columns=X_columns)
    if y_is_series:
        y_resampled = pd.Series(y_resampled, name=y_name)
    
    return X_resampled, y_resampled
```

#### Index Handling

Output DataFrames/Series have a **reset integer index** (0, 1, 2, ...). Original indices are not preserved because:

1. Synthetic samples have no original index
2. Row order changes during resampling

---

## Numerical Considerations

### Distance Calculation Scaling

NaN-aware Euclidean distance scales by valid feature proportion:

```python
def _calculate_distance_with_nan(self, x1, x2):
    valid_mask = ~(np.isnan(x1) | np.isnan(x2))
    
    if not np.any(valid_mask):
        return np.inf
    
    diff = x1[valid_mask] - x2[valid_mask]
    n_valid = np.sum(valid_mask)
    n_total = len(x1)
    
    # Scale to compensate for reduced dimensionality
    distance = np.sqrt(np.sum(diff ** 2) * (n_total / n_valid))
    
    return distance
```

**Rationale**: Without scaling, samples with fewer valid features would appear artificially closer.

### Bandwidth Computation (ROSE)

Silverman's rule of thumb:

```python
def _compute_bandwidth(self, X_class):
    n_samples, n_features = X_class.shape
    
    # Silverman's constant
    silverman_constant = (4.0 / (n_features + 2.0)) ** (1.0 / (n_features + 4.0))
    
    # Sample size factor
    n_factor = n_samples ** (-1.0 / (n_features + 4.0))
    
    # Per-feature bandwidth
    bandwidths = np.zeros(n_features)
    for j in range(n_features):
        col = X_class[:, j]
        valid_values = col[~np.isnan(col)]
        if len(valid_values) > 1:
            sigma_j = np.std(valid_values, ddof=1)
            bandwidths[j] = silverman_constant * n_factor * sigma_j * self.shrinkage
        else:
            bandwidths[j] = 0.0  # No perturbation for constant/missing features
    
    return bandwidths
```

### Floating Point Precision

All computations use `float64` (double precision):

```python
X = check_array(X, ..., dtype=np.float64)
```

This ensures:
- Sufficient precision for distance calculations
- Consistent behavior across platforms
- Compatibility with downstream models

---

## Error Handling

### Common Exceptions

| Exception | Cause | Resolution |
|-----------|-------|------------|
| `ValueError("Invalid method")` | Unknown method string | Use `'SMOTE'`, `'ADASYN'`, or `'ROSE'` |
| `ValueError("Invalid sampling_strategy")` | Unrecognized strategy | Check parameter type and value |
| `UserWarning("Only found k neighbors")` | Fewer neighbors than requested | Reduce `neighbours` parameter |

### Warning Behavior

Warnings use Python's `warnings` module:

```python
import warnings

if len(distances) < n_neighbours:
    warnings.warn(f"Only found {len(distances)} neighbors, requested {n_neighbours}")
    n_neighbours = len(distances)
```

To suppress warnings:

```python
import warnings
warnings.filterwarnings('ignore')

# Or specifically:
warnings.filterwarnings('ignore', category=UserWarning, module='overnan')
```

### Graceful Degradation

When neighbor-based methods encounter edge cases:

| Condition | Behavior |
|-----------|----------|
| No valid neighbors (all inf distance) | Sample skipped, warning issued |
| Fewer neighbors than k | Uses available neighbors, warning issued |
| All-NaN sample | Excluded from neighbor search, preserved in output |
| Single minority sample | No synthetic generation possible (needs at least 2) |

---

## Version Information

```python
import overnan
print(overnan.__version__)  # '0.2.0'
```

### Changelog

| Version | Changes |
|---------|---------|
| 0.1.0 | Initial release (SMOTENaN, ADASYNNaN) |
| 0.2.0 | Added ROSENaN, refactored class hierarchy, Windows int32 fix |

---

## Dependencies

### Required

| Package | Minimum Version | Purpose |
|---------|-----------------|---------|
| numpy | 1.19.0 | Array operations |
| pandas | 1.1.0 | DataFrame support |
| scikit-learn | 0.24.0 | Validation utilities, base classes |
| joblib | 1.0.0 | Parallel processing |

### Import Structure

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
import warnings
import gc
```

---

*OverNaN: Oversampling for Imbalanced Learning with Missing Values*

Repository: [https://github.com/amaxiom/OverNaN](https://github.com/amaxiom/OverNaN)