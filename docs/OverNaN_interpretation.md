# OverNaN Interpretation Guide

Comprehensive guide to understanding and applying OverNaN's NaN-aware oversampling methods for imbalanced classification with missing data.

## Table of Contents

- [Introduction](#introduction)
  - [The Problem](#the-problem)
  - [The OverNaN Solution](#the-overnan-solution)
- [Algorithm Overview](#algorithm-overview)
  - [SMOTENaN](#smotenan)
  - [ADASYNNaN](#adasynnan)
  - [ROSENaN](#rosenan)
  - [Algorithm Comparison](#algorithm-comparison)
- [NaN Handling Strategies](#nan-handling-strategies)
  - [preserve_pattern](#preserve_pattern)
  - [interpolate](#interpolate)
  - [random_pattern](#random_pattern)
  - [Strategy Selection Guide](#strategy-selection-guide)
- [Sampling Strategies](#sampling-strategies)
  - [String Strategies](#string-strategies)
  - [Ratio-Based Strategy](#ratio-based-strategy)
  - [Dictionary Strategy](#dictionary-strategy)
  - [Sampling Strategy Selection](#sampling-strategy-selection)
- [Parameter Tuning](#parameter-tuning)
  - [neighbours Parameter](#neighbours-parameter)
  - [ADASYN Parameters](#adasyn-parameters)
  - [ROSE shrinkage Parameter](#rose-shrinkage-parameter)
- [Output Interpretation](#output-interpretation)
  - [Understanding Resampled Data](#understanding-resampled-data)
  - [NaN Distribution in Output](#nan-distribution-in-output)
  - [Quality Assessment](#quality-assessment)
- [Method Selection Guide](#method-selection-guide)
- [Usage Examples](#usage-examples)
  - [Basic Usage](#basic-usage)
  - [High-Dimensional Data](#high-dimensional-data)
  - [Preserving Missingness Structure](#preserving-missingness-structure)
  - [Aggressive NaN Reduction](#aggressive-nan-reduction)
  - [Partial Balancing](#partial-balancing)
  - [Multi-Class Imbalance](#multi-class-imbalance)
  - [Integration with Classifiers](#integration-with-classifiers)
  - [Cross-Validation Pipeline](#cross-validation-pipeline)
- [Domain-Specific Considerations](#domain-specific-considerations)
- [Limitations and Caveats](#limitations-and-caveats)
- [References](#references)

---

## Introduction

### The Problem

Classification with imbalanced data and missing values presents a compound challenge:

**Class Imbalance**: When one class significantly outnumbers others, classifiers tend to favor the majority class, resulting in poor minority class detection. This is problematic in applications where the minority class is the primary interest (e.g., fraud detection, rare event prediction, anomaly identification).

**Missing Values**: Real-world datasets frequently contain missing values that carry semantic meaning:
- Sensor failures indicate equipment issues
- Unanswered survey questions reflect respondent characteristics
- Missing financial records may signal reporting problems

**The Compound Problem**: Standard oversampling methods (SMOTE, ADASYN) require complete data, forcing practitioners to either:
1. **Impute then oversample**: Destroys information encoded in missingness patterns
2. **Delete incomplete samples**: Loses valuable data and may introduce bias
3. **Ignore the problem**: Accept degraded model performance

### The OverNaN Solution

OverNaN implements three oversampling algorithms that natively handle missing values:

| Method | Approach | Key Advantage |
|--------|----------|---------------|
| SMOTENaN | Neighbor interpolation with NaN-aware distance | Preserves local data structure |
| ADASYNNaN | Adaptive neighbor interpolation | Focuses on difficult regions |
| ROSENaN | Kernel density perturbation | No neighbor search required |

Each method generates synthetic minority samples while respecting and preserving missing value patterns, eliminating the need for imputation. The aim is to achieve similar or better classification performance, while only using real data.

---

## Algorithm Overview

### SMOTENaN

**Synthetic Minority Over-sampling Technique with NaN Handling**

#### Concept

SMOTENaN generates synthetic samples by interpolating between minority class samples and their k-nearest neighbors. The interpolation creates new samples along the line segments connecting existing samples in feature space.

#### How It Works

1. For each minority sample, find k nearest neighbors within the minority class
2. Randomly select one neighbor
3. Create a synthetic sample at a random point between the sample and neighbor
4. Handle NaN values according to the specified strategy

#### Distance Calculation

SMOTENaN uses a modified Euclidean distance that:
- Computes distance using only features where both samples have valid values
- Scales the result to compensate for reduced dimensionality
- Returns infinity when no valid feature pairs exist

#### When to Use SMOTENaN

- General-purpose oversampling with missing data
- When local data structure should be preserved
- Datasets with moderate missingness (up to ~30%)
- When you want synthetic samples to lie between existing samples

#### Key Parameters

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `neighbours` | Controls neighborhood size | 3-7 |
| `nan_handling` | NaN synthesis strategy | `'preserve_pattern'` |

### ADASYNNaN

**Adaptive Synthetic Sampling with NaN Handling**

#### Concept

ADASYNNaN extends SMOTE by adaptively determining how many synthetic samples to generate for each minority sample. Samples that are harder to learn (those surrounded by more majority class neighbors) receive more synthetic samples.

#### How It Works

1. For each minority sample, compute a density ratio based on majority class neighbors
2. Normalize density ratios to form a probability distribution
3. Allocate synthetic samples proportionally to density ratios
4. Generate samples using neighbor interpolation (like SMOTE)

#### Density Calculation

The density for sample *i* is:

```
density_i = (majority neighbors of i) / (total neighbors of i)
```

Higher density indicates the sample is in a region dominated by the majority class, making it harder to classify correctly.

#### When to Use ADASYNNaN

- When decision boundaries are complex or unclear
- Datasets with overlapping class distributions
- When you want to focus synthetic generation on difficult regions
- Problems where borderline cases are most important

#### Key Parameters

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `neighbours` | Neighborhood size for density and interpolation | 5-10 |
| `beta` | Controls total synthetic sample volume | 0.5-1.5 |
| `learning_rate` | Exponent for density weights | 1.0-2.0 |

### ROSENaN

**Random Over-Sampling Examples with NaN Handling**

#### Concept

ROSENaN generates synthetic samples by perturbing existing minority samples with Gaussian noise. Unlike SMOTE and ADASYN, it does not require neighbor search, making it computationally simpler and often faster.

#### How It Works

1. Compute per-feature bandwidth using Silverman's rule of thumb
2. For each synthetic sample needed:
   - Randomly select a seed sample from the minority class
   - Add Gaussian noise with the computed bandwidth to each feature
   - Handle NaN values according to the specified strategy

#### Bandwidth Computation

Silverman's rule provides an optimal bandwidth for kernel density estimation:

```
h_j = c * n^(-1/(d+4)) * sigma_j * shrinkage

where:
  c = (4/(d+2))^(1/(d+4))  (Silverman's constant)
  n = number of samples
  d = number of features
  sigma_j = standard deviation of feature j
```

#### When to Use ROSENaN

- High-dimensional data (many features)
- When neighbor search is computationally expensive
- Datasets with high missingness
- When you want samples that explore beyond existing data boundaries
- Faster processing is needed

#### Key Parameters

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `shrinkage` | Bandwidth scaling factor | 0.5-2.0 |
| `nan_handling` | NaN synthesis strategy | `'preserve_pattern'` |

### Algorithm Comparison

| Aspect | SMOTENaN | ADASYNNaN | ROSENaN |
|--------|-----------|------------|----------|
| **Synthetic Sample Location** | Between existing samples | Between existing samples | Around existing samples |
| **Neighbor Search** | Required | Required | Not required |
| **Computational Complexity** | O(n²) distance calculations | O(n²) distance calculations | O(n) per sample |
| **Adaptation** | Uniform across samples | Focuses on hard examples | Uniform across samples |
| **Sample Diversity** | Constrained to convex hull | Constrained to convex hull | Can exceed convex hull |
| **High Dimensionality** | Slower | Slower | Faster |
| **High Missingness** | May struggle with neighbors | May struggle with neighbors | Handles well |

---

## NaN Handling Strategies

The `nan_handling` parameter controls how missing values are treated when generating synthetic samples.

### preserve_pattern

**Most conservative strategy**

#### Behavior

A feature in the synthetic sample is NaN if **either** parent sample has NaN in that feature.

```
Parent A: [1.0, NaN, 3.0]
Parent B: [2.0, 4.0, NaN]
Synthetic: [1.5, NaN, NaN]  # NaN in positions 1 and 2
```

#### Rationale

- Missingness is meaningful and should not be artificially filled
- Synthetic samples should not claim to have information that was absent in originals
- Preserves the structural properties of the missingness pattern

#### Use When

- Missing values carry semantic meaning (e.g., sensor failure, non-response)
- Downstream models can handle NaN (e.g., XGBoost, LightGBM)
- You want to avoid introducing artificial information
- The missingness pattern itself may be predictive

#### Effect on Output

- NaN percentage in output >= NaN percentage in input
- Most NaN values preserved or increased

### interpolate

**Most aggressive strategy**

#### Behavior

- If both parents have values: interpolate
- If one parent has value, one has NaN: use the available value
- If both parents have NaN: result is NaN

```
Parent A: [1.0, NaN, 3.0]
Parent B: [2.0, 4.0, NaN]
Synthetic: [1.5, 4.0, 3.0]  # Values filled where possible
```

#### Rationale

- Maximizes information in synthetic samples
- Uses available data to fill gaps
- May improve model performance if missingness is random

#### Use When

- Missing values are believed to be random (MCAR)
- Downstream models struggle with NaN
- You want to minimize NaN in the resampled dataset
- Imputation is acceptable for your application

#### Effect on Output

- NaN percentage in output < NaN percentage in input
- Only features where both parents have NaN remain NaN

### random_pattern

**Probabilistic strategy**

#### Behavior

- If both parents have values: interpolate
- If both parents have NaN: result is NaN
- If one has value, one has NaN: randomly choose (50% chance each)

```
Parent A: [1.0, NaN, 3.0]
Parent B: [2.0, 4.0, NaN]
Synthetic: [1.5, NaN, 3.0]  # Position 1: randomly chose NaN
          # or [1.5, 4.0, NaN]  # Position 2: randomly chose NaN
```

#### Rationale

- Preserves the statistical distribution of missingness
- Neither too conservative nor too aggressive
- Maintains expected NaN rate across the dataset

#### Use When

- You want synthetic samples to statistically resemble originals
- Missingness pattern should be preserved in aggregate but not rigidly
- Uncertainty about the missingness mechanism

#### Effect on Output

- NaN percentage in output ≈ NaN percentage in input
- Stochastic variation in NaN placement

### Strategy Selection Guide

```
Is missingness meaningful in your domain?
├── Yes → preserve_pattern
│         (sensor failures, non-response, etc.)
│
└── No → Does your model handle NaN?
         ├── No → interpolate
         │        (need complete data for model)
         │
         └── Yes → Is missingness random (MCAR)?
                   ├── Yes → interpolate or random_pattern
                   └── No/Unknown → preserve_pattern or random_pattern
```

---

## Sampling Strategies

The `sampling_strategy` parameter controls the target class distribution after resampling.

### String Strategies

#### `'auto'` (default)

Resample all minority classes to match the majority class count.

```python
# Before: {0: 900, 1: 80, 2: 20}
# After:  {0: 900, 1: 900, 2: 900}
```

**Use when**: You want fully balanced classes.

#### `'minority'`

Resample only the smallest class to match the majority.

```python
# Before: {0: 900, 1: 80, 2: 20}
# After:  {0: 900, 1: 80, 2: 900}  # Only class 2 increased
```

**Use when**: Only the rarest class needs augmentation.

#### `'not majority'`

Same as `'auto'`. Resample everything except the largest class.

#### `'not minority'`

Resample all classes except the smallest to match the majority.

```python
# Before: {0: 900, 1: 80, 2: 20}
# After:  {0: 900, 1: 900, 2: 20}  # Class 2 unchanged
```

**Use when**: You want to preserve the rarest class as-is.

#### `'all'`

Resample all classes (including majority) to match the largest count.

```python
# Before: {0: 900, 1: 80, 2: 20}
# After:  {0: 900, 1: 900, 2: 900}  # Same as 'auto' in this case
```

**Use when**: Rarely needed; same effect as `'auto'` in binary classification.

### Ratio-Based Strategy

Pass a float to specify the target minority-to-majority ratio.

```python
# sampling_strategy=0.5
# Before: {0: 1000, 1: 100}
# After:  {0: 1000, 1: 500}  # 500/1000 = 0.5 ratio
```

**Use when**: Full balance is not desired; partial oversampling preferred.

### Dictionary Strategy

Pass a dictionary specifying exact target counts per class.

```python
# sampling_strategy={0: 1000, 1: 400, 2: 300}
# After: {0: 1000, 1: 400, 2: 300}  # Exact counts
```

**Use when**: You need precise control over final class sizes.

### Sampling Strategy Selection

| Goal | Strategy |
|------|----------|
| Fully balanced classes | `'auto'` or `'not majority'` |
| Partial balance (reduce ratio) | `0.5`, `0.8`, etc. |
| Exact class sizes | `{0: n0, 1: n1, ...}` |
| Only augment rarest class | `'minority'` |
| Preserve natural imbalance somewhat | `0.3` to `0.7` |

---

## Parameter Tuning

### neighbours Parameter

**Applies to**: SMOTENaN, ADASYNNaN

#### Effect of Value

| Value | Effect |
|-------|--------|
| Small (1-3) | Synthetic samples very close to originals; risk of overfitting |
| Medium (5-7) | Balanced exploration; typical default |
| Large (10+) | More diverse samples; may cross class boundaries |

#### Tuning Guidelines

1. **Start with default** (`neighbours=5`)
2. **Reduce if**: Few minority samples, overfitting observed, or many neighbors have infinite distance (high NaN)
3. **Increase if**: Large minority class, underfitting observed, want more sample diversity

#### Constraint

`neighbours` must be less than the number of minority samples:

```python
n_minority = np.sum(y == minority_class)
assert neighbours < n_minority, "Not enough minority samples"
```

### ADASYN Parameters

#### beta (Balance Level)

Controls the total volume of synthetic samples generated.

| Value | Effect |
|-------|--------|
| `beta < 1.0` | Generate fewer samples than needed for full balance |
| `beta = 1.0` | Generate enough for full balance (default) |
| `beta > 1.0` | Generate more than needed (over-balance) |

```python
# beta=0.5: Generate half the samples needed for balance
# If 100 samples needed for balance, generate ~50

# beta=1.5: Generate 50% more than balance
# If 100 samples needed for balance, generate ~150
```

**Tuning**: Start with 1.0; reduce if overfitting, increase if underfitting on minority.

#### learning_rate (Density Weight Exponent)

Controls how much to focus on hard-to-learn samples.

| Value | Effect |
|-------|--------|
| `learning_rate < 1.0` | Reduce focus on hard samples |
| `learning_rate = 1.0` | Linear weighting (default) |
| `learning_rate > 1.0` | Amplify focus on hard samples |

```python
# learning_rate=2.0: Square the density weights
# Hard samples get proportionally more synthetic samples
```

**Tuning**: Increase if model struggles with borderline cases; decrease if synthetic samples are too concentrated.

### ROSE shrinkage Parameter

Controls the bandwidth (spread) of the Gaussian noise added to samples.

| Value | Effect |
|-------|--------|
| `shrinkage = 0` | No noise; exact copies of seeds (random oversampling) |
| `shrinkage = 0.5` | Samples stay close to seeds |
| `shrinkage = 1.0` | Standard Silverman bandwidth (default) |
| `shrinkage > 1.0` | Samples spread further from seeds |

#### Visual Intuition

```
shrinkage = 0.5:    shrinkage = 1.0:    shrinkage = 2.0:
    *                   *                   *
   ***                *****               *******
  *****              *******             *********
   ***                *****               *******
    *                   *                   *
  (tight)            (normal)            (spread)
```

**Tuning**:
- Increase if model underfits (more exploration needed)
- Decrease if synthetic samples seem unrealistic or cause noise
- Use 0 to diagnose whether oversampling itself helps (vs. just random copies)

---

## Output Interpretation

### Understanding Resampled Data

The `fit_resample()` method returns:

```python
X_resampled, y_resampled = oversampler.fit_resample(X, y)
```

| Output | Type | Description |
|--------|------|-------------|
| `X_resampled` | `np.ndarray` or `pd.DataFrame` | Features including original and synthetic samples |
| `y_resampled` | `np.ndarray` or `pd.Series` | Labels including original and synthetic labels |

#### Sample Ordering

```python
# Original samples come first, synthetic samples appended
n_original = len(X)
n_resampled = len(X_resampled)
n_synthetic = n_resampled - n_original

X_original = X_resampled[:n_original]      # Original data
X_synthetic = X_resampled[n_original:]     # Synthetic data
```

### NaN Distribution in Output

#### Checking NaN Statistics

```python
# Overall NaN percentage
nan_pct_original = np.isnan(X).sum() / X.size * 100
nan_pct_resampled = np.isnan(X_resampled).sum() / X_resampled.size * 100

print(f"Original NaN: {nan_pct_original:.2f}%")
print(f"Resampled NaN: {nan_pct_resampled:.2f}%")

# Per-feature NaN counts
nan_per_feature = np.isnan(X_resampled).sum(axis=0)
```

#### Expected NaN Behavior by Strategy

| Strategy | Expected Output NaN % |
|----------|----------------------|
| `'preserve_pattern'` | >= Input NaN % |
| `'interpolate'` | < Input NaN % |
| `'random_pattern'` | ≈ Input NaN % |

### Quality Assessment

#### Class Distribution Verification

```python
from collections import Counter

print("Original:", Counter(y))
print("Resampled:", Counter(y_resampled))

# Verify balance
unique, counts = np.unique(y_resampled, return_counts=True)
imbalance_ratio = max(counts) / min(counts)
print(f"Imbalance ratio: {imbalance_ratio:.2f}")  # Should be ~1.0 if balanced
```

#### Synthetic Sample Inspection

```python
# Examine a few synthetic samples
n_original = len(X)
synthetic_samples = X_resampled[n_original:n_original+5]
print("Sample synthetic rows:")
print(synthetic_samples)

# Check they're not exact copies
for i, syn in enumerate(synthetic_samples):
    is_copy = any(np.allclose(syn, orig, equal_nan=True) for orig in X)
    print(f"Synthetic {i} is exact copy: {is_copy}")
```

---

## Method Selection Guide

### Decision Framework

```
START
  │
  ▼
Do you have high-dimensional data (>50 features)?
  │
  ├── Yes ──► ROSENaN (no neighbor search)
  │
  └── No
       │
       ▼
     Is the minority class distribution complex?
     (overlapping with majority, irregular boundaries)
       │
       ├── Yes ──► ADASYNNaN (adaptive focus on hard regions)
       │
       └── No ──► SMOTENaN (uniform interpolation)
```

### Quick Selection Table

| Scenario | Recommended Method |
|----------|-------------------|
| General purpose, moderate dimensions | SMOTENaN |
| Complex decision boundaries | ADASYNNaN |
| High-dimensional data (>50 features) | ROSENaN |
| High missingness (>30%) | ROSENaN |
| Need fast processing | ROSENaN |
| Want samples within data boundaries | SMOTENaN or ADASYNNaN |
| Want exploration beyond boundaries | ROSENaN |
| Borderline samples most important | ADASYNNaN |

### When Each Method Excels

**SMOTENaN**
- Well-separated classes
- Clear cluster structure in minority class
- Moderate number of features and samples

**ADASYNNaN**
- Overlapping class distributions
- Noisy decision boundaries
- When recall on difficult cases matters most

**ROSENaN**
- High-dimensional feature spaces
- Datasets with substantial missingness
- When computational efficiency matters
- When standard SMOTE overfits

---

## Usage Examples

### Basic Usage

```python
from overnan import OverNaN
import numpy as np

# Sample data with NaN
X = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 3.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
    [10.0, 11.0, 12.0]  # Minority sample
])
y = np.array([0, 0, 0, 0, 1])

# Basic SMOTE
oversampler = OverNaN(method='SMOTE', neighbours=2, random_state=42)
X_res, y_res = oversampler.fit_resample(X, y)

print(f"Original: {X.shape}, Resampled: {X_res.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
```

### High-Dimensional Data

```python
from overnan import OverNaN
import numpy as np

# High-dimensional data (100 features)
np.random.seed(42)
X = np.random.randn(1000, 100)
X[np.random.random(X.shape) < 0.1] = np.nan  # 10% NaN
y = np.array([0] * 900 + [1] * 100)  # 9:1 imbalance

# Use ROSE for efficiency with high dimensions
oversampler = OverNaN(
    method='ROSE',
    shrinkage=1.0,
    nan_handling='preserve_pattern',
    random_state=42
)
X_res, y_res = oversampler.fit_resample(X, y)

print(f"Features: {X.shape[1]}")
print(f"Resampled shape: {X_res.shape}")
```

### Preserving Missingness Structure

```python
from overnan import OverNaN
import numpy as np

# Data where NaN is meaningful (e.g., sensor readings)
X = np.array([
    [1.0, np.nan, 3.0],  # Sensor 2 failed
    [2.0, 4.0, np.nan],  # Sensor 3 failed
    [3.0, 5.0, 6.0],
    [4.0, 6.0, 7.0],
    [10.0, np.nan, 12.0]  # Minority: Sensor 2 failed
])
y = np.array([0, 0, 0, 0, 1])

# Preserve the failure pattern
oversampler = OverNaN(
    method='SMOTE',
    neighbours=2,
    nan_handling='preserve_pattern',  # Critical for meaningful NaN
    random_state=42
)
X_res, y_res = oversampler.fit_resample(X, y)

# Verify NaN pattern preserved
print("Synthetic samples NaN pattern:")
print(np.isnan(X_res[len(X):]))
```

### Aggressive NaN Reduction

```python
from overnan import OverNaN
import numpy as np

# Data where NaN should be filled if possible
X = np.array([
    [1.0, np.nan, 3.0],
    [2.0, 4.0, np.nan],
    [3.0, 5.0, 6.0],
    [4.0, 6.0, 7.0],
    [10.0, 11.0, 12.0]
])
y = np.array([0, 0, 0, 0, 1])

# Minimize NaN in output
oversampler = OverNaN(
    method='SMOTE',
    neighbours=2,
    nan_handling='interpolate',  # Fill NaN where possible
    random_state=42
)
X_res, y_res = oversampler.fit_resample(X, y)

nan_before = np.isnan(X).sum() / X.size * 100
nan_after = np.isnan(X_res).sum() / X_res.size * 100
print(f"NaN reduced from {nan_before:.1f}% to {nan_after:.1f}%")
```

### Partial Balancing

```python
from overnan import OverNaN
import numpy as np

# Severe imbalance: 1000:50
np.random.seed(42)
X = np.random.randn(1050, 10)
y = np.array([0] * 1000 + [1] * 50)

# Don't fully balance; use 0.5 ratio
oversampler = OverNaN(
    method='SMOTE',
    sampling_strategy=0.5,  # Target 50% of majority
    random_state=42
)
X_res, y_res = oversampler.fit_resample(X, y)

counts = dict(zip(*np.unique(y_res, return_counts=True)))
print(f"Final distribution: {counts}")
print(f"Ratio: {counts[1]/counts[0]:.2f}")  # Should be ~0.5
```

### Multi-Class Imbalance

```python
from overnan import OverNaN
import numpy as np

# Three-class problem with varying imbalance
np.random.seed(42)
X = np.random.randn(600, 10)
y = np.array([0] * 400 + [1] * 150 + [2] * 50)  # 400:150:50

# Balance all classes
oversampler = OverNaN(
    method='ADASYN',
    sampling_strategy='auto',  # All minorities to majority level
    random_state=42
)
X_res, y_res = oversampler.fit_resample(X, y)

print("Before:", dict(zip(*np.unique(y, return_counts=True))))
print("After:", dict(zip(*np.unique(y_res, return_counts=True))))
```

### Integration with Classifiers

```python
from overnan import OverNaN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

# Generate imbalanced data with NaN
np.random.seed(42)
X = np.random.randn(1000, 20)
X[np.random.random(X.shape) < 0.15] = np.nan
y = np.array([0] * 900 + [1] * 100)

# Split BEFORE oversampling (critical!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Oversample training data only
oversampler = OverNaN(method='SMOTE', random_state=42)
X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

# Train on resampled data
model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train_res, y_train_res)

# Evaluate on original (untouched) test data
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))
```

### Cross-Validation Pipeline

```python
from overnan import OverNaN
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb

np.random.seed(42)
X = np.random.randn(500, 15)
X[np.random.random(X.shape) < 0.2] = np.nan
y = np.array([0] * 400 + [1] * 100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Oversample within the fold
    oversampler = OverNaN(method='ROSE', random_state=42 + fold)
    X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
    
    # Train and evaluate
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    
    score = balanced_accuracy_score(y_test, y_pred)
    scores.append(score)
    print(f"Fold {fold+1}: {score:.3f}")

print(f"\nMean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

## Domain-Specific Considerations

### Financial Data

- Missing values often indicate data availability issues or reporting patterns
- Use `'preserve_pattern'` to maintain these signals
- Consider `ADASYNNaN` for fraud detection (focus on borderline cases)

### Sensor/IoT Data

- NaN typically means sensor failure or communication loss
- These patterns are highly informative; always use `'preserve_pattern'`
- `ROSENaN` works well with high-dimensional sensor arrays

### Survey Data

- Non-response is informative about respondent characteristics
- Use `'preserve_pattern'` or `'random_pattern'`
- Avoid `'interpolate'` which would fabricate responses

### Time Series Features

- Lagged features may have NaN at boundaries
- Consider whether boundary NaN is structural vs. informative
- `'interpolate'` may be appropriate for structural boundary NaN

---

## Limitations and Caveats

### General Limitations

1. **Not a replacement for good data**: Oversampling cannot create information that does not exist
2. **Overfitting risk**: Synthetic samples derived from limited originals may cause overfitting
3. **Evaluation must use original data**: Never evaluate on synthetic samples

### NaN-Specific Limitations

1. **Very high missingness**: When >50% of values are NaN, neighbor-based methods struggle
2. **All-NaN samples**: Samples with no valid features cannot participate in neighbor search
3. **Systematic missingness**: If missingness is related to the target, oversampling may amplify bias

### When Not to Use OverNaN

- When imputation is domain-appropriate and well-understood
- Very small minority classes (fewer than `neighbours` samples)
- When interpretability of individual samples matters (synthetic samples are artificial)

---

## References

### Algorithms

1. **SMOTE**: Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357. [DOI: 10.1613/jair.953](https://doi.org/10.1613/jair.953)

2. **ADASYN**: He, H., Bai, Y., Garcia, E.A., Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. *IEEE International Joint Conference on Neural Networks*, 1322-1328. [DOI: 10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969)

3. **ROSE**: Menardi, G. and Torelli, N. (2014). Training and assessing classification rules with imbalanced data. *Data Mining and Knowledge Discovery*, 28, 92-122. [DOI: 10.1007/s10618-012-0295-5](https://doi.org/10.1007/s10618-012-0295-5)

### Background

4. **Imbalanced Learning Survey**: He, H. and Garcia, E.A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284. [DOI: 10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239)

5. **Missing Data**: Little, R.J.A. and Rubin, D.B. (2019). *Statistical Analysis with Missing Data*, 3rd Edition. Wiley. [DOI: 10.1002/9781119482260](https://doi.org/10.1002/9781119482260)

---

*OverNaN: Oversampling for Imbalanced Learning with Missing Values*

Repository: [https://github.com/amaxiom/OverNaN](https://github.com/amaxiom/OverNaN)
