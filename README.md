# OverNaN

**Oversampling for Imbalanced Learning with Missing Values**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

There are lots of reasons why many datasets contain missing values, particularly when dealing with real-world data. Missingness can be resolved using imputation, but the addition of synthetic instances can bias models and give overly optimistic training performance metrics and poor generalisability. Imputation also masks the hidden information about the missing mechanisms, since missingness can also be meaningful. To help eliminate the need for imputation and better prepare models for real-world validation, OverNaN implements NaN-aware oversampling algorithms for handling class imbalance in datasets with missing values. OverNaN preserves missingness patterns while generating synthetic minority class samples, and is the perfect partner to learning algorithms that natively handle NaNs. The aim is to achieve similar or better classification performance compared to imputation, while preserving the missingness and only using real data.

## Features

- **Three NaN-Aware Algorithms**: SMOTE, ADASYN, and ROSE with native missing value support
- **Flexible NaN Handling**: Choose how synthetic samples inherit missingness patterns
- **Scikit-learn Compatible**: Familiar `fit_resample()` interface
- **Pandas Integration**: Preserves DataFrame column names and Series names
- **Parallel Processing**: Joblib-based parallelization for large datasets
- **Cross-Platform**: Windows, Linux, and macOS compatible

## Installation

```bash
# Clone the repository
git clone https://github.com/amaxiom/OverNaN.git
cd OverNaN

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```python
from overnan import OverNaN
import numpy as np

# Create imbalanced data with missing values
X = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 3.0],
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
    [10.0, 11.0, 12.0]
])
y = np.array([0, 0, 0, 0, 1])  # 4:1 imbalance

# Resample with SMOTE
oversampler = OverNaN(method='SMOTE', neighbours=2, random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

print(f"Before: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"After:  {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
```

## Available Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **SMOTE** | Interpolates between minority samples and neighbors | General purpose |
| **ADASYN** | Adaptive interpolation focusing on hard samples | Complex boundaries |
| **ROSE** | Kernel density estimation with Gaussian perturbation | High dimensions |

```python
# SMOTE: Neighbor-based interpolation
OverNaN(method='SMOTE', neighbours=5)

# ADASYN: Adaptive synthetic sampling
OverNaN(method='ADASYN', neighbours=5, beta=1.0, learning_rate=1.0)

# ROSE: Kernel density estimation (no neighbors required)
OverNaN(method='ROSE', shrinkage=1.0)
```

## NaN Handling Strategies

| Strategy | Behavior |
|----------|----------|
| `'preserve_pattern'` | NaN if either parent has NaN (default, conservative) |
| `'interpolate'` | Use available values; minimizes NaN in output |
| `'random_pattern'` | Probabilistically preserve NaN based on class rates |

```python
# Preserve missingness structure
OverNaN(method='SMOTE', nan_handling='preserve_pattern')

# Minimize NaN in output
OverNaN(method='SMOTE', nan_handling='interpolate')
```

## Sampling Strategies

Control how much to oversample each class:

```python
# Balance to match majority class (default)
oversampler = OverNaN(sampling_strategy='auto')

# Only oversample the minority class
oversampler = OverNaN(sampling_strategy='minority')

# Oversample to 80% of majority class size
oversampler = OverNaN(sampling_strategy=0.8)

# Specify exact counts per class
oversampler = OverNaN(sampling_strategy={0: 100, 1: 100})
```

## Integration with XGBoost

XGBoost handles NaN natively, making it ideal for use with OverNaN:

```python
from overnan import OverNaN
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Split before oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Oversample training data only
oversampler = OverNaN(method='ROSE', random_state=42)
X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

# Train and evaluate
model = xgb.XGBClassifier()
model.fit(X_train_res, y_train_res)
accuracy = model.score(X_test, y_test)
```

## Direct Class Usage

```python
from overnan import SMOTENaN, ADASYNNaN, ROSENaN

# Direct SMOTE-NaN usage
smote = SMOTENaN(neighbours=5, nan_handling='preserve_pattern', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Direct ADASYN-NaN usage
adasyn = ADASYNNaN(neighbours=5, beta=1.0, learning_rate=1.5, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Direct ROSE-NaN usage
rose = ROSENaN(shrinkage=1.0, nan_handling='interpolate', random_state=42)
X_resampled, y_resampled = rose.fit_resample(X, y)
```

## How It Works

### NaN-Aware Distance Calculation (SMOTE and ADASYN)

Distance is calculated using only the features that are non-NaN in both samples, then scaled by the proportion of valid features:

```
distance = sqrt(sum((x1[valid] - x2[valid])^2) * (n_total / n_valid))
```

### ROSE Kernel Density Estimation

ROSE generates synthetic samples by perturbing seed samples with Gaussian noise:

```
x_new = x_seed + N(0, H)
```

where H is computed using Silverman's rule of thumb, scaled by the `shrinkage` parameter.

## Performance Considerations

- **Parallel Processing**: Enable with `n_jobs=-1` for large datasets
- **ROSE for High Dimensions**: ROSE does not require neighbor search, making it more efficient for high-dimensional data
- **Memory Usage**: Synthetic samples are generated in batches to manage memory

## Documentation

| Document | Description |
|----------|-------------|
| [Interpretation Guide](docs/OverNaN_interpretation.md) | Methods, parameters, usage examples |
| [Computation Guide](docs/OverNaN_computation.md) | Implementation, parallelization, memory |
| [Testing Guide](tests/OverNaN_testing.md) | Test suite and benchmarking |

## Project Structure

```
OverNaN/
├── overnan.py                                   # Main module (OverNaN, SMOTENaN, ADASYNNaN, ROSENaN)
├── tests/
│   ├── OverNaN_testing.md                       # Testing documentation
│   ├── OverNaN_testing.ipynb                    # Testing notebook
│   ├── OverNaN_test.py                          # Unit and integration tests
│   └── OverNaN_bench.py                         # OpenML benchmarks
├── docs/
│   ├── OverNaN_interpretation.md                # Methods and usage guide
│   └── OverNaN_computation.md                   # Technical implementation guide
├── examples/
│   ├── OverNaN_example_synthetic.ipynb          # Basic example notebook
│   ├── overnan_synthetic_comparison.png         # Basic example output
│   ├── OverNaN_example_grapheneoxide.ipynb      # Graphene Oxide example notebook
│   ├── overnan_grapheneoxide_comparisons.png    # Graphene Oxide example output
│   └── grapheneoxide.csv                        # Graphene Oxide input
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python >= 3.8
- numpy >= 1.19.0
- pandas >= 1.1.0
- scikit-learn >= 0.24.0
- joblib >= 1.0.0

Optional for benchmarking:
- xgboost >= 1.4.0
- openml >= 0.12.0
- imbalanced-learn >= 0.8.0

## Running Tests

```bash
# Run test suite
python tests/OverNaN_test.py

# Run benchmarks (requires openml)
python tests/OverNaN_bench.py
```

## References

1. **SMOTE**: Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357. [DOI: 10.1613/jair.953](https://doi.org/10.1613/jair.953)

2. **ADASYN**: He, H., Bai, Y., Garcia, E.A., Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. *IEEE IJCNN*, 1322-1328. [DOI: 10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969)

3. **ROSE**: Menardi, G. and Torelli, N. (2014). Training and assessing classification rules with imbalanced data. *DMKD*, 28, 92-122. [DOI: 10.1007/s10618-012-0295-5](https://doi.org/10.1007/s10618-012-0295-5)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use OverNaN in your research, please cite:

```bibtex
@software{overnan2026,
  author = {Barnard, Amanda S.},
  title = {OverNaN: Oversampling for Imbalanced Learning with Missing Values},
  year = {2026},
  url = {https://github.com/amaxiom/OverNaN},
  version = {0.2}
}
```

**Ready to preserve data integrity during imbalanced learning?**

```bash
pip install overnan
```

## Author

Amanda Barnard ([@amaxiom](https://github.com/amaxiom))
