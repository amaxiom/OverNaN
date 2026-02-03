"""
setup.py
--------
Setup configuration for the OverNaN package.
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="overnan",
    version="0.2.0",
    author="Amanda Barnard",
    author_email="amanda.s.barnard@anu.edu.au",
    description="Oversampling for imbalanced learning with missing values (SMOTE, ADASYN, ROSE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amaxiom/OverNaN",
    py_modules=["overnan"],  # Single-file module
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "examples": [
            "xgboost>=1.4.0",
            "matplotlib>=3.3.0",
        ],
    },
)


# =============================================================================
# README.md Content
# =============================================================================

README_CONTENT = """
# OverNaN: Oversampling for Imbalanced Learning with Missing Values

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OverNaN is a Python package that implements oversampling algorithms (SMOTE, ADASYN, and ROSE) specifically designed to handle datasets with missing values (NaN). Traditional oversampling methods often fail or produce poor results when faced with missing data. OverNaN addresses this limitation by incorporating intelligent NaN-aware distance calculations and synthesis strategies.

## Features

- **Three NaN-Aware Algorithms**: SMOTE, ADASYN, and ROSE implementations that properly handle missing values
- **Multiple NaN Handling Strategies**: Choose how synthetic samples handle missingness patterns
- **Parallel Processing**: Speed up resampling with built-in parallel processing support
- **Scikit-learn Compatible**: Follows scikit-learn's transformer interface
- **Pandas Integration**: Seamlessly works with pandas DataFrames and Series
- **Flexible Sampling Strategies**: Multiple options for controlling the resampling ratio

## Installation

```bash
pip install overnan
```

Or install from source:

```bash
git clone https://github.com/amaxiom/OverNaN.git
cd overnan
pip install -e .
```

## Quick Start

```python
from overnan import OverNaN
import numpy as np

# Create imbalanced data with missing values
X = np.array([[1, 2, np.nan], 
              [3, np.nan, 4], 
              [5, 6, 7], 
              [8, 9, 10],
              [11, 12, 13]])
y = np.array([0, 0, 0, 0, 1])  # Imbalanced: 4 samples of class 0, 1 sample of class 1

# Apply SMOTE with NaN handling
oversampler = OverNaN(method='SMOTE', neighbours=2, random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

print(f"Original shape: {X.shape}")
print(f"Resampled shape: {X_resampled.shape}")
print(f"Class distribution: {np.bincount(y_resampled.astype(int))}")
```

## Available Methods

### SMOTE (Synthetic Minority Over-sampling Technique)

Generates synthetic samples by interpolating between minority class samples and their nearest neighbors.

```python
oversampler = OverNaN(
    method='SMOTE',
    neighbours=5,                      # Number of nearest neighbors
    sampling_strategy='auto',          # Balance all minority classes
    nan_handling='preserve_pattern',   # Preserve NaN patterns in synthetic samples
    random_state=42,
    n_jobs=-1                          # Use all CPU cores
)
```

**Reference**: Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357. DOI: 10.1613/jair.953

### ADASYN (Adaptive Synthetic Sampling)

Generates more synthetic samples for minority instances that are harder to learn (those with more majority class neighbors).

```python
oversampler = OverNaN(
    method='ADASYN',
    neighbours=5,
    beta=1.0,           # Balance level (1.0 = fully balanced)
    learning_rate=1.0,  # Adaptation power for hard examples
    nan_handling='interpolate'
)
```

**Reference**: He, H., Bai, Y., Garcia, E.A., Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. IEEE IJCNN, 1322-1328. DOI: 10.1109/IJCNN.2008.4633969

### ROSE (Random Over-Sampling Examples)

Generates synthetic samples using kernel density estimation with Gaussian perturbation. No neighbor search required, making it efficient for high-dimensional data.

```python
oversampler = OverNaN(
    method='ROSE',
    shrinkage=1.0,      # Bandwidth scaling (1.0 = Silverman's rule)
    nan_handling='preserve_pattern',
    random_state=42
)
```

**Reference**: Menardi, G. and Torelli, N. (2014). Training and assessing classification rules with imbalanced data. Data Mining and Knowledge Discovery, 28, 92-122. DOI: 10.1007/s10618-012-0295-5

## NaN Handling Strategies

OverNaN provides three strategies for handling missing values in synthetic samples:

| Strategy | Description |
|----------|-------------|
| `preserve_pattern` | Preserves NaN conservatively (NaN if either parent has NaN) |
| `interpolate` | Uses available values when possible |
| `random_pattern` | Randomly chooses NaN pattern based on class-level rates |

```python
# Compare different NaN handling strategies
for strategy in ['preserve_pattern', 'interpolate', 'random_pattern']:
    oversampler = OverNaN(method='SMOTE', nan_handling=strategy, random_state=42)
    X_res, y_res = oversampler.fit_resample(X, y)
    print(f"{strategy}: NaN percentage = {np.isnan(X_res).mean():.2%}")
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

## Integration with Machine Learning Pipelines

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Apply oversampling to training data only
oversampler = OverNaN(method='ROSE', shrinkage=1.0, random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

# Train classifier (XGBoost handles NaN naturally)
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate on original test set
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")
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

## Testing

Run the test suite:

```bash
python test_overnan.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use OverNaN in your research, please cite:

```bibtex
@software{overnan2024,
  title={OverNaN: Oversampling for Imbalanced Learning with Missing Values},
  author={Barnard, Amanda},
  year={2024},
  url={https://github.com/amaxiom/OverNaN}
}
```

## References

- Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357. DOI: 10.1613/jair.953
- He, H., Bai, Y., Garcia, E.A., Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. IEEE IJCNN, 1322-1328. DOI: 10.1109/IJCNN.2008.4633969
- Menardi, G. and Torelli, N. (2014). Training and assessing classification rules with imbalanced data. DMKD, 28, 92-122. DOI: 10.1007/s10618-012-0295-5

## Acknowledgments

- Inspired by the imbalanced-learn library
- Based on the original SMOTE, ADASYN, and ROSE papers
- Designed for real-world biotech and health data applications

## Contact

For questions and feedback, please open an issue on GitHub.
"""

# Write README.md file when this script is run directly
if __name__ == "__main__":
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(README_CONTENT.strip())
    print("README.md written successfully.")
