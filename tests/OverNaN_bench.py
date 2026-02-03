"""
OverNaN Benchmark Suite
=======================

Benchmarks OverNaN oversampling methods (SMOTE, ADASYN, ROSE) on publicly
available OpenML datasets with naturally occurring missing values.

Datasets are selected to be:
- Classification tasks with class imbalance
- Containing natural missing values (not artificially introduced)

The aim is to achieve similar or better classification performance, while preserving the missingness.
Metrics reported:
- Balanced Accuracy
- F1-Score (macro and minority class)
- ROC-AUC
- Geometric Mean (G-Mean)

References
----------
OpenML: https://www.openml.org/
Vanschoren, J., van Rijn, J.N., Bischl, B., Torgo, L. (2014).
OpenML: Networked Science in Machine Learning. SIGKDD Explorations, 15(2), 49-60.
DOI: 10.1145/2641190.2641198
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import gc

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    make_scorer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Classifiers that handle NaN natively
import xgboost as xgb

# OpenML for dataset access
try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    print("Warning: openml not installed. Install with: pip install openml")

# OverNaN imports
from overnan import OverNaN

warnings.filterwarnings('ignore')


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for an OpenML dataset."""
    openml_id: int
    name: str
    description: str
    domain: str


# Selected OpenML datasets with missing values (non-biomedical)
BENCHMARK_DATASETS = [
    DatasetConfig(
        openml_id=4,
        name="labor",
        description="Labor negotiations outcomes",
        domain="Labor/Economics"
    ),
    DatasetConfig(
        openml_id=6332,
        name="cylinder-bands",
        description="Process delay prerdiction",
        domain="Engineering"
    ),
    DatasetConfig(
        openml_id=40945,
        name="Titanic",
        description="Titanic survival prediction",
        domain="Transportation/Historical"
    ),
    DatasetConfig(
        openml_id=1023,
        name="soybean",
        description="Disease class prediction",
        domain="Environmental Science"
    ),
    DatasetConfig(
        openml_id=1017,
        name="car-evaluation",
        description="Car evaluation classification",
        domain="Consumer/Automotive"
    ),
]


# =============================================================================
# Custom Metrics
# =============================================================================

def geometric_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate geometric mean of class-wise recall (G-Mean).
    
    G-Mean is particularly useful for imbalanced datasets as it balances
    performance across all classes.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    
    Returns
    -------
    float
        Geometric mean score.
    """
    classes = np.unique(y_true)
    recalls = []
    
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            recall = np.sum((y_pred == cls) & mask) / np.sum(mask)
            recalls.append(recall)
    
    if len(recalls) == 0 or any(r == 0 for r in recalls):
        return 0.0
    
    return np.prod(recalls) ** (1.0 / len(recalls))


def minority_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1-score for the minority class.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    
    Returns
    -------
    float
        F1-score for minority class.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    
    # Binary F1 for minority class
    tp = np.sum((y_pred == minority_class) & (y_true == minority_class))
    fp = np.sum((y_pred == minority_class) & (y_true != minority_class))
    fn = np.sum((y_pred != minority_class) & (y_true == minority_class))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


# Create scorers for cross-validation
gmean_scorer = make_scorer(geometric_mean_score)
minority_f1_scorer = make_scorer(minority_f1_score)


# =============================================================================
# Standard ROSE Implementation (for Impute+ROSE comparison)
# =============================================================================

def standard_rose_resample(
    X: np.ndarray,
    y: np.ndarray,
    shrinkage: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard ROSE (Random Over-Sampling Examples) for complete data.
    
    Generates synthetic samples using kernel density estimation with
    Gaussian perturbation. This implementation requires complete data
    (no NaN values) and is used for the impute-then-oversample comparison.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (must not contain NaN).
    y : np.ndarray
        Target labels.
    shrinkage : float, default=1.0
        Bandwidth scaling factor.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    X_resampled : np.ndarray
        Resampled feature matrix.
    y_resampled : np.ndarray
        Resampled target labels.
    
    References
    ----------
    Menardi, G. and Torelli, N. (2014). Training and assessing classification
    rules with imbalanced data. Data Mining and Knowledge Discovery, 28, 92-122.
    DOI: 10.1007/s10618-012-0295-5
    """
    rng = np.random.RandomState(random_state)
    
    # Get class information
    classes, class_counts = np.unique(y, return_counts=True)
    max_count = np.max(class_counts)
    n_features = X.shape[1]
    
    # Compute Silverman's constant
    silverman_constant = (4.0 / (n_features + 2.0)) ** (1.0 / (n_features + 4.0))
    
    X_resampled = [X.copy()]
    y_resampled = [y.copy()]
    
    for cls, count in zip(classes, class_counts):
        if count >= max_count:
            continue
        
        n_synthetic = max_count - count
        X_class = X[y == cls]
        n_class = len(X_class)
        
        # Compute per-feature bandwidth using Silverman's rule
        n_factor = n_class ** (-1.0 / (n_features + 4.0))
        bandwidths = np.zeros(n_features)
        for j in range(n_features):
            sigma_j = np.std(X_class[:, j], ddof=1) if n_class > 1 else 0.0
            bandwidths[j] = silverman_constant * n_factor * sigma_j * shrinkage
        
        # Generate synthetic samples
        synthetic = np.zeros((n_synthetic, n_features))
        for i in range(n_synthetic):
            # Random seed sample from minority class
            seed_idx = rng.randint(0, n_class)
            seed = X_class[seed_idx].copy()
            
            # Add Gaussian perturbation to each feature
            for j in range(n_features):
                if bandwidths[j] > 0:
                    synthetic[i, j] = seed[j] + rng.normal(0, bandwidths[j])
                else:
                    synthetic[i, j] = seed[j]
        
        X_resampled.append(synthetic)
        y_resampled.append(np.full(n_synthetic, cls))
    
    return np.vstack(X_resampled), np.concatenate(y_resampled)


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_openml_dataset(dataset_config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a dataset from OpenML.
    
    Parameters
    ----------
    dataset_config : DatasetConfig
        Configuration for the dataset to load.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix with NaN for missing values.
    y : np.ndarray
        Target labels (encoded as integers).
    info : dict
        Dataset information including shape, class distribution, NaN statistics.
    """
    if not OPENML_AVAILABLE:
        raise ImportError("openml package required. Install with: pip install openml")
    
    print(f"  Loading {dataset_config.name} (OpenML ID: {dataset_config.openml_id})...")
    
    # Fetch dataset from OpenML
    dataset = openml.datasets.get_dataset(
        dataset_config.openml_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False
    )
    
    # Get data as DataFrame to handle mixed types
    X_df, y_series, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    
    # Encode categorical features as numeric
    X = X_df.copy()
    
    for col in X.columns:
        # Convert categorical to object first to allow modifications
        if X[col].dtype.name == 'category':
            X[col] = X[col].astype(object)
        
        if X[col].dtype == 'object':
            # Encode categorical as numeric, preserving NaN
            le = LabelEncoder()
            mask = X[col].notna()
            if mask.any():
                encoded_values = le.fit_transform(X.loc[mask, col].astype(str))
                # Create a new numeric column
                X[col] = np.nan  # Reset to NaN
                X.loc[mask, col] = encoded_values
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.values.astype(np.float64)
    
    # Encode target
    le_y = LabelEncoder()
    # Convert categorical to object if needed
    if hasattr(y_series, 'cat'):
        y_series = y_series.astype(object)
    y = le_y.fit_transform(y_series.fillna('__MISSING__').astype(str))
    
    # Calculate dataset statistics
    n_samples, n_features = X.shape
    classes, class_counts = np.unique(y, return_counts=True)
    imbalance_ratio = max(class_counts) / min(class_counts)
    
    nan_count = np.isnan(X).sum()
    nan_percentage = nan_count / X.size * 100
    nan_features = np.sum(np.any(np.isnan(X), axis=0))
    nan_samples = np.sum(np.any(np.isnan(X), axis=1))
    
    info = {
        'name': dataset_config.name,
        'domain': dataset_config.domain,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': len(classes),
        'class_distribution': dict(zip(classes.tolist(), class_counts.tolist())),
        'imbalance_ratio': imbalance_ratio,
        'nan_count': nan_count,
        'nan_percentage': nan_percentage,
        'nan_features': nan_features,
        'nan_samples': nan_samples,
    }
    
    return X, y, info


def print_dataset_info(info: dict) -> None:
    """Print dataset information in a formatted way."""
    print(f"\n  Dataset: {info['name']} ({info['domain']})")
    print(f"    Samples: {info['n_samples']}, Features: {info['n_features']}, Classes: {info['n_classes']}")
    print(f"    Class distribution: {info['class_distribution']}")
    print(f"    Imbalance ratio: {info['imbalance_ratio']:.2f}")
    print(f"    Missing values: {info['nan_count']} ({info['nan_percentage']:.2f}%)")
    print(f"    Features with NaN: {info['nan_features']}, Samples with NaN: {info['nan_samples']}")


# =============================================================================
# Benchmark Functions
# =============================================================================

def run_single_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    method: Optional[str],
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Run benchmark for a single oversampling method.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    method : str or None
        Oversampling method ('SMOTE', 'ADASYN', 'ROSE') or None for baseline.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary of metric scores (mean and std).
    """
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage for fold results
    results = {
        'balanced_accuracy': [],
        'f1_macro': [],
        'f1_minority': [],
        'gmean': [],
        'roc_auc': [],
        'time': []
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_start = time.time()
        
        # Apply oversampling if method specified
        if method is not None:
            oversampler = OverNaN(
                method=method,
                neighbours=5 if method in ['SMOTE', 'ADASYN'] else None,
                random_state=random_state + fold_idx
            )
            try:
                X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"      Warning: {method} failed on fold {fold_idx}: {e}")
                # Fall back to original data
                X_train_res, y_train_res = X_train, y_train
        else:
            X_train_res, y_train_res = X_train, y_train
        
        # Train XGBoost (handles NaN natively)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        model.fit(X_train_res, y_train_res)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        fold_time = time.time() - fold_start
        
        # Calculate metrics
        results['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
        results['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
        results['f1_minority'].append(minority_f1_score(y_test, y_pred))
        results['gmean'].append(geometric_mean_score(y_test, y_pred))
        
        # ROC-AUC (handle multiclass)
        n_classes = len(np.unique(y))
        if n_classes == 2:
            results['roc_auc'].append(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            try:
                results['roc_auc'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
            except ValueError:
                results['roc_auc'].append(np.nan)
        
        results['time'].append(fold_time)
    
    # Aggregate results
    aggregated = {}
    for metric, values in results.items():
        values = np.array(values)
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            aggregated[f'{metric}_mean'] = np.mean(valid_values)
            aggregated[f'{metric}_std'] = np.std(valid_values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_std'] = np.nan
    
    return aggregated


def run_dataset_benchmark(
    dataset_config: DatasetConfig,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run full benchmark on a single dataset.
    
    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Results for all methods on this dataset.
    """
    # Load dataset
    try:
        X, y, info = load_openml_dataset(dataset_config)
    except Exception as e:
        print(f"  Error loading {dataset_config.name}: {e}")
        return None
    
    print_dataset_info(info)
    
    # Skip if no missing values
    if info['nan_percentage'] == 0:
        print("  Skipping: No missing values in dataset")
        return None
    
    # Methods to benchmark
    methods = [None, 'SMOTE', 'ADASYN', 'ROSE']
    method_names = ['Baseline', 'SMOTENaN', 'ADASYNNaN', 'ROSENaN']
    
    results_list = []
    
    for method, method_name in zip(methods, method_names):
        print(f"    Running {method_name}...", end=' ')
        
        start_time = time.time()
        result = run_single_benchmark(X, y, method, n_splits, random_state)
        total_time = time.time() - start_time
        
        result['method'] = method_name
        result['dataset'] = dataset_config.name
        result['domain'] = dataset_config.domain
        result['n_samples'] = info['n_samples']
        result['n_features'] = info['n_features']
        result['imbalance_ratio'] = info['imbalance_ratio']
        result['nan_percentage'] = info['nan_percentage']
        result['total_time'] = total_time
        
        results_list.append(result)
        
        print(f"done ({total_time:.1f}s) - Bal.Acc: {result['balanced_accuracy_mean']:.3f}")
    
    # Memory cleanup
    del X, y
    gc.collect()
    
    return pd.DataFrame(results_list)


def run_full_benchmark(
    datasets: List[DatasetConfig] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run benchmark on all specified datasets.
    
    Parameters
    ----------
    datasets : list of DatasetConfig, optional
        Datasets to benchmark. Defaults to BENCHMARK_DATASETS.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Combined results for all datasets and methods.
    """
    if datasets is None:
        datasets = BENCHMARK_DATASETS
    
    print("=" * 70)
    print("OverNaN Benchmark Suite")
    print("=" * 70)
    print(f"Datasets: {len(datasets)}")
    print(f"Methods: Baseline, SMOTENaN, ADASYNNaN, ROSENaN")
    print(f"Cross-validation: {n_splits}-fold stratified")
    print(f"Random state: {random_state}")
    print("=" * 70)
    
    all_results = []
    
    for i, dataset_config in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] Processing {dataset_config.name}...")
        
        result_df = run_dataset_benchmark(dataset_config, n_splits, random_state)
        
        if result_df is not None:
            all_results.append(result_df)
    
    if not all_results:
        print("\nNo results collected. Check dataset availability.")
        return pd.DataFrame()
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    return combined_results


def print_summary_table(results: pd.DataFrame) -> None:
    """
    Print a summary table of benchmark results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Benchmark results from run_full_benchmark.
    """
    if results.empty:
        print("No results to display.")
        return
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Group by dataset
    datasets = results['dataset'].unique()
    
    for dataset in datasets:
        df = results[results['dataset'] == dataset]
        
        print(f"\n{dataset} (NaN: {df['nan_percentage'].iloc[0]:.1f}%, IR: {df['imbalance_ratio'].iloc[0]:.1f})")
        print("-" * 60)
        print(f"{'Method':<12} {'Bal.Acc':>10} {'F1-Macro':>10} {'F1-Min':>10} {'G-Mean':>10} {'ROC-AUC':>10}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            method = row['method']
            ba = f"{row['balanced_accuracy_mean']:.3f}±{row['balanced_accuracy_std']:.3f}"
            f1m = f"{row['f1_macro_mean']:.3f}±{row['f1_macro_std']:.3f}"
            f1min = f"{row['f1_minority_mean']:.3f}±{row['f1_minority_std']:.3f}"
            gm = f"{row['gmean_mean']:.3f}±{row['gmean_std']:.3f}"
            auc = f"{row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}" if not np.isnan(row['roc_auc_mean']) else "N/A"
            
            print(f"{method:<12} {ba:>10} {f1m:>10} {f1min:>10} {gm:>10} {auc:>10}")
    
    # Overall comparison
    print("\n" + "=" * 70)
    print("OVERALL AVERAGE ACROSS DATASETS")
    print("=" * 70)
    
    overall = results.groupby('method').agg({
        'balanced_accuracy_mean': 'mean',
        'f1_macro_mean': 'mean',
        'f1_minority_mean': 'mean',
        'gmean_mean': 'mean',
        'roc_auc_mean': 'mean',
        'total_time': 'mean'
    }).round(4)
    
    print(f"\n{'Method':<12} {'Bal.Acc':>10} {'F1-Macro':>10} {'F1-Min':>10} {'G-Mean':>10} {'ROC-AUC':>10} {'Time(s)':>10}")
    print("-" * 72)
    
    for method in ['Baseline', 'SMOTENaN', 'ADASYNNaN', 'ROSENaN']:
        if method in overall.index:
            row = overall.loc[method]
            print(f"{method:<12} {row['balanced_accuracy_mean']:>10.4f} {row['f1_macro_mean']:>10.4f} "
                  f"{row['f1_minority_mean']:>10.4f} {row['gmean_mean']:>10.4f} "
                  f"{row['roc_auc_mean']:>10.4f} {row['total_time']:>10.1f}")
    
    # Improvement over baseline
    print("\n" + "-" * 70)
    print("IMPROVEMENT OVER BASELINE (percentage points)")
    print("-" * 70)
    
    baseline = overall.loc['Baseline'] if 'Baseline' in overall.index else None
    if baseline is not None:
        print(f"\n{'Method':<12} {'Bal.Acc':>10} {'F1-Macro':>10} {'F1-Min':>10} {'G-Mean':>10}")
        print("-" * 52)
        
        for method in ['SMOTENaN', 'ADASYNNaN', 'ROSENaN']:
            if method in overall.index:
                row = overall.loc[method]
                ba_imp = (row['balanced_accuracy_mean'] - baseline['balanced_accuracy_mean']) * 100
                f1m_imp = (row['f1_macro_mean'] - baseline['f1_macro_mean']) * 100
                f1min_imp = (row['f1_minority_mean'] - baseline['f1_minority_mean']) * 100
                gm_imp = (row['gmean_mean'] - baseline['gmean_mean']) * 100
                
                print(f"{method:<12} {ba_imp:>+10.2f} {f1m_imp:>+10.2f} {f1min_imp:>+10.2f} {gm_imp:>+10.2f}")


def save_results(results: pd.DataFrame, filepath: str = "overnan_benchmark_results.csv") -> None:
    """
    Save benchmark results to CSV.
    
    Parameters
    ----------
    results : pd.DataFrame
        Benchmark results.
    filepath : str
        Output file path.
    """
    results.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")


# =============================================================================
# Comparison with Impute-Then-Oversample Approach
# =============================================================================

def run_impute_comparison(
    dataset_config: DatasetConfig,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare NaN-aware oversampling vs impute-then-oversample approach.
    
    This demonstrates the value of OverNaN over the naive approach of
    imputing missing values before applying standard oversampling.
    
    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Comparison results.
    """
    from imblearn.over_sampling import SMOTE as ImblearnSMOTE, ADASYN as ImblearnADASYN
    
    # Load dataset
    try:
        X, y, info = load_openml_dataset(dataset_config)
    except Exception as e:
        print(f"  Error loading {dataset_config.name}: {e}")
        return None
    
    print_dataset_info(info)
    
    if info['nan_percentage'] == 0:
        print("  Skipping: No missing values")
        return None
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Define all approaches to compare
    # Format: (name, method, use_impute)
    approaches = [
        ('Baseline', None, False),
        ('Impute+SMOTE', 'SMOTE', True),
        ('Impute+ADASYN', 'ADASYN', True),
        ('Impute+ROSE', 'ROSE', True),
        ('OverNaN-SMOTE', 'SMOTE', False),
        ('OverNaN-ADASYN', 'ADASYN', False),
        ('OverNaN-ROSE', 'ROSE', False),
    ]
    
    results_list = []
    
    for approach_name, method, use_impute in approaches:
        print(f"    Running {approach_name}...", end=' ')
        
        ba_scores = []
        f1_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
            y_train, y_test = y[train_idx], y[test_idx]
            
            if use_impute and method is not None:
                # Impute-then-oversample approach
                imputer = SimpleImputer(strategy='mean')
                X_train_imp = imputer.fit_transform(X_train)
                X_test_imp = imputer.transform(X_test)
                
                try:
                    if method == 'SMOTE':
                        oversampler = ImblearnSMOTE(random_state=random_state + fold_idx)
                        X_train_res, y_train_res = oversampler.fit_resample(X_train_imp, y_train)
                    elif method == 'ADASYN':
                        oversampler = ImblearnADASYN(random_state=random_state + fold_idx)
                        X_train_res, y_train_res = oversampler.fit_resample(X_train_imp, y_train)
                    elif method == 'ROSE':
                        # Use standard ROSE implementation (no imblearn equivalent)
                        X_train_res, y_train_res = standard_rose_resample(
                            X_train_imp, y_train,
                            shrinkage=1.0,
                            random_state=random_state + fold_idx
                        )
                except Exception:
                    X_train_res, y_train_res = X_train_imp, y_train
                
                X_test_final = X_test_imp
                
            elif method is not None:
                # OverNaN approach (NaN-aware)
                oversampler = OverNaN(
                    method=method,
                    neighbours=5 if method in ['SMOTE', 'ADASYN'] else None,
                    random_state=random_state + fold_idx
                )
                try:
                    X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
                except Exception:
                    X_train_res, y_train_res = X_train, y_train
                
                X_test_final = X_test
            else:
                # Baseline
                X_train_res, y_train_res = X_train, y_train
                X_test_final = X_test
            
            # Train and evaluate
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=random_state,
                eval_metric='logloss', verbosity=0
            )
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_final)
            
            ba_scores.append(balanced_accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        
        result = {
            'approach': approach_name,
            'dataset': dataset_config.name,
            'balanced_accuracy_mean': np.mean(ba_scores),
            'balanced_accuracy_std': np.std(ba_scores),
            'f1_macro_mean': np.mean(f1_scores),
            'f1_macro_std': np.std(f1_scores),
        }
        results_list.append(result)
        
        print(f"Bal.Acc: {result['balanced_accuracy_mean']:.3f}")
    
    return pd.DataFrame(results_list)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    
    if not OPENML_AVAILABLE:
        print("Error: openml package is required.")
        print("Install with: pip install openml")
        return
    
    # Run full benchmark
    results = run_full_benchmark(
        datasets=BENCHMARK_DATASETS,
        n_splits=5,
        random_state=42
    )
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    if not results.empty:
        save_results(results, "overnan_benchmark_results.csv")
    
    # Run impute comparison on Titanic dataset (good mix of missing values)
    print("\n" + "=" * 70)
    print("COMPARISON: OverNaN vs Impute-Then-Oversample")
    print("=" * 70)
    
    try:
        from imblearn.over_sampling import SMOTE as ImblearnSMOTE
        
        # Run comparison on Titanic dataset
        comparison_dataset = DatasetConfig(
            openml_id=989,
            name="anneal",
            description="Steel grade prediction",
            domain="Materials Science"
        )
        
        comparison_results = run_impute_comparison(
            comparison_dataset,
            n_splits=5,
            random_state=42
        )
        
        if comparison_results is not None:
            print("\n  Comparison Results (anneal dataset):")
            print("  " + "-" * 60)
            print(f"  {'Approach':<20} {'Bal.Acc':>15} {'F1-Macro':>15}")
            print("  " + "-" * 60)
            for _, row in comparison_results.iterrows():
                ba = f"{row['balanced_accuracy_mean']:.3f}±{row['balanced_accuracy_std']:.3f}"
                f1 = f"{row['f1_macro_mean']:.3f}±{row['f1_macro_std']:.3f}"
                print(f"  {row['approach']:<20} {ba:>15} {f1:>15}")
    
    except ImportError:
        print("\n  Note: Install imbalanced-learn for comparison with standard SMOTE/ADASYN/ROSE")
        print("  pip install imbalanced-learn")
    
    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()