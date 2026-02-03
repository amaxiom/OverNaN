"""
Test suite for the OverNaN package.

Tests SMOTE, ADASYN, and ROSE algorithms with NaN-aware oversampling.
Includes unit tests, integration tests with classifiers, and edge case handling.
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

warnings.filterwarnings('ignore')

# Import the OverNaN package
from overnan import OverNaN, SMOTENaN, ADASYNNaN, ROSENaN


# =============================================================================
# Test Data Generation
# =============================================================================

def create_imbalanced_data_with_nan(n_majority=500, n_minority=50, n_features=10, 
                                    nan_percentage=0.2, random_state=42):
    """
    Create an imbalanced dataset with missing values.
    
    Parameters
    ----------
    n_majority : int, default=500
        Number of majority class samples.
    n_minority : int, default=50
        Number of minority class samples.
    n_features : int, default=10
        Number of features.
    nan_percentage : float, default=0.2
        Proportion of values to set as NaN.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix with NaN values.
    y : np.ndarray
        Target labels (0 for majority, 1 for minority).
    """
    np.random.seed(random_state)
    
    # Generate majority class centered at origin
    X_majority = np.random.randn(n_majority, n_features)
    
    # Generate minority class shifted by +2 in all dimensions
    X_minority = np.random.randn(n_minority, n_features) + 2
    
    # Introduce NaN values randomly
    mask_majority = np.random.random((n_majority, n_features)) < nan_percentage
    mask_minority = np.random.random((n_minority, n_features)) < nan_percentage
    
    X_majority[mask_majority] = np.nan
    X_minority[mask_minority] = np.nan
    
    # Combine into single dataset
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([np.zeros(n_majority), np.ones(n_minority)])
    
    return X, y


# =============================================================================
# Basic Functionality Tests
# =============================================================================

def test_basic_functionality():
    """
    Test basic functionality of OverNaN with SMOTE, ADASYN, and ROSE.
    """
    print("=" * 70)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 70)
    
    # Create test data
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20)
    
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"NaN percentage: {np.isnan(X).sum() / X.size * 100:.2f}%")
    
    # Test SMOTE
    print("\n" + "-" * 50)
    print("Testing SMOTE with NaN handling")
    print("-" * 50)
    
    smote = OverNaN(method='SMOTE', neighbours=5, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    print(f"After SMOTE shape: {X_smote.shape}")
    print(f"After SMOTE class distribution: {dict(zip(*np.unique(y_smote, return_counts=True)))}")
    print(f"NaN percentage after SMOTE: {np.isnan(X_smote).sum() / X_smote.size * 100:.2f}%")
    
    # Test ADASYN
    print("\n" + "-" * 50)
    print("Testing ADASYN with NaN handling")
    print("-" * 50)
    
    adasyn = OverNaN(method='ADASYN', neighbours=5, beta=1.0, random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
    
    print(f"After ADASYN shape: {X_adasyn.shape}")
    print(f"After ADASYN class distribution: {dict(zip(*np.unique(y_adasyn, return_counts=True)))}")
    print(f"NaN percentage after ADASYN: {np.isnan(X_adasyn).sum() / X_adasyn.size * 100:.2f}%")
    
    # Test ROSE
    print("\n" + "-" * 50)
    print("Testing ROSE with NaN handling")
    print("-" * 50)
    
    rose = OverNaN(method='ROSE', shrinkage=1.0, random_state=42)
    X_rose, y_rose = rose.fit_resample(X, y)
    
    print(f"After ROSE shape: {X_rose.shape}")
    print(f"After ROSE class distribution: {dict(zip(*np.unique(y_rose, return_counts=True)))}")
    print(f"NaN percentage after ROSE: {np.isnan(X_rose).sum() / X_rose.size * 100:.2f}%")


def test_sampling_strategies():
    """
    Test different sampling strategies across all methods.
    """
    print("\n" + "=" * 70)
    print("TESTING SAMPLING STRATEGIES")
    print("=" * 70)
    
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20)
    
    strategies = ['auto', 'minority', 'not majority', 0.5, {0: 100, 1: 80}]
    
    for strategy in strategies:
        print(f"\n--- Sampling strategy: {strategy} ---")
        
        # Test with each method
        for method in ['SMOTE', 'ADASYN', 'ROSE']:
            if method == 'ROSE':
                oversampler = OverNaN(method=method, sampling_strategy=strategy, random_state=42)
            else:
                oversampler = OverNaN(method=method, neighbours=5, sampling_strategy=strategy, random_state=42)
            
            X_res, y_res = oversampler.fit_resample(X, y)
            dist = dict(zip(*np.unique(y_res, return_counts=True)))
            print(f"  {method:8s}: {dist}")


def test_nan_handling_strategies():
    """
    Test different NaN handling strategies for all methods.
    """
    print("\n" + "=" * 70)
    print("TESTING NaN HANDLING STRATEGIES")
    print("=" * 70)
    
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20, nan_percentage=0.3)
    original_nan_pct = np.isnan(X).sum() / X.size * 100
    
    strategies = ['preserve_pattern', 'interpolate', 'random_pattern']
    methods = ['SMOTE', 'ADASYN', 'ROSE']
    
    for strategy in strategies:
        print(f"\n--- NaN handling: {strategy} ---")
        print(f"Original NaN %: {original_nan_pct:.2f}%")
        
        for method in methods:
            if method == 'ROSE':
                oversampler = OverNaN(method=method, nan_handling=strategy, random_state=42)
            else:
                oversampler = OverNaN(method=method, neighbours=5, nan_handling=strategy, random_state=42)
            
            X_res, y_res = oversampler.fit_resample(X, y)
            nan_pct = np.isnan(X_res).sum() / X_res.size * 100
            print(f"  {method:8s} resampled NaN %: {nan_pct:.2f}%")


def test_rose_shrinkage():
    """
    Test ROSE shrinkage parameter effect on synthetic sample dispersion.
    """
    print("\n" + "=" * 70)
    print("TESTING ROSE SHRINKAGE PARAMETER")
    print("=" * 70)
    
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20, nan_percentage=0.1)
    
    # Get original minority class statistics
    X_minority_orig = X[y == 1]
    orig_std = np.nanstd(X_minority_orig, axis=0).mean()
    
    print(f"\nOriginal minority class mean std: {orig_std:.4f}")
    print("\nShrinkage effects on synthetic sample dispersion:")
    
    for shrinkage in [0.0, 0.5, 1.0, 1.5, 2.0]:
        rose = OverNaN(method='ROSE', shrinkage=shrinkage, random_state=42)
        X_res, y_res = rose.fit_resample(X, y)
        
        # Get synthetic samples only (those added beyond original)
        n_original = len(X)
        X_synthetic = X_res[n_original:]
        
        if len(X_synthetic) > 0:
            syn_std = np.nanstd(X_synthetic, axis=0).mean()
            print(f"  shrinkage={shrinkage:.1f}: synthetic std = {syn_std:.4f}")
        else:
            print(f"  shrinkage={shrinkage:.1f}: no synthetic samples generated")


# =============================================================================
# Compatibility Tests
# =============================================================================

def test_pandas_compatibility():
    """
    Test compatibility with pandas DataFrames for all methods.
    """
    print("\n" + "=" * 70)
    print("TESTING PANDAS COMPATIBILITY")
    print("=" * 70)
    
    # Create pandas data
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    series = pd.Series(y, name='target')
    
    print(f"\nInput types: X={type(df).__name__}, y={type(series).__name__}")
    
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        print(f"\n--- {method} ---")
        
        if method == 'ROSE':
            oversampler = OverNaN(method=method, random_state=42)
        else:
            oversampler = OverNaN(method=method, neighbours=5, random_state=42)
        
        X_res, y_res = oversampler.fit_resample(df, series)
        
        print(f"  Output types: X={type(X_res).__name__}, y={type(y_res).__name__}")
        print(f"  Columns preserved: {list(X_res.columns) == list(df.columns)}")
        print(f"  Series name preserved: {y_res.name == series.name}")
        print(f"  Final shape: {X_res.shape}")


def test_parallel_processing():
    """
    Test parallel processing capability for all methods.
    """
    print("\n" + "=" * 70)
    print("TESTING PARALLEL PROCESSING")
    print("=" * 70)
    
    # Create larger dataset for parallel processing
    X, y = create_imbalanced_data_with_nan(n_majority=500, n_minority=100)
    
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        print(f"\n--- {method} ---")
        
        # Sequential
        start = time.time()
        if method == 'ROSE':
            oversampler = OverNaN(method=method, n_jobs=1, random_state=42)
        else:
            oversampler = OverNaN(method=method, neighbours=5, n_jobs=1, random_state=42)
        X_seq, y_seq = oversampler.fit_resample(X, y)
        seq_time = time.time() - start
        
        # Parallel
        start = time.time()
        if method == 'ROSE':
            oversampler = OverNaN(method=method, n_jobs=-1, random_state=42)
        else:
            oversampler = OverNaN(method=method, neighbours=5, n_jobs=-1, random_state=42)
        X_par, y_par = oversampler.fit_resample(X, y)
        par_time = time.time() - start
        
        print(f"  Sequential: {seq_time:.3f}s, shape: {X_seq.shape}")
        print(f"  Parallel:   {par_time:.3f}s, shape: {X_par.shape}")
        print(f"  Shapes match: {X_seq.shape == X_par.shape}")


def test_direct_class_usage():
    """
    Test direct usage of SMOTENaN, ADASYNNaN, and ROSENaN classes.
    """
    print("\n" + "=" * 70)
    print("TESTING DIRECT CLASS USAGE")
    print("=" * 70)
    
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20)
    
    # Test SMOTENaN directly
    print("\n--- SMOTENaN direct ---")
    smote = SMOTENaN(neighbours=5, nan_handling='preserve_pattern', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  Shape: {X_res.shape}, Distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    
    # Test ADASYNNaN directly
    print("\n--- ADASYNNaN direct ---")
    adasyn = ADASYNNaN(neighbours=5, beta=1.0, learning_rate=1.0, random_state=42)
    X_res, y_res = adasyn.fit_resample(X, y)
    print(f"  Shape: {X_res.shape}, Distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    
    # Test ROSENaN directly
    print("\n--- ROSENaN direct ---")
    rose = ROSENaN(shrinkage=1.0, nan_handling='preserve_pattern', random_state=42)
    X_res, y_res = rose.fit_resample(X, y)
    print(f"  Shape: {X_res.shape}, Distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")


# =============================================================================
# Classifier Integration Tests
# =============================================================================

def test_with_classifiers():
    """
    Test OverNaN with real classifiers (XGBoost) comparing all methods.
    """
    print("\n" + "=" * 70)
    print("TESTING WITH CLASSIFIERS")
    print("=" * 70)
    
    # Create dataset
    X, y = create_imbalanced_data_with_nan(
        n_majority=500, n_minority=50, n_features=20, nan_percentage=0.15
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test set class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    results = {}
    
    # Baseline - No oversampling
    print("\n" + "-" * 50)
    print("BASELINE - No oversampling")
    print("-" * 50)
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))
    results['Baseline'] = classification_report(y_test, y_pred, output_dict=True)
    
    # With SMOTE
    print("\n" + "-" * 50)
    print("WITH SMOTE-NaN")
    print("-" * 50)
    
    smote = OverNaN(method='SMOTE', neighbours=5, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {dict(zip(*np.unique(y_train_smote, return_counts=True)))}")
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_smote, y_train_smote)
    y_pred = xgb_model.predict(X_test)
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))
    results['SMOTE'] = classification_report(y_test, y_pred, output_dict=True)
    
    # With ADASYN
    print("\n" + "-" * 50)
    print("WITH ADASYN-NaN")
    print("-" * 50)
    
    adasyn = OverNaN(method='ADASYN', neighbours=5, beta=1.0, random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    print(f"After ADASYN: {dict(zip(*np.unique(y_train_adasyn, return_counts=True)))}")
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_adasyn, y_train_adasyn)
    y_pred = xgb_model.predict(X_test)
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))
    results['ADASYN'] = classification_report(y_test, y_pred, output_dict=True)
    
    # With ROSE
    print("\n" + "-" * 50)
    print("WITH ROSE-NaN")
    print("-" * 50)
    
    rose = OverNaN(method='ROSE', shrinkage=1.0, random_state=42)
    X_train_rose, y_train_rose = rose.fit_resample(X_train, y_train)
    print(f"After ROSE: {dict(zip(*np.unique(y_train_rose, return_counts=True)))}")
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_rose, y_train_rose)
    y_pred = xgb_model.predict(X_test)
    
    print("\nXGBoost Results:")
    print(classification_report(y_test, y_pred, target_names=['Majority', 'Minority']))
    results['ROSE'] = classification_report(y_test, y_pred, output_dict=True)
    
    # Compare results
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print("\nMinority Class Recall (Sensitivity):")
    for method in results:
        recall = results[method]['1.0']['recall']
        print(f"  {method:10s}: {recall:.3f}")
    
    print("\nMinority Class F1-Score:")
    for method in results:
        f1 = results[method]['1.0']['f1-score']
        print(f"  {method:10s}: {f1:.3f}")
    
    print("\nWeighted Average F1-Score:")
    for method in results:
        f1 = results[method]['weighted avg']['f1-score']
        print(f"  {method:10s}: {f1:.3f}")


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_edge_cases():
    """
    Test edge cases and error handling for all methods.
    """
    print("\n" + "=" * 70)
    print("TESTING EDGE CASES")
    print("=" * 70)
    
    # Test with very few samples
    print("\n--- Very few minority samples ---")
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 0, 0, 0, 1])
    
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        try:
            if method == 'ROSE':
                oversampler = OverNaN(method=method, random_state=42)
            else:
                oversampler = OverNaN(method=method, neighbours=5, random_state=42)
            X_res, y_res = oversampler.fit_resample(X, y)
            dist = dict(zip(*np.unique(y_res, return_counts=True)))
            print(f"  {method}: Success, distribution = {dist}")
        except Exception as e:
            print(f"  {method}: Error - {e}")
    
    # Test with all NaN features in a sample
    print("\n--- Sample with all NaN features ---")
    X = np.array([[1, 2, 3], [np.nan, np.nan, np.nan], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0, 0])
    
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        try:
            if method == 'ROSE':
                oversampler = OverNaN(method=method, random_state=42)
            else:
                oversampler = OverNaN(method=method, neighbours=2, random_state=42)
            X_res, y_res = oversampler.fit_resample(X, y)
            print(f"  {method}: Success, shape = {X_res.shape}")
        except Exception as e:
            print(f"  {method}: Error - {e}")
    
    # Test with high NaN percentage
    print("\n--- High NaN percentage (50%) ---")
    X, y = create_imbalanced_data_with_nan(n_majority=50, n_minority=10, nan_percentage=0.5)
    
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        try:
            if method == 'ROSE':
                oversampler = OverNaN(method=method, random_state=42)
            else:
                oversampler = OverNaN(method=method, neighbours=3, random_state=42)
            X_res, y_res = oversampler.fit_resample(X, y)
            nan_pct = np.isnan(X_res).sum() / X_res.size * 100
            print(f"  {method}: Success, NaN% = {nan_pct:.2f}%")
        except Exception as e:
            print(f"  {method}: Error - {e}")
    
    # Test with single feature
    print("\n--- Single feature ---")
    X = np.array([[1], [2], [3], [4], [10]])
    y = np.array([0, 0, 0, 0, 1])
    
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        try:
            if method == 'ROSE':
                oversampler = OverNaN(method=method, random_state=42)
            else:
                oversampler = OverNaN(method=method, neighbours=2, random_state=42)
            X_res, y_res = oversampler.fit_resample(X, y)
            print(f"  {method}: Success, shape = {X_res.shape}")
        except Exception as e:
            print(f"  {method}: Error - {e}")


def test_reproducibility():
    """
    Test that random_state ensures reproducible results.
    """
    print("\n" + "=" * 70)
    print("TESTING REPRODUCIBILITY")
    print("=" * 70)
    
    X, y = create_imbalanced_data_with_nan(n_majority=100, n_minority=20)
    
    for method in ['SMOTE', 'ADASYN', 'ROSE']:
        print(f"\n--- {method} ---")
        
        # Run twice with same random state
        if method == 'ROSE':
            over1 = OverNaN(method=method, random_state=42)
            over2 = OverNaN(method=method, random_state=42)
        else:
            over1 = OverNaN(method=method, neighbours=5, random_state=42)
            over2 = OverNaN(method=method, neighbours=5, random_state=42)
        
        X_res1, y_res1 = over1.fit_resample(X, y)
        X_res2, y_res2 = over2.fit_resample(X, y)
        
        # Check shapes match
        shapes_match = X_res1.shape == X_res2.shape
        
        # Check values match (accounting for NaN)
        values_match = np.allclose(X_res1, X_res2, equal_nan=True)
        
        print(f"  Shapes match: {shapes_match}")
        print(f"  Values match: {values_match}")


def test_repr():
    """
    Test string representation of OverNaN instances.
    """
    print("\n" + "=" * 70)
    print("TESTING REPR")
    print("=" * 70)
    
    print("\n--- OverNaN representations ---")
    print(repr(OverNaN(method='SMOTE', neighbours=5, random_state=42)))
    print(repr(OverNaN(method='ADASYN', neighbours=5, beta=0.8, learning_rate=1.5, random_state=42)))
    print(repr(OverNaN(method='ROSE', shrinkage=1.5, nan_handling='interpolate', random_state=42)))


# =============================================================================
# Usage Examples
# =============================================================================

def demonstrate_usage():
    """
    Demonstrate typical usage patterns for all methods.
    """
    print("\n" + "=" * 70)
    print("USAGE DEMONSTRATION")
    print("=" * 70)
    
    print("""
    # Example 1: Basic SMOTE usage
    from overnan import OverNaN
    
    oversampler = OverNaN(method='SMOTE', neighbours=5, random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # Example 2: ADASYN with adaptive parameters
    oversampler = OverNaN(
        method='ADASYN',
        neighbours=7,
        beta=0.5,           # Generate fewer synthetic samples
        learning_rate=1.5,  # Focus more on hard examples
        nan_handling='interpolate'
    )
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # Example 3: ROSE - no neighbors required, good for high dimensions
    oversampler = OverNaN(
        method='ROSE',
        shrinkage=1.0,      # Standard Silverman bandwidth
        nan_handling='preserve_pattern',
        n_jobs=-1           # Use all CPU cores
    )
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # Example 4: ROSE with tighter samples (closer to originals)
    oversampler = OverNaN(
        method='ROSE',
        shrinkage=0.5,      # Samples closer to seeds
        random_state=42
    )
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # Example 5: Direct class usage
    from overnan import SMOTENaN, ADASYNNaN, ROSENaN
    
    smote = SMOTENaN(neighbours=5, nan_handling='preserve_pattern')
    adasyn = ADASYNNaN(neighbours=5, beta=1.0, learning_rate=1.0)
    rose = ROSENaN(shrinkage=1.0, nan_handling='interpolate')
    
    # Example 6: Sampling strategy options
    oversampler = OverNaN(
        method='SMOTE',
        sampling_strategy='auto',  # Balance all minority classes
        # OR sampling_strategy=0.8,  # 80% of majority
        # OR sampling_strategy={0: 100, 1: 80},  # Exact counts
    )
    
    # Example 7: Integration with XGBoost (handles NaN naturally)
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    
    oversampler = OverNaN(method='ROSE', random_state=42)
    X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
    
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    accuracy = model.score(X_test, y_test)
    """)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """
    Run all tests.
    """
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "OverNaN v0.2.0 Test Suite" + " " * 19 + "#")
    print("#" * 70)
    
    test_basic_functionality()
    test_sampling_strategies()
    test_nan_handling_strategies()
    test_rose_shrinkage()
    test_pandas_compatibility()
    test_parallel_processing()
    test_direct_class_usage()
    test_with_classifiers()
    test_edge_cases()
    test_reproducibility()
    test_repr()
    demonstrate_usage()
    
    print("\n" + "#" * 70)
    print("#" + " " * 21 + "All Tests Completed!" + " " * 21 + "#")
    print("#" * 70)


if __name__ == "__main__":
    run_all_tests()
