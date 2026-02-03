"""
OverNaN: Oversampling for Imbalanced Learning with Missing Values
==================================================================

A Python package for handling class imbalance in datasets with missing values.
Implements SMOTE, ADASYN, and ROSE algorithms that respect and preserve 
missingness patterns.

References
----------
SMOTE:
    Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002).
    SMOTE: Synthetic Minority Over-sampling Technique.
    Journal of Artificial Intelligence Research, 16, 321-357.
    DOI: 10.1613/jair.953

ADASYN:
    He, H., Bai, Y., Garcia, E.A., Li, S. (2008).
    ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning.
    IEEE International Joint Conference on Neural Networks, 1322-1328.
    DOI: 10.1109/IJCNN.2008.4633969

ROSE:
    Menardi, G. and Torelli, N. (2014).
    Training and assessing classification rules with imbalanced data.
    Data Mining and Knowledge Discovery, 28, 92-122.
    DOI: 10.1007/s10618-012-0295-5
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from typing import Optional, Union, Dict, Literal
from joblib import Parallel, delayed
import warnings
from abc import ABC, abstractmethod
import gc


__version__ = "0.2.0"
__all__ = ['OverNaN', 'SMOTENaN', 'ADASYNNaN', 'ROSENaN']


# =============================================================================
# Base Class
# =============================================================================

class BaseOverSamplerNaN(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for oversampling algorithms that handle missing values.
    
    Provides common functionality for sampling strategy determination,
    NaN-aware distance calculations, and pandas compatibility.
    """
    
    def __init__(self,
                 sampling_strategy: Union[str, float, Dict] = 'auto',
                 random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None):
        """
        Initialize base oversampler.
        
        Parameters
        ----------
        sampling_strategy : float, str, or dict, default='auto'
            Sampling information to resample the data set.
            - If float: ratio of minority to majority class after resampling
            - If str: 'auto', 'minority', 'not minority', 'not majority', 'all'
            - If dict: keys are class labels, values are desired sample counts
        
        random_state : int or RandomState, default=None
            Random state for reproducibility.
        
        n_jobs : int, default=None
            Number of parallel jobs. None means 1, -1 means all processors.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def _determine_sampling_strategy(self, y: np.ndarray) -> Dict:
        """
        Convert sampling_strategy parameter to concrete target counts per class.
        
        Parameters
        ----------
        y : np.ndarray
            Target labels.
        
        Returns
        -------
        Dict
            Dictionary mapping class labels to target sample counts.
        """
        # Count samples per class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_dict = dict(zip(unique_classes, class_counts))
        
        # If already a dict, return as-is
        if isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
        
        # Identify majority and minority classes
        max_count = np.max(class_counts)
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        # Convert string strategies to target counts
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'not majority':
            # Resample all classes except majority to match majority count
            target_counts = {}
            for cls, count in class_dict.items():
                if cls != majority_class:
                    target_counts[cls] = max_count
                else:
                    target_counts[cls] = count
                    
        elif self.sampling_strategy == 'minority':
            # Only resample the minority class to match majority
            target_counts = class_dict.copy()
            target_counts[minority_class] = max_count
            
        elif self.sampling_strategy == 'not minority':
            # Resample all classes except minority to match majority
            target_counts = {}
            for cls, count in class_dict.items():
                if cls != minority_class:
                    target_counts[cls] = max_count
                else:
                    target_counts[cls] = count
                    
        elif self.sampling_strategy == 'all':
            # Resample all classes to match majority
            target_counts = {cls: max_count for cls in unique_classes}
            
        elif isinstance(self.sampling_strategy, (int, float)):
            # Use as ratio relative to majority class
            target_counts = {}
            for cls, count in class_dict.items():
                if cls != majority_class:
                    target_counts[cls] = int(max_count * self.sampling_strategy)
                else:
                    target_counts[cls] = count
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")
        
        return target_counts
    
    @abstractmethod
    def _resample_class(self, X_class: np.ndarray, X: np.ndarray, 
                       y: np.ndarray, cls, n_samples: int,
                       random_state) -> np.ndarray:
        """
        Generate synthetic samples for a single class.
        
        Must be implemented by subclasses.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of the target class.
        X : np.ndarray
            Full feature matrix.
        y : np.ndarray
            Full target array.
        cls : any
            Class label being resampled.
        n_samples : int
            Number of synthetic samples to generate.
        random_state : RandomState
            Random state for reproducibility.
        
        Returns
        -------
        np.ndarray
            Array of synthetic samples, shape (n_samples, n_features).
        """
        pass
    
    def fit_resample(self, X, y):
        """
        Resample the dataset to balance class distribution.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (can contain NaN values).
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns
        -------
        X_resampled : array-like of shape (n_samples_new, n_features)
            Resampled feature matrix.
        y_resampled : array-like of shape (n_samples_new,)
            Resampled target labels.
        """
        # Preserve pandas metadata if applicable
        X_is_df = isinstance(X, pd.DataFrame)
        y_is_series = isinstance(y, pd.Series)
        X_columns = X.columns if X_is_df else None
        y_name = y.name if y_is_series else None
        
        # Convert to numpy arrays, allowing NaN values
        # Use ensure_all_finite for sklearn >= 1.6 compatibility
        try:
            X = check_array(X, ensure_all_finite='allow-nan', dtype=np.float64)
        except TypeError:
            # Fallback for older sklearn versions
            X = check_array(X, force_all_finite='allow-nan', dtype=np.float64)
        y = check_array(y, ensure_2d=False, dtype=None)
        
        # Initialize random state for reproducibility
        random_state = check_random_state(self.random_state)
        
        # Determine target sample counts per class
        target_counts = self._determine_sampling_strategy(y)
        unique_classes = np.unique(y)
        
        # Start with copies of original data
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        # Generate synthetic samples for each class that needs resampling
        for cls in unique_classes:
            if cls not in target_counts:
                continue
            
            # Get current class samples
            class_indices = np.where(y == cls)[0]
            current_count = len(class_indices)
            n_synthetic = target_counts[cls] - current_count
            
            # Skip if no synthetic samples needed
            if n_synthetic <= 0:
                continue
            
            X_class = X[class_indices]
            
            # Generate synthetic samples using subclass implementation
            synthetic_samples = self._resample_class(
                X_class, X, y, cls, n_synthetic, random_state
            )
            
            # Append synthetic samples to resampled data
            if len(synthetic_samples) > 0:
                X_resampled = np.vstack([X_resampled, synthetic_samples])
                y_resampled = np.hstack([y_resampled, [cls] * len(synthetic_samples)])
        
        # Convert back to pandas if input was pandas
        if X_is_df:
            X_resampled = pd.DataFrame(X_resampled, columns=X_columns)
        if y_is_series:
            y_resampled = pd.Series(y_resampled, name=y_name)
        
        # Memory cleanup
        gc.collect()
        
        return X_resampled, y_resampled


# =============================================================================
# Neighbor-Based Base Class (for SMOTE and ADASYN)
# =============================================================================

class BaseNeighborOverSamplerNaN(BaseOverSamplerNaN):
    """
    Base class for neighbor-based oversampling algorithms (SMOTE, ADASYN).
    
    Provides NaN-aware distance calculation, neighbor finding, and
    synthetic sample generation via interpolation.
    """
    
    def __init__(self,
                 neighbours: int = 5,
                 sampling_strategy: Union[str, float, Dict] = 'auto',
                 random_state: Optional[int] = None,
                 nan_handling: Literal['preserve_pattern', 'interpolate', 'random_pattern'] = 'preserve_pattern',
                 n_jobs: Optional[int] = None):
        """
        Initialize neighbor-based oversampler.
        
        Parameters
        ----------
        neighbours : int, default=5
            Number of nearest neighbors for synthetic sample generation.
        
        sampling_strategy : float, str, or dict, default='auto'
            Sampling information to resample the data set.
        
        random_state : int or RandomState, default=None
            Random state for reproducibility.
        
        nan_handling : str, default='preserve_pattern'
            Strategy for handling NaN in synthetic samples:
            - 'preserve_pattern': NaN if either parent has NaN
            - 'interpolate': Use available value if one parent has it
            - 'random_pattern': Randomly choose NaN pattern from parents
        
        n_jobs : int, default=None
            Number of parallel jobs.
        """
        super().__init__(sampling_strategy, random_state, n_jobs)
        self.neighbours = neighbours
        self.nan_handling = nan_handling
    
    def _calculate_distance_with_nan(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two samples, handling NaN values.
        
        Distance is computed using only features where both samples have
        valid (non-NaN) values, then scaled by the proportion of valid features.
        
        Parameters
        ----------
        x1 : np.ndarray
            First sample.
        x2 : np.ndarray
            Second sample.
        
        Returns
        -------
        float
            Scaled Euclidean distance, or np.inf if no valid features.
        """
        # Identify features where both samples have valid values
        valid_mask = ~(np.isnan(x1) | np.isnan(x2))
        
        # Return infinity if no valid features for comparison
        if not np.any(valid_mask):
            return np.inf
        
        # Compute Euclidean distance on valid features
        diff = x1[valid_mask] - x2[valid_mask]
        n_valid = np.sum(valid_mask)
        n_total = len(x1)
        
        # Scale distance by proportion of valid features to compensate
        # for reduced dimensionality in the distance calculation
        distance = np.sqrt(np.sum(diff ** 2) * (n_total / n_valid))
        
        return distance
    
    def _find_neighbours(self, sample: np.ndarray, 
                        X_pool: np.ndarray, 
                        n_neighbours: int,
                        exclude_self: bool = True) -> np.ndarray:
        """
        Find k nearest neighbors for a sample using NaN-aware distance.
        
        Parameters
        ----------
        sample : np.ndarray
            Query sample.
        X_pool : np.ndarray
            Pool of candidate neighbors.
        n_neighbours : int
            Number of neighbors to find.
        exclude_self : bool, default=True
            Whether to exclude exact matches (the sample itself).
        
        Returns
        -------
        np.ndarray
            Indices of the k nearest neighbors in X_pool.
        """
        distances = []
        for i, x in enumerate(X_pool):
            # Skip self-matches if requested
            if exclude_self and np.array_equal(sample, x, equal_nan=True):
                continue
            dist = self._calculate_distance_with_nan(sample, x)
            distances.append((dist, i))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[0])
        
        # Warn if fewer neighbors available than requested
        if len(distances) < n_neighbours:
            if len(distances) > 0:
                warnings.warn(f"Only found {len(distances)} neighbors, requested {n_neighbours}")
            n_neighbours = len(distances)
        
        return np.array([idx for _, idx in distances[:n_neighbours]])
    
    def _generate_synthetic_sample(self, sample: np.ndarray, 
                                  neighbor: np.ndarray,
                                  random_state) -> np.ndarray:
        """
        Generate a synthetic sample by interpolating between sample and neighbor.
        
        Parameters
        ----------
        sample : np.ndarray
            Seed sample.
        neighbor : np.ndarray
            Neighbor sample for interpolation.
        random_state : RandomState
            Random state for reproducibility.
        
        Returns
        -------
        np.ndarray
            Synthetic sample.
        """
        synthetic = np.zeros_like(sample)
        gap = random_state.random()  # Random interpolation factor in [0, 1)
        
        for i in range(len(sample)):
            sample_nan = np.isnan(sample[i])
            neighbor_nan = np.isnan(neighbor[i])
            
            if self.nan_handling == 'interpolate':
                # Use available values; interpolate if both available
                if not sample_nan and not neighbor_nan:
                    synthetic[i] = sample[i] + gap * (neighbor[i] - sample[i])
                elif not sample_nan:
                    synthetic[i] = sample[i]
                elif not neighbor_nan:
                    synthetic[i] = neighbor[i]
                else:
                    synthetic[i] = np.nan
                    
            elif self.nan_handling == 'preserve_pattern':
                # Preserve NaN if either parent has NaN (most conservative)
                if sample_nan or neighbor_nan:
                    synthetic[i] = np.nan
                else:
                    synthetic[i] = sample[i] + gap * (neighbor[i] - sample[i])
                    
            elif self.nan_handling == 'random_pattern':
                # Both NaN: result is NaN
                # Both valid: interpolate
                # One NaN: randomly choose whether to preserve NaN
                if sample_nan and neighbor_nan:
                    synthetic[i] = np.nan
                elif not sample_nan and not neighbor_nan:
                    synthetic[i] = sample[i] + gap * (neighbor[i] - sample[i])
                else:
                    if random_state.random() > 0.5:
                        synthetic[i] = np.nan
                    else:
                        # Use the available value
                        synthetic[i] = sample[i] if not sample_nan else neighbor[i]
        
        return synthetic


# =============================================================================
# SMOTENaN Implementation
# =============================================================================

class SMOTENaN(BaseNeighborOverSamplerNaN):
    """
    SMOTE (Synthetic Minority Over-sampling Technique) with NaN handling.
    
    Generates synthetic samples by interpolating between minority class
    samples and their nearest neighbors, while preserving missing value patterns.
    
    Parameters
    ----------
    neighbours : int, default=5
        Number of nearest neighbors for synthetic sample generation.
    
    sampling_strategy : float, str, or dict, default='auto'
        Sampling information to resample the data set.
    
    random_state : int or RandomState, default=None
        Random state for reproducibility.
    
    nan_handling : str, default='preserve_pattern'
        Strategy for handling NaN in synthetic samples.
    
    n_jobs : int, default=None
        Number of parallel jobs.
    
    References
    ----------
    Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002).
    SMOTE: Synthetic Minority Over-sampling Technique.
    Journal of Artificial Intelligence Research, 16, 321-357.
    DOI: 10.1613/jair.953
    """
    
    def _generate_samples_for_instance(self, sample: np.ndarray, X_class: np.ndarray,
                                      n_samples: int, random_state) -> list:
        """
        Generate synthetic samples for a single seed instance.
        
        Used for parallel processing across seed samples.
        
        Parameters
        ----------
        sample : np.ndarray
            Seed sample.
        X_class : np.ndarray
            All samples of the target class.
        n_samples : int
            Number of synthetic samples to generate from this seed.
        random_state : RandomState
            Random state for reproducibility.
        
        Returns
        -------
        list
            List of synthetic samples.
        """
        synthetic_samples = []
        
        # Find neighbors within the same class
        neighbor_indices = self._find_neighbours(
            sample, X_class, self.neighbours, exclude_self=True
        )
        
        if len(neighbor_indices) == 0:
            return synthetic_samples
        
        # Generate requested number of synthetic samples
        for _ in range(n_samples):
            # Randomly select a neighbor
            neighbor_idx = random_state.choice(neighbor_indices)
            neighbor = X_class[neighbor_idx]
            # Interpolate between seed and neighbor
            synthetic = self._generate_synthetic_sample(sample, neighbor, random_state)
            synthetic_samples.append(synthetic)
        
        return synthetic_samples
    
    def _resample_class(self, X_class: np.ndarray, X: np.ndarray, 
                       y: np.ndarray, cls, n_samples: int,
                       random_state) -> np.ndarray:
        """
        Generate synthetic samples for a class using SMOTE algorithm.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of the target class.
        X : np.ndarray
            Full feature matrix (unused in basic SMOTE).
        y : np.ndarray
            Full target array (unused in basic SMOTE).
        cls : any
            Class label being resampled.
        n_samples : int
            Number of synthetic samples to generate.
        random_state : RandomState
            Random state for reproducibility.
        
        Returns
        -------
        np.ndarray
            Array of synthetic samples.
        """
        n_minority = len(X_class)
        
        # Distribute synthetic samples evenly across seed samples
        samples_per_instance = n_samples // n_minority
        extra_samples = n_samples % n_minority
        
        # Randomly select instances to receive extra samples
        extra_indices = set(random_state.choice(n_minority, extra_samples, replace=False))
        
        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential processing
            synthetic_samples = []
            for i in range(n_minority):
                n_synthetic = samples_per_instance + (1 if i in extra_indices else 0)
                if n_synthetic > 0:
                    samples = self._generate_samples_for_instance(
                        X_class[i], X_class, n_synthetic, random_state
                    )
                    synthetic_samples.extend(samples)
        else:
            # Parallel processing with separate random states per job
            n_synthetic_list = [
                samples_per_instance + (1 if i in extra_indices else 0)
                for i in range(n_minority)
            ]
            
            # Generate independent seeds for parallel jobs
            # Use 2**31 - 1 for Windows int32 compatibility
            seeds = random_state.randint(0, 2**31 - 1, size=n_minority)
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_samples_for_instance)(
                    X_class[i], X_class, n_synthetic_list[i], 
                    check_random_state(seeds[i])
                )
                for i in range(n_minority) if n_synthetic_list[i] > 0
            )
            
            synthetic_samples = [sample for result in results for sample in result]
        
        if synthetic_samples:
            return np.array(synthetic_samples)
        else:
            return np.array([]).reshape(0, X_class.shape[1])


# =============================================================================
# ADASYNNaN Implementation
# =============================================================================

class ADASYNNaN(BaseNeighborOverSamplerNaN):
    """
    ADASYN (Adaptive Synthetic Sampling) with NaN handling.
    
    Generates more synthetic samples for minority class instances that are
    harder to learn (those with more majority class neighbors), while
    preserving missing value patterns.
    
    Parameters
    ----------
    neighbours : int, default=5
        Number of nearest neighbors for synthetic sample generation.
    
    sampling_strategy : float, str, or dict, default='auto'
        Sampling information to resample the data set.
    
    random_state : int or RandomState, default=None
        Random state for reproducibility.
    
    nan_handling : str, default='preserve_pattern'
        Strategy for handling NaN in synthetic samples.
    
    n_jobs : int, default=None
        Number of parallel jobs.
    
    beta : float, default=1.0
        Balance parameter. 1.0 generates enough samples for full balance.
    
    learning_rate : float, default=1.0
        Exponent for density weights. Higher values focus more on hard samples.
    
    References
    ----------
    He, H., Bai, Y., Garcia, E.A., Li, S. (2008).
    ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning.
    IEEE International Joint Conference on Neural Networks, 1322-1328.
    DOI: 10.1109/IJCNN.2008.4633969
    """
    
    def __init__(self,
                 neighbours: int = 5,
                 sampling_strategy: Union[str, float, Dict] = 'auto',
                 random_state: Optional[int] = None,
                 nan_handling: Literal['preserve_pattern', 'interpolate', 'random_pattern'] = 'preserve_pattern',
                 n_jobs: Optional[int] = None,
                 beta: float = 1.0,
                 learning_rate: float = 1.0):
        """
        Initialize ADASYN oversampler.
        
        Parameters
        ----------
        beta : float, default=1.0
            Balance parameter controlling synthetic sample volume.
            beta < 1: fewer synthetic samples
            beta = 1: full balance
            beta > 1: more than needed for balance
        
        learning_rate : float, default=1.0
            Exponent applied to density weights. Higher values increase
            focus on hard-to-learn instances.
        """
        super().__init__(neighbours, sampling_strategy, random_state, nan_handling, n_jobs)
        self.beta = beta
        self.learning_rate = learning_rate
    
    def _calculate_density_distribution(self, X_minority: np.ndarray, 
                                       X: np.ndarray, y: np.ndarray,
                                       minority_class) -> np.ndarray:
        """
        Calculate difficulty-based density distribution for minority samples.
        
        Samples with more majority class neighbors are considered harder
        to learn and receive higher weights.
        
        Parameters
        ----------
        X_minority : np.ndarray
            Feature matrix for minority class samples.
        X : np.ndarray
            Full feature matrix.
        y : np.ndarray
            Full target array.
        minority_class : any
            Label of the minority class.
        
        Returns
        -------
        np.ndarray
            Normalized density weights for each minority sample.
        """
        n_minority = len(X_minority)
        
        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential processing
            densities = np.zeros(n_minority)
            for i, sample in enumerate(X_minority):
                # Find neighbors in the full dataset
                neighbor_indices = self._find_neighbours(
                    sample, X, self.neighbours, exclude_self=True
                )
                if len(neighbor_indices) > 0:
                    neighbor_classes = y[neighbor_indices]
                    # Density = proportion of majority class neighbors
                    n_majority = np.sum(neighbor_classes != minority_class)
                    densities[i] = n_majority / len(neighbor_indices)
        else:
            # Parallel processing
            def compute_density(sample):
                neighbor_indices = self._find_neighbours(
                    sample, X, self.neighbours, exclude_self=True
                )
                if len(neighbor_indices) == 0:
                    return 0
                neighbor_classes = y[neighbor_indices]
                n_majority = np.sum(neighbor_classes != minority_class)
                return n_majority / len(neighbor_indices)
            
            densities = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_density)(sample) for sample in X_minority
            )
            densities = np.array(densities)
        
        # Apply learning rate exponent to emphasize hard samples
        densities = densities ** self.learning_rate
        
        # Normalize to probability distribution
        if np.sum(densities) > 0:
            densities = densities / np.sum(densities)
        else:
            # Uniform distribution if all densities are zero
            densities = np.ones(n_minority) / n_minority
        
        return densities
    
    def _resample_class(self, X_class: np.ndarray, X: np.ndarray, 
                       y: np.ndarray, cls, n_samples: int,
                       random_state) -> np.ndarray:
        """
        Generate synthetic samples for a class using ADASYN algorithm.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of the target class.
        X : np.ndarray
            Full feature matrix.
        y : np.ndarray
            Full target array.
        cls : any
            Class label being resampled.
        n_samples : int
            Number of synthetic samples to generate.
        random_state : RandomState
            Random state for reproducibility.
        
        Returns
        -------
        np.ndarray
            Array of synthetic samples.
        """
        # Calculate adaptive density distribution
        densities = self._calculate_density_distribution(X_class, X, y, cls)
        
        # Allocate synthetic samples proportionally to density
        n_synthetic_per_sample = np.round(densities * n_samples * self.beta).astype(int)
        
        synthetic_samples = []
        
        for i, n_synthetic in enumerate(n_synthetic_per_sample):
            if n_synthetic == 0:
                continue
            
            sample = X_class[i]
            
            # Find neighbors within the same class for interpolation
            neighbor_indices = self._find_neighbours(
                sample, X_class, self.neighbours, exclude_self=True
            )
            
            if len(neighbor_indices) == 0:
                continue
            
            # Generate synthetic samples for this seed
            for _ in range(n_synthetic):
                neighbor_idx = random_state.choice(neighbor_indices)
                neighbor = X_class[neighbor_idx]
                synthetic = self._generate_synthetic_sample(sample, neighbor, random_state)
                synthetic_samples.append(synthetic)
        
        if synthetic_samples:
            return np.array(synthetic_samples)
        else:
            return np.array([]).reshape(0, X_class.shape[1])


# =============================================================================
# ROSENaN Implementation
# =============================================================================

class ROSENaN(BaseOverSamplerNaN):
    """
    ROSE (Random Over-Sampling Examples) with NaN handling.
    
    Generates synthetic samples by perturbing seed samples using kernel
    density estimation with Gaussian kernels. Unlike SMOTE and ADASYN,
    ROSE does not require neighbor finding, making it simpler and often
    faster for high-dimensional data.
    
    The bandwidth (smoothing parameter) is computed using Silverman's rule
    of thumb, optionally scaled by a shrinkage factor.
    
    Parameters
    ----------
    sampling_strategy : float, str, or dict, default='auto'
        Sampling information to resample the data set.
    
    random_state : int or RandomState, default=None
        Random state for reproducibility.
    
    nan_handling : str, default='preserve_pattern'
        Strategy for handling NaN in synthetic samples:
        - 'preserve_pattern': Preserve NaN from seed sample (default)
        - 'interpolate': Fill NaN with class mean, then perturb
        - 'random_pattern': Randomly preserve NaN based on class NaN rate
    
    n_jobs : int, default=None
        Number of parallel jobs.
    
    shrinkage : float, default=1.0
        Multiplier for the bandwidth. Values < 1 produce samples closer
        to seeds; values > 1 produce more dispersed samples.
    
    References
    ----------
    Menardi, G. and Torelli, N. (2014).
    Training and assessing classification rules with imbalanced data.
    Data Mining and Knowledge Discovery, 28, 92-122.
    DOI: 10.1007/s10618-012-0295-5
    
    Lunardon, N., Menardi, G., and Torelli, N. (2014).
    ROSE: a Package for Binary Imbalanced Learning.
    R Journal, 6, 82-92.
    """
    
    def __init__(self,
                 sampling_strategy: Union[str, float, Dict] = 'auto',
                 random_state: Optional[int] = None,
                 nan_handling: Literal['preserve_pattern', 'interpolate', 'random_pattern'] = 'preserve_pattern',
                 n_jobs: Optional[int] = None,
                 shrinkage: float = 1.0):
        """
        Initialize ROSE oversampler.
        
        Parameters
        ----------
        shrinkage : float, default=1.0
            Bandwidth scaling factor. Controls dispersion of synthetic samples:
            - shrinkage = 0: Equivalent to random oversampling (exact copies)
            - shrinkage = 1: Standard Silverman bandwidth
            - shrinkage > 1: More dispersed synthetic samples
        """
        super().__init__(sampling_strategy, random_state, n_jobs)
        self.nan_handling = nan_handling
        self.shrinkage = shrinkage
    
    def _compute_bandwidth(self, X_class: np.ndarray) -> np.ndarray:
        """
        Compute per-feature bandwidth using Silverman's rule of thumb.
        
        Silverman's rule: h_j = (4 / (d + 2))^(1/(d+4)) * n^(-1/(d+4)) * sigma_j
        
        For features with NaN values, standard deviation is computed using
        only the non-NaN values.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of one class.
        
        Returns
        -------
        np.ndarray
            Bandwidth for each feature, shape (n_features,).
        """
        n_samples, n_features = X_class.shape
        
        # Silverman's constant: (4 / (d + 2))^(1/(d+4))
        silverman_constant = (4.0 / (n_features + 2.0)) ** (1.0 / (n_features + 4.0))
        
        # Sample size factor: n^(-1/(d+4))
        n_factor = n_samples ** (-1.0 / (n_features + 4.0))
        
        # Compute per-feature standard deviation, ignoring NaN
        bandwidths = np.zeros(n_features)
        for j in range(n_features):
            col = X_class[:, j]
            valid_values = col[~np.isnan(col)]
            if len(valid_values) > 1:
                sigma_j = np.std(valid_values, ddof=1)
                bandwidths[j] = silverman_constant * n_factor * sigma_j * self.shrinkage
            else:
                # No valid values or single value: zero bandwidth (no perturbation)
                bandwidths[j] = 0.0
        
        return bandwidths
    
    def _compute_class_nan_rates(self, X_class: np.ndarray) -> np.ndarray:
        """
        Compute per-feature NaN rates for a class.
        
        Used by 'random_pattern' nan_handling strategy.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of one class.
        
        Returns
        -------
        np.ndarray
            Proportion of NaN values for each feature, shape (n_features,).
        """
        return np.mean(np.isnan(X_class), axis=0)
    
    def _compute_class_means(self, X_class: np.ndarray) -> np.ndarray:
        """
        Compute per-feature means for a class, ignoring NaN.
        
        Used by 'interpolate' nan_handling strategy.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of one class.
        
        Returns
        -------
        np.ndarray
            Mean value for each feature, shape (n_features,).
            NaN if all values are NaN for a feature.
        """
        return np.nanmean(X_class, axis=0)
    
    def _generate_synthetic_sample_rose(self, seed: np.ndarray,
                                        bandwidths: np.ndarray,
                                        random_state,
                                        nan_rates: Optional[np.ndarray] = None,
                                        class_means: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a synthetic sample by perturbing a seed sample.
        
        Parameters
        ----------
        seed : np.ndarray
            Seed sample to perturb.
        bandwidths : np.ndarray
            Per-feature bandwidth values.
        random_state : RandomState
            Random state for reproducibility.
        nan_rates : np.ndarray, optional
            Per-feature NaN rates (for 'random_pattern' strategy).
        class_means : np.ndarray, optional
            Per-feature means (for 'interpolate' strategy).
        
        Returns
        -------
        np.ndarray
            Synthetic sample.
        """
        n_features = len(seed)
        synthetic = np.zeros(n_features)
        
        for j in range(n_features):
            seed_nan = np.isnan(seed[j])
            
            if self.nan_handling == 'preserve_pattern':
                # Preserve NaN from seed; perturb non-NaN values
                if seed_nan:
                    synthetic[j] = np.nan
                else:
                    # Gaussian perturbation centered at seed value
                    synthetic[j] = seed[j] + random_state.normal(0, bandwidths[j])
                    
            elif self.nan_handling == 'interpolate':
                # Fill NaN with class mean, then perturb
                if seed_nan:
                    if class_means is not None and not np.isnan(class_means[j]):
                        # Use class mean as base value
                        synthetic[j] = class_means[j] + random_state.normal(0, bandwidths[j])
                    else:
                        # No valid mean available; preserve NaN
                        synthetic[j] = np.nan
                else:
                    synthetic[j] = seed[j] + random_state.normal(0, bandwidths[j])
                    
            elif self.nan_handling == 'random_pattern':
                # Randomly decide NaN based on class-level NaN rate
                if seed_nan:
                    # Seed is NaN; randomly decide whether to fill or preserve
                    if nan_rates is not None and random_state.random() > nan_rates[j]:
                        # Fill with class mean + perturbation
                        if class_means is not None and not np.isnan(class_means[j]):
                            synthetic[j] = class_means[j] + random_state.normal(0, bandwidths[j])
                        else:
                            synthetic[j] = np.nan
                    else:
                        synthetic[j] = np.nan
                else:
                    # Seed has value; perturb it
                    synthetic[j] = seed[j] + random_state.normal(0, bandwidths[j])
        
        return synthetic
    
    def _generate_samples_batch(self, X_class: np.ndarray, bandwidths: np.ndarray,
                               n_samples: int, random_state,
                               nan_rates: Optional[np.ndarray] = None,
                               class_means: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a batch of synthetic samples.
        
        Used for parallel processing.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of one class.
        bandwidths : np.ndarray
            Per-feature bandwidth values.
        n_samples : int
            Number of synthetic samples to generate.
        random_state : RandomState
            Random state for reproducibility.
        nan_rates : np.ndarray, optional
            Per-feature NaN rates.
        class_means : np.ndarray, optional
            Per-feature means.
        
        Returns
        -------
        np.ndarray
            Array of synthetic samples, shape (n_samples, n_features).
        """
        n_minority = len(X_class)
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Randomly select a seed sample
            seed_idx = random_state.randint(0, n_minority)
            seed = X_class[seed_idx]
            
            # Generate synthetic sample by perturbing the seed
            synthetic = self._generate_synthetic_sample_rose(
                seed, bandwidths, random_state, nan_rates, class_means
            )
            synthetic_samples.append(synthetic)
        
        return np.array(synthetic_samples)
    
    def _resample_class(self, X_class: np.ndarray, X: np.ndarray, 
                       y: np.ndarray, cls, n_samples: int,
                       random_state) -> np.ndarray:
        """
        Generate synthetic samples for a class using ROSE algorithm.
        
        Parameters
        ----------
        X_class : np.ndarray
            Feature matrix for samples of the target class.
        X : np.ndarray
            Full feature matrix (unused in ROSE).
        y : np.ndarray
            Full target array (unused in ROSE).
        cls : any
            Class label being resampled.
        n_samples : int
            Number of synthetic samples to generate.
        random_state : RandomState
            Random state for reproducibility.
        
        Returns
        -------
        np.ndarray
            Array of synthetic samples.
        """
        # Compute bandwidth using Silverman's rule
        bandwidths = self._compute_bandwidth(X_class)
        
        # Precompute NaN rates and class means if needed
        nan_rates = None
        class_means = None
        if self.nan_handling in ['interpolate', 'random_pattern']:
            class_means = self._compute_class_means(X_class)
        if self.nan_handling == 'random_pattern':
            nan_rates = self._compute_class_nan_rates(X_class)
        
        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential processing
            synthetic_samples = self._generate_samples_batch(
                X_class, bandwidths, n_samples, random_state, nan_rates, class_means
            )
        else:
            # Parallel processing: split samples across jobs
            from joblib import cpu_count
            n_jobs_actual = self.n_jobs if self.n_jobs > 0 else cpu_count()
            samples_per_job = n_samples // n_jobs_actual
            extra_samples = n_samples % n_jobs_actual
            
            # Distribute samples across jobs
            job_sizes = [samples_per_job + (1 if i < extra_samples else 0) 
                        for i in range(n_jobs_actual)]
            job_sizes = [s for s in job_sizes if s > 0]
            
            # Generate independent seeds for parallel jobs
            # Use 2**31 - 1 for Windows int32 compatibility
            seeds = random_state.randint(0, 2**31 - 1, size=len(job_sizes))
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_samples_batch)(
                    X_class, bandwidths, job_sizes[i], check_random_state(seeds[i]),
                    nan_rates, class_means
                )
                for i in range(len(job_sizes))
            )
            
            synthetic_samples = np.vstack(results)
        
        return synthetic_samples


# =============================================================================
# Unified OverNaN Interface
# =============================================================================

class OverNaN(BaseOverSamplerNaN):
    """
    Unified interface for NaN-aware oversampling algorithms.
    
    Provides access to SMOTE, ADASYN, and ROSE algorithms through a single
    class with consistent API.
    
    Parameters
    ----------
    method : str, default='SMOTE'
        Oversampling algorithm to use: 'SMOTE', 'ADASYN', or 'ROSE'.
    
    neighbours : int, default=5
        Number of nearest neighbors (SMOTE and ADASYN only).
    
    sampling_strategy : float, str, or dict, default='auto'
        Sampling information to resample the data set.
    
    random_state : int or RandomState, default=None
        Random state for reproducibility.
    
    nan_handling : str, default='preserve_pattern'
        Strategy for handling NaN in synthetic samples:
        - 'preserve_pattern': Preserve NaN conservatively
        - 'interpolate': Use available values when possible
        - 'random_pattern': Randomly choose NaN pattern
    
    n_jobs : int, default=None
        Number of parallel jobs. None means 1, -1 means all processors.
    
    **kwargs : dict
        Additional parameters for specific methods:
        - ADASYN: beta (float), learning_rate (float)
        - ROSE: shrinkage (float)
    
    Examples
    --------
    >>> from overnan import OverNaN
    >>> import numpy as np
    >>> 
    >>> # Create imbalanced data with NaN values
    >>> X = np.array([[1, 2, np.nan], [3, np.nan, 4], [5, 6, 7], 
    ...               [8, 9, 10], [11, 12, 13]])
    >>> y = np.array([0, 0, 0, 0, 1])
    >>> 
    >>> # Use SMOTE with NaN handling
    >>> oversampler = OverNaN(method='SMOTE', neighbours=2, random_state=42)
    >>> X_resampled, y_resampled = oversampler.fit_resample(X, y)
    >>> 
    >>> # Use ADASYN with custom parameters
    >>> oversampler = OverNaN(method='ADASYN', neighbours=3, beta=0.5, random_state=42)
    >>> X_resampled, y_resampled = oversampler.fit_resample(X, y)
    >>> 
    >>> # Use ROSE (no neighbors required)
    >>> oversampler = OverNaN(method='ROSE', shrinkage=1.0, random_state=42)
    >>> X_resampled, y_resampled = oversampler.fit_resample(X, y)
    """
    
    def __init__(self,
                 method: Literal['SMOTE', 'ADASYN', 'ROSE'] = 'SMOTE',
                 neighbours: int = 5,
                 sampling_strategy: Union[str, float, Dict] = 'auto',
                 random_state: Optional[int] = None,
                 nan_handling: Literal['preserve_pattern', 'interpolate', 'random_pattern'] = 'preserve_pattern',
                 n_jobs: Optional[int] = None,
                 **kwargs):
        """
        Initialize OverNaN with specified method and parameters.
        """
        super().__init__(sampling_strategy, random_state, n_jobs)
        self.method = method.upper()
        self.neighbours = neighbours
        self.nan_handling = nan_handling
        
        # Store method-specific parameters
        self.beta = kwargs.get('beta', 1.0)
        self.learning_rate = kwargs.get('learning_rate', 1.0)
        self.shrinkage = kwargs.get('shrinkage', 1.0)
        
        # Initialize the appropriate oversampler
        if self.method == 'SMOTE':
            self._oversampler = SMOTENaN(
                neighbours=neighbours,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                nan_handling=nan_handling,
                n_jobs=n_jobs
            )
        elif self.method == 'ADASYN':
            self._oversampler = ADASYNNaN(
                neighbours=neighbours,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                nan_handling=nan_handling,
                n_jobs=n_jobs,
                beta=self.beta,
                learning_rate=self.learning_rate
            )
        elif self.method == 'ROSE':
            self._oversampler = ROSENaN(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                nan_handling=nan_handling,
                n_jobs=n_jobs,
                shrinkage=self.shrinkage
            )
        else:
            raise ValueError(
                f"Invalid method: '{method}'. Choose 'SMOTE', 'ADASYN', or 'ROSE'."
            )
    
    def _resample_class(self, X_class: np.ndarray, X: np.ndarray, 
                       y: np.ndarray, cls, n_samples: int,
                       random_state) -> np.ndarray:
        """
        Delegate to underlying oversampler.
        
        This method exists to satisfy the abstract base class requirement
        but is not called directly when using OverNaN.
        """
        return self._oversampler._resample_class(
            X_class, X, y, cls, n_samples, random_state
        )
    
    def fit_resample(self, X, y):
        """
        Resample the dataset using the selected method.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (can contain NaN values).
        y : array-like of shape (n_samples,)
            Target labels.
        
        Returns
        -------
        X_resampled : array-like of shape (n_samples_new, n_features)
            Resampled feature matrix.
        y_resampled : array-like of shape (n_samples_new,)
            Resampled target labels.
        """
        return self._oversampler.fit_resample(X, y)
    
    def __repr__(self):
        """Return string representation of the oversampler."""
        params = [
            f"method='{self.method}'",
            f"sampling_strategy={self.sampling_strategy!r}",
            f"nan_handling='{self.nan_handling}'"
        ]
        
        # Add method-specific parameters
        if self.method in ['SMOTE', 'ADASYN']:
            params.insert(1, f"neighbours={self.neighbours}")
        
        if self.method == 'ADASYN':
            params.append(f"beta={self.beta}")
            params.append(f"learning_rate={self.learning_rate}")
        
        if self.method == 'ROSE':
            params.append(f"shrinkage={self.shrinkage}")
        
        if self.n_jobs is not None:
            params.append(f"n_jobs={self.n_jobs}")
        
        if self.random_state is not None:
            params.append(f"random_state={self.random_state}")
        
        return f"OverNaN({', '.join(params)})"
