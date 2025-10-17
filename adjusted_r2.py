#!/usr/bin/env python3
"""
Adjusted R² Calculation Module
=============================

Provides functions to calculate adjusted R-squared instead of regular R-squared.
Adjusted R² accounts for model complexity and provides a more accurate assessment
of model performance by penalizing overfitting.

Formula: Adjusted R² = 1 - ((1 - R²) * (n - 1)) / (n - k - 1)
Where:
- R² is the regular R-squared
- n is the total sample size 
- k is the number of features/predictors
"""

import numpy as np
from sklearn.metrics import r2_score


def calculate_adjusted_r2(y_true, y_pred, n_features):
    """
    Calculate adjusted R-squared given true values, predictions, and number of features
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values  
        n_features (int): Number of features/predictors in the model
        
    Returns:
        float: Adjusted R-squared value
    """
    # First calculate regular R²
    r2 = r2_score(y_true, y_pred)
    
    # Sample size
    n = len(y_true)
    
    # Cannot calculate adjusted R² if n <= k+1
    if n <= n_features + 1:
        return np.nan
    
    # Calculate adjusted R²
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - n_features - 1)
    
    return adjusted_r2


def calculate_adjusted_r2_from_r2(r2, n_samples, n_features):
    """
    Calculate adjusted R-squared from regular R-squared
    
    Args:
        r2 (float): Regular R-squared value
        n_samples (int): Total sample size
        n_features (int): Number of features/predictors
        
    Returns:
        float: Adjusted R-squared value
    """
    # Cannot calculate adjusted R² if n <= k+1
    if n_samples <= n_features + 1:
        return np.nan
    
    # Calculate adjusted R²
    adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1)) / (n_samples - n_features - 1)
    
    return adjusted_r2


def get_feature_count_from_data(X_data):
    """
    Get the number of features from the feature matrix
    
    Args:
        X_data: Feature matrix (pandas DataFrame or numpy array)
        
    Returns:
        int: Number of features
    """
    if hasattr(X_data, 'shape'):
        return X_data.shape[1] if len(X_data.shape) > 1 else 1
    else:
        # If it's a list or other iterable
        return len(X_data[0]) if len(X_data) > 0 else 0


def adjusted_r2_score(y_true, y_pred, X_features):
    """
    Drop-in replacement for sklearn's r2_score that returns adjusted R²
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values  
        X_features: Feature matrix to determine number of features
        
    Returns:
        float: Adjusted R-squared value
    """
    n_features = get_feature_count_from_data(X_features)
    return calculate_adjusted_r2(y_true, y_pred, n_features)


def estimate_feature_count_from_context(model_key=None, n_samples=None):
    """
    Estimate feature count based on model context when feature matrix is not available
    This is a fallback for legacy code that doesn't have access to feature matrices
    
    Args:
        model_key (str): Model key to help estimate features
        n_samples (int): Sample size to help with estimation
        
    Returns:
        int: Estimated number of features
    """
    # Default estimates based on feature set patterns
    if model_key and 'set_5' in model_key:
        return 85  # Exploratory base model
    elif model_key and 'set_6' in model_key:
        return 120  # Embeddings + algo comps
    elif model_key and 'set_1' in model_key:
        return 60  # Comps approach
    elif model_key and 'set_2' in model_key:
        return 90  # Embeddings approach
    elif model_key and 'set_3' in model_key:
        return 50  # Base model
    elif model_key and 'set_4' in model_key:
        return 70  # Comps embeddings
    else:
        # Conservative estimate - 10% of sample size, capped between 30-100
        if n_samples:
            estimated = max(30, min(100, int(n_samples * 0.1)))
            return estimated
        return 75  # Fallback default


def adjusted_r2_score_legacy(y_true, y_pred, model_key=None):
    """
    Calculate adjusted R² for legacy code that doesn't have feature matrices
    Uses estimated feature counts based on model context
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values  
        model_key (str): Model key to help estimate features
        
    Returns:
        float: Adjusted R-squared value
    """
    n_features = estimate_feature_count_from_context(model_key, len(y_true))
    return calculate_adjusted_r2(y_true, y_pred, n_features)


def print_r2_comparison(y_true, y_pred, X_features, model_name="Model"):
    """
    Print comparison between regular R² and adjusted R² for debugging
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values  
        X_features: Feature matrix
        model_name (str): Name of the model for display
    """
    regular_r2 = r2_score(y_true, y_pred)
    adjusted_r2 = adjusted_r2_score(y_true, y_pred, X_features)
    n_features = get_feature_count_from_data(X_features)
    n_samples = len(y_true)
    
    penalty = regular_r2 - adjusted_r2
    penalty_pct = (penalty / regular_r2) * 100 if regular_r2 != 0 else 0
    
    print(f"{model_name} R² Comparison:")
    print(f"  Regular R²: {regular_r2:.4f}")
    print(f"  Adjusted R²: {adjusted_r2:.4f}")
    print(f"  Penalty: {penalty:.4f} ({penalty_pct:.1f}% reduction)")
    print(f"  Features: {n_features}, Samples: {n_samples}")
    print()
