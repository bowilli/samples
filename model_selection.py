"""
Enhanced Model Selection with Hyperparameter Tuning and MAPE-based Selection
"""

import os
import json
import hashlib
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


def calculate_comprehensive_metrics(y_true, y_pred, n_features=None):
    """
    Calculate all four key metrics: MAPE, MAE, RMSE, and Adjusted R²
    
    Args:
        y_true: Actual values
        y_pred: Predicted values  
        n_features: Number of features used (for adjusted R² calculation)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    
    # Median Absolute Percentage Error (MAPE) - Primary metric
    try:
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.median(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100)
        else:
            mape = np.nan
    except:
        mape = np.nan
    
    # Median Absolute Error (MAE) - Tie-breaker metric
    mae = np.median(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Regular R²
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R² (if n_features provided)
    if n_features is not None and len(y_true) > n_features + 1:
        n_samples = len(y_true)
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    else:
        adjusted_r2 = r2  # Fallback to regular R²
    
    return {
        'mape': mape,
        'mae': mae, 
        'rmse': rmse,
        'r2_score': r2,
        'adjusted_r2': adjusted_r2,
        'mse': mean_squared_error(y_true, y_pred)
    }


def mape_scorer(y_true, y_pred):
    """Custom scorer for MAPE (lower is better, so we negate it for sklearn)."""
    try:
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.median(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100)
            return -mape  # Negative because sklearn maximizes scores
        else:
            return -np.inf  # Worst possible score if all actuals are zero
    except:
        return -np.inf


def get_hyperparameter_grids():
    """
    Define hyperparameter grids for different model types.
    
    Loss Function Strategy:
    - CatBoost: Test MAE loss (aligns with our MAE tie-breaker) vs RMSE (current)
    - Random Forest: Test different splitting criteria
    
    Returns:
        dict: Model type -> parameter grid mapping
    """
    return {
        'random_forest': {
            'n_estimators': [200, 500],  # Reduced from 3 to 2 options
            'max_depth': [20, None],     # Reduced from 3 to 2 options  
            'min_samples_split': [2, 5], # Reduced from 3 to 2 options
            'criterion': ['squared_error', 'absolute_error']  # Keep both for loss alignment
            # Total: 2×2×2×2 = 16 combinations (vs 288)
        },
        'catboost': {
            'iterations': [1000],        # Fixed to best performing value
            'learning_rate': [0.1],      # Fixed to best performing value
            'depth': [6, 8],            # Reduced from 3 to 2 options
            'l2_leaf_reg': [1, 3],      # Reduced from 3 to 2 options  
            'loss_function': ['RMSE', 'MAE']  # Keep both for loss alignment
            # Total: 1×1×2×2×2 = 8 combinations (vs 162)
        },
        'linear_regression': {
            # Linear regression has no hyperparameters to tune
            # Note: Could add Ridge/Lasso variants in the future with alpha parameter
        }
    }


def _ensure_cache_directory():
    """Create cache directory if it doesn't exist."""
    cache_dir = os.path.join('data', 'model_cache', 'hyperparameters')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _generate_cache_key(model_type, X_shape, y_stats, param_grid):
    """
    Generate a unique cache key based on model type, data characteristics, and parameter grid.
    
    Args:
        model_type: Type of model ('random_forest', 'catboost', etc.)
        X_shape: Shape of feature matrix (n_samples, n_features)
        y_stats: Basic statistics of target variable (mean, std, min, max)
        param_grid: Parameter grid being searched
        
    Returns:
        str: Unique cache key
    """
    # Create a dictionary with all relevant information
    cache_data = {
        'model_type': model_type,
        'data_shape': X_shape,
        'target_stats': y_stats,
        'param_grid': param_grid
    }
    
    # Convert to JSON string and create hash
    cache_string = json.dumps(cache_data, sort_keys=True, default=str)
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
    
    return f"{model_type}_{cache_hash}"


def save_hyperparameters_to_cache(cache_key, best_params, best_score, tuning_results):
    """
    Save best hyperparameters to cache.
    
    Args:
        cache_key: Unique identifier for this hyperparameter set
        best_params: Best parameters found by grid search
        best_score: Best score achieved
        tuning_results: Additional tuning metadata
    """
    cache_dir = _ensure_cache_directory()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    cache_data = {
        'best_params': best_params,
        'best_score': best_score,
        'tuning_results': tuning_results,
        'cached_at': datetime.now().isoformat(),
        'cache_key': cache_key
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2, default=str)
    
    return cache_file


def load_hyperparameters_from_cache(cache_key):
    """
    Load hyperparameters from cache.
    
    Args:
        cache_key: Unique identifier for hyperparameter set
        
    Returns:
        tuple: (best_params, best_score, tuning_results, cache_info) or (None, None, None, None) if not found
    """
    cache_dir = _ensure_cache_directory()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None, None, None, None
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        return (
            cache_data.get('best_params', {}),
            cache_data.get('best_score'),
            cache_data.get('tuning_results', {}),
            {
                'cached_at': cache_data.get('cached_at'),
                'cache_file': cache_file
            }
        )
    except Exception as e:
        print(f"Warning: Failed to load cache {cache_file}: {e}")
        return None, None, None, None


def list_cached_hyperparameters():
    """
    List all cached hyperparameter sets.
    
    Returns:
        list: List of dictionaries with cache information
    """
    cache_dir = _ensure_cache_directory()
    cached_files = []
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('.json'):
            cache_file = os.path.join(cache_dir, filename)
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cached_files.append({
                    'cache_key': cache_data.get('cache_key', filename[:-5]),  # Remove .json
                    'cached_at': cache_data.get('cached_at'),
                    'best_score': cache_data.get('best_score'),
                    'file_path': cache_file
                })
            except Exception as e:
                print(f"Warning: Failed to read cache file {cache_file}: {e}")
    
    return sorted(cached_files, key=lambda x: x.get('cached_at', ''), reverse=True)


def clear_hyperparameter_cache():
    """Clear all cached hyperparameters."""
    cache_dir = _ensure_cache_directory()
    cleared_count = 0
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('.json'):
            cache_file = os.path.join(cache_dir, filename)
            try:
                os.remove(cache_file)
                cleared_count += 1
            except Exception as e:
                print(f"Warning: Failed to remove cache file {cache_file}: {e}")
    
    print(f"Cleared {cleared_count} cached hyperparameter files")
    return cleared_count


def get_model_with_cached_hyperparameters(model_instance, model_type, cache_key=None, verbose=False):
    """
    Get a model with cached hyperparameters applied (for use in main.py).
    
    Args:
        model_instance: Base model instance
        model_type: Type of model ('random_forest', 'catboost', etc.)
        cache_key: Specific cache key to use (if None, will be generated)
        verbose: Whether to print loading details
        
    Returns:
        tuple: (configured_model, used_cached_params)
    """
    param_grids = get_hyperparameter_grids()
    param_grid = param_grids.get(model_type, {})
    
    if not param_grid:
        # No hyperparameters to cache for this model type
        return model_instance, False
    
    # If no cache_key provided, we can't load from cache
    if cache_key is None:
        if verbose:
            print(f"No cache key provided for {model_type}, using default parameters")
        return model_instance, False
    
    # Try to load cached parameters
    cached_params, cached_score, cached_tuning_results, cache_info = load_hyperparameters_from_cache(cache_key)
    
    if cached_params is not None:
        try:
            model_instance.set_params(**cached_params)
            if verbose:
                print(f"Applied cached hyperparameters to {model_type}: {cached_params}")
            return model_instance, True
        except Exception as e:
            if verbose:
                print(f"Failed to apply cached parameters to {model_type}: {e}")
            return model_instance, False
    else:
        if verbose:
            print(f"No cached hyperparameters found for {model_type} with key {cache_key}")
        return model_instance, False


def perform_hyperparameter_tuning(model, model_type, X_train, y_train, cv=5, verbose=True, use_cache=True, force_retune=False):
    """
    Perform hyperparameter tuning using GridSearchCV with MAPE as the primary metric.
    Supports caching of hyperparameters to avoid retuning on subsequent runs.
    
    Args:
        model: Base model to tune
        model_type: Type of model ('random_forest', 'catboost', 'linear_regression')
        X_train: Training features
        y_train: Training targets
        cv: Number of cross-validation folds
        verbose: Whether to print tuning progress
        use_cache: Whether to use cached hyperparameters if available
        force_retune: If True, ignore cache and retune hyperparameters
        
    Returns:
        tuple: (best_model, best_params, best_score, tuning_results)
    """
    
    param_grids = get_hyperparameter_grids()
    param_grid = param_grids.get(model_type, {})
    
    if not param_grid:
        if verbose:
            print(f"No hyperparameters to tune for {model_type}")
        # Fit the model even if there are no hyperparameters to tune
        model.fit(X_train, y_train)
        return model, {}, None, {}
    
    # Generate cache key based on data characteristics
    y_stats = {
        'mean': float(np.mean(y_train)),
        'std': float(np.std(y_train)),
        'min': float(np.min(y_train)),
        'max': float(np.max(y_train))
    }
    cache_key = _generate_cache_key(model_type, X_train.shape, y_stats, param_grid)
    
    # Try to load from cache first (unless force_retune is True)
    if use_cache and not force_retune:
        cached_params, cached_score, cached_tuning_results, cache_info = load_hyperparameters_from_cache(cache_key)
        
        if cached_params is not None:
            if verbose:
                print(f"🔄 Using cached hyperparameters for {model_type}")
                if cache_info and cache_info.get('cached_at'):
                    print(f"  Cached at: {cache_info['cached_at']}")
                if cached_score is not None:
                    print(f"  Cached MAPE score: {-cached_score:.2f}%")
                print(f"  Parameters: {cached_params}")
            
            # Apply cached parameters to model and fit
            try:
                model.set_params(**cached_params)
                model.fit(X_train, y_train)
                return model, cached_params, cached_score, cached_tuning_results
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ Failed to apply cached parameters: {e}")
                    print(f"  Proceeding with hyperparameter tuning...")
    
    # Perform hyperparameter tuning (either no cache available or force_retune=True)
    # Calculate grid dimensions
    n_combinations = 1
    for key, values in param_grid.items():
        n_combinations *= len(values)
    total_fits = n_combinations * cv
    
    if verbose:
        action = "Re-tuning" if force_retune else "Tuning"
        print(f"🔧 {action} hyperparameters for {model_type}...")
        print(f"  Parameter grid: {param_grid}")
        print(f"  Total combinations: {n_combinations} (×{cv} CV folds = {total_fits} fits)")
    
    # Start timing
    start_time = time.time()
    
    # Create custom MAPE scorer
    mape_scoring = make_scorer(mape_scorer, greater_is_better=True)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring=mape_scoring,  # Use MAPE as primary tuning metric
        cv=cv, 
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print(f"  ⏱️  Starting grid search at {datetime.now().strftime('%H:%M:%S')}...")
    
    grid_search.fit(X_train, y_train)
    
    # End timing
    end_time = time.time()
    duration = end_time - start_time
    
    if verbose:
        print(f"  ✅ Hyperparameter tuning completed!")
        print(f"  ⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"  🎯 Best parameters: {grid_search.best_params_}")
        print(f"  📊 Best MAPE score: {-grid_search.best_score_:.2f}%")  # Convert back to positive MAPE
        print(f"  ⚡ Performance: {total_fits/duration:.1f} fits/second")
    
    # Compile detailed results
    tuning_results = {
        'best_score_mape': -grid_search.best_score_,  # Convert back to positive MAPE
        'cv_results': grid_search.cv_results_,
        'n_candidates': len(grid_search.cv_results_['params']),
        'tuning_duration_seconds': duration,
        'tuning_duration_minutes': duration / 60,
        'fits_per_second': total_fits / duration if duration > 0 else 0,
        'total_fits': total_fits,
        'tuned_at': datetime.now().isoformat()
    }
    
    # Save to cache for future use
    if use_cache:
        cache_file = save_hyperparameters_to_cache(cache_key, grid_search.best_params_, grid_search.best_score_, tuning_results)
        if verbose:
            print(f"  💾 Saved hyperparameters to cache: {os.path.basename(cache_file)}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, tuning_results


def evaluate_model_comprehensive(model, model_name, X_train, y_train, X_test, y_test, n_features=None):
    """
    Comprehensive model evaluation using all four key metrics.
    
    Args:
        model: Trained model
        model_name: Name/identifier for the model
        X_train: Training features
        y_train: Training targets
        X_test: Test features  
        y_test: Test targets
        n_features: Number of features (for adjusted R²)
        
    Returns:
        dict: Comprehensive evaluation results
    """
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for both training and test sets
    train_metrics = calculate_comprehensive_metrics(y_train, y_train_pred, n_features)
    test_metrics = calculate_comprehensive_metrics(y_test, y_test_pred, n_features)
    
    return {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model': model
    }


def evaluate_model_for_selection(model, model_name, X_train, y_train, X_val, y_val, n_features=None):
    """
    Model evaluation for model selection (uses validation set, NOT test set).
    
    Args:
        model: Trained model
        model_name: Name/identifier for the model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (for model selection)
        y_val: Validation targets (for model selection)
        n_features: Number of features (for adjusted R²)
        
    Returns:
        dict: Evaluation results for model selection
    """
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics for training and validation sets
    train_metrics = calculate_comprehensive_metrics(y_train, y_train_pred, n_features)
    val_metrics = calculate_comprehensive_metrics(y_val, y_val_pred, n_features)
    
    return {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,  # Use validation metrics for model selection
        'model': model
    }


def select_best_models_by_mape(evaluation_results, verbose=True, use_validation_metrics=True):
    """
    Select the single best model based on MAPE (primary) and MAE (tie-breaker).
    
    Selection Logic:
    1. Primary: Minimum validation MAPE (lower is better) - PROPER ML PRACTICE
    2. Tie-breaker: If multiple models have identical MAPE, use minimum validation MAE
    3. Fallback: If no valid MAPE, use maximum validation adjusted R²
    
    Args:
        evaluation_results: List of evaluation result dictionaries
        verbose: Whether to print selection details
        use_validation_metrics: If True, use validation metrics (proper). If False, use test metrics (legacy)
        
    Returns:
        dict: Best model result
    """
    
    if not evaluation_results:
        return None
    
    # Convert to DataFrame for easier analysis
    results_data = []
    for result in evaluation_results:
        # Use validation metrics for model selection (proper ML practice)
        if use_validation_metrics and 'val_metrics' in result:
            metrics_source = result['val_metrics']
            metrics_label = "validation"
        else:
            # Fallback to test metrics (legacy behavior)
            metrics_source = result.get('test_metrics', result.get('val_metrics', {}))
            metrics_label = "test" if 'test_metrics' in result else "validation"
            
        results_data.append({
            'model_name': result['model_name'],
            'model': result['model'],
            'mape': metrics_source.get('mape'),
            'mae': metrics_source.get('mae'),
            'rmse': metrics_source.get('rmse'),
            'adjusted_r2': metrics_source.get('adjusted_r2'),
            'r2_score': metrics_source.get('r2_score'),
            'mse': metrics_source.get('mse'),
            'metrics_source': metrics_label
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Filter out models with NaN MAPE
    valid_mape_df = results_df.dropna(subset=['mape'])
    
    if not valid_mape_df.empty:
        # Find minimum MAPE (primary criterion)
        min_mape = valid_mape_df['mape'].min()
        best_mape_models = valid_mape_df[valid_mape_df['mape'] == min_mape]
        
        if len(best_mape_models) > 1:
            # Tie-breaker: Use MAE (lower is better)
            tied_models_with_mae = best_mape_models.dropna(subset=['mae'])
            if not tied_models_with_mae.empty:
                best_idx = tied_models_with_mae['mae'].idxmin()
                tie_breaker_used = True
            else:
                best_idx = best_mape_models.index[0]
                tie_breaker_used = False
        else:
            best_idx = best_mape_models.index[0]
            tie_breaker_used = False
        
        best_model = results_df.loc[best_idx]
        
        if verbose:
            metrics_source = best_model.get('metrics_source', 'unknown')
            print(f"\n🏆 Best Model Selected: {best_model['model_name']}")
            print(f"  Selection based on {metrics_source} set performance (proper ML practice)")
            print(f"  MAPE: {best_model['mape']:.1f}% (primary criterion)")
            if tie_breaker_used:
                print(f"  MAE: ${best_model['mae']:,.0f} (tie-breaker - multiple models had same MAPE)")
            else:
                print(f"  MAE: ${best_model['mae']:,.0f}")
            print(f"  RMSE: ${best_model['rmse']:,.0f}")
            print(f"  Adjusted R²: {best_model['adjusted_r2']:.3f}")
            
    else:
        # Fallback: Use adjusted R² if no valid MAPE
        best_idx = results_df['adjusted_r2'].idxmax()
        best_model = results_df.loc[best_idx]
        
        if verbose:
            metrics_source = best_model.get('metrics_source', 'unknown')
            print(f"\n🏆 Best Model Selected (MAPE fallback): {best_model['model_name']}")
            print(f"  Selection based on {metrics_source} set performance")
            print(f"  Adjusted R²: {best_model['adjusted_r2']:.3f} (MAPE not available)")
            print(f"  MAE: ${best_model['mae']:,.0f}")
            print(f"  RMSE: ${best_model['rmse']:,.0f}")
    
    return best_model.to_dict()


def train_and_select_best_model(models_config, X, y, test_size=0.2, val_size=0.2, random_state=42, verbose=True, use_cache=True, force_retune=False):
    """
    Complete workflow: hyperparameter tuning + training + evaluation + selection.
    
    IMPORTANT: Uses proper train/validation/test split to avoid data leakage:
    - Training data: Used for hyperparameter tuning and model training
    - Validation data: Used for model selection (choosing best model)
    - Test data: Used ONLY for final evaluation (touched once)
    
    Args:
        models_config: Dict of {model_name: (model_instance, model_type)} 
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing (final evaluation only)
        val_size: Proportion of remaining data for validation (model selection)
        random_state: Random seed for reproducibility
        verbose: Whether to print detailed progress
        use_cache: Whether to use cached hyperparameters if available
        force_retune: If True, ignore cache and retune all hyperparameters
        
    Returns:
        dict: Complete results including best model and all evaluations
    """
    
    # First split: separate test set (for final evaluation only)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation sets
    # val_size_adjusted accounts for the fact we already removed test_size
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    if verbose:
        print(f"📊 Training Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"📊 Validation Data: {X_val.shape[0]} samples (for model selection)")
        print(f"📊 Test Data: {X_test.shape[0]} samples (for final evaluation only)")
        if use_cache and not force_retune:
            print(f"🔄 Using cached hyperparameters when available")
        elif force_retune:
            print(f"🔄 Force retuning all hyperparameters (ignoring cache)")
    
    evaluation_results = []
    
    # Train and evaluate each model
    for model_name, (model_instance, model_type) in models_config.items():
        if verbose:
            print(f"\n🔧 Processing {model_name} ({model_type})...")
        
        try:
            # Hyperparameter tuning with cache support
            tuned_model, best_params, best_score, tuning_results = perform_hyperparameter_tuning(
                model_instance, model_type, X_train, y_train, 
                verbose=verbose, use_cache=use_cache, force_retune=force_retune
            )
            
            # Model evaluation for selection (uses validation set)
            eval_result = evaluate_model_for_selection(
                tuned_model, model_name, X_train, y_train, X_val, y_val, X.shape[1]
            )
            
            # Add tuning information
            eval_result['hyperparameter_tuning'] = {
                'best_params': best_params,
                'tuning_results': tuning_results
            }
            
            evaluation_results.append(eval_result)
            
            if verbose:
                val_metrics = eval_result['val_metrics']
                print(f"  ✅ Completed - Validation MAPE: {val_metrics['mape']:.1f}%, MAE: ${val_metrics['mae']:,.0f}")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed: {str(e)}")
            continue
    
    # Select best model based on validation performance
    if verbose:
        print(f"\n🎯 Model Selection (MAPE-based on validation set)...")
    
    best_model_result = select_best_models_by_mape(evaluation_results, verbose=verbose, use_validation_metrics=True)
    
    # Final evaluation of best model on test set (touched only once!)
    if best_model_result and verbose:
        print(f"\n📊 Final Test Set Evaluation (best model only)...")
        best_model_obj = best_model_result['model']
        final_eval = evaluate_model_comprehensive(
            best_model_obj, best_model_result['model_name'], X_train, y_train, X_test, y_test, X.shape[1]
        )
        
        test_metrics = final_eval['test_metrics']
        print(f"🎯 Final Test Performance ({best_model_result['model_name']}):")
        print(f"  Test MAPE: {test_metrics['mape']:.1f}%")
        print(f"  Test MAE: ${test_metrics['mae']:,.0f}")
        print(f"  Test RMSE: ${test_metrics['rmse']:,.0f}")
        print(f"  Test Adjusted R²: {test_metrics['adjusted_r2']:.3f}")
        
        # Add final test metrics to result
        best_model_result['final_test_metrics'] = test_metrics
    
    return {
        'best_model': best_model_result,
        'all_evaluations': evaluation_results,
        'training_data_shape': X_train.shape,
        'validation_data_shape': X_val.shape,
        'test_data_shape': X_test.shape
    }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Enhanced Model Selection System...")
    
    # Generate sample data (more realistic for cost prediction)
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear relationships similar to cost prediction:
    # - Non-linear interactions between features
    # - Exponential scaling 
    # - Some categorical-like effects
    base_cost = 100000  # Base cost
    
    # Complex non-linear relationship (more realistic for cost prediction)
    y = (
        base_cost +                                    # Base cost
        X[:, 0] * 50000 +                             # Linear component (team effect)
        (X[:, 1] ** 2) * 30000 +                      # Quadratic (runtime/shooting days)
        np.maximum(0, X[:, 2]) * 100000 +             # Positive effects only (cast count)
        (X[:, 3] * X[:, 4]) * 25000 +                 # Interaction effects
        np.exp(np.clip(X[:, 5] * 0.5, -2, 2)) * 40000 + # Exponential scaling
        (np.sin(X[:, 6]) * 20000) +                   # Cyclical patterns
        np.random.lognormal(0, 0.3) * 50000           # Log-normal noise (realistic for costs)
    )
    
    # Ensure positive costs and add some realistic outliers
    y = np.maximum(y, 10000)  # Minimum cost
    outlier_mask = np.random.random(n_samples) < 0.05  # 5% outliers
    y[outlier_mask] *= np.random.uniform(3, 10, outlier_mask.sum())  # High-cost outliers
    
    # Define models to test
    models_config = {
        'Linear_Regression': (LinearRegression(), 'linear_regression'),
        'Random_Forest': (RandomForestRegressor(random_state=42), 'random_forest'),
        'CatBoost': (CatBoostRegressor(random_seed=42, verbose=False), 'catboost')
    }
    
    # Run complete workflow (first run will cache hyperparameters)
    print("\n=== First Run (will cache hyperparameters) ===")
    results = train_and_select_best_model(models_config, X, y, verbose=True, use_cache=True)
    
    print("\n=== Second Run (should use cached hyperparameters) ===")
    results2 = train_and_select_best_model(models_config, X, y, verbose=True, use_cache=True)
    
    print(f"\n📈 Final Results Summary:")
    best = results['best_model']
    print(f"Best Model: {best['model_name']}")
    print(f"Selection Validation MAPE: {best['mape']:.1f}% (used for model selection)")
    print(f"Selection Validation MAE: ${best['mae']:,.0f}")
    
    # Show final test performance if available
    if 'final_test_metrics' in best:
        test_metrics = best['final_test_metrics']
        print(f"\nFinal Test Performance (unbiased estimate):")
        print(f"Test MAPE: {test_metrics['mape']:.1f}%")
        print(f"Test MAE: ${test_metrics['mae']:,.0f}")
        print(f"Test RMSE: ${test_metrics['rmse']:,.0f}")
        print(f"Test Adjusted R²: {test_metrics['adjusted_r2']:.3f}")
    
    print(f"\nData Split Summary:")
    print(f"Training: {results['training_data_shape'][0]} samples")
    print(f"Validation: {results['validation_data_shape'][0]} samples (for model selection)")
    print(f"Test: {results['test_data_shape'][0]} samples (final evaluation only)")
