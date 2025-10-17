"""
Main script for machine learning model training and evaluation.

This script loads data, trains multiple models with different feature sets,
evaluates their performance, and generates comprehensive reports including
feature importance analysis and visualizations.
"""

from features.feature_selection import select_features, get_feature_set_description, FEATURE_SETS
from features.feature_engineering import encode_rare_categories, load_data, clean_data, handle_vfx_tiers, add_time_based_features, _process_outlier_detection
from features.utils.feature_reporter import save_processed_datasets
from models.model_definitions import linear_regression, random_forest, catboost
from models.model_wrappers import ModelWrapper, CatBoostWrapper
from models.model_selection import (
    train_and_select_best_model, list_cached_hyperparameters,
    clear_hyperparameter_cache, get_model_with_cached_hyperparameters,
    _generate_cache_key, get_hyperparameter_grids
)
from utils.model_evaluation import identify_best_models, save_best_model_predictions, calculate_metrics_by_group, calculate_error_rate_metrics, generate_intelligent_routing_report
from utils.feature_analysis import calculate_feature_importance_and_shap
from utils.visualization import create_summary_charts
from utils.intelligent_director_predictor import create_intelligent_prediction_demo

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils.adjusted_r2 import adjusted_r2_score
import numpy as np
import pandas as pd
import os
import shap
from shap.utils._exceptions import ExplainerError
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Fix numpy bool deprecation warning for SHAP
np.bool = bool


def _apply_cached_hyperparameters_to_model(model_func, X_train_encoded, y_train, retune_hyperparameters=False, verbose=False):
    """
    Apply cached hyperparameters to a model instance, with hyperparameter tuning when needed.
    
    Args:
        model_func: Model factory function (linear_regression, random_forest, catboost)
        X_train_encoded: Training features (used for tuning and cache key generation)
        y_train: Training target (used for tuning and cache key generation)
        retune_hyperparameters: If True, force hyperparameter retuning (ignore cache)
        verbose: Whether to print caching details
        
    Returns:
        Model instance with optimal hyperparameters applied
    """
    
    # Map model function names to model types
    model_type_mapping = {
        'linear_regression': 'linear_regression',
        'random_forest': 'random_forest',
        'catboost': 'catboost'
    }
    
    model_type = model_type_mapping.get(model_func.__name__)
    if not model_type:
        if verbose:
            print(f"Unknown model type for {model_func.__name__}, using default parameters")
        return model_func()
    
    # Check if this model type has hyperparameters to tune
    param_grids = get_hyperparameter_grids()
    param_grid = param_grids.get(model_type, {})
    
    if not param_grid:
        # No hyperparameters to tune for this model type
        if verbose:
            print(f"No hyperparameters to tune for {model_type}")
        return model_func()
    
    # Generate cache key based on data characteristics
    y_stats = {
        'mean': float(np.mean(y_train)),
        'std': float(np.std(y_train)),
        'min': float(np.min(y_train)),
        'max': float(np.max(y_train))
    }
    cache_key = _generate_cache_key(model_type, X_train_encoded.shape, y_stats, param_grid)
    
    # Try to load cached hyperparameters (unless force retuning)
    if not retune_hyperparameters:
        configured_model, used_cache = get_model_with_cached_hyperparameters(
            model_func(), model_type, cache_key, verbose=verbose
        )
        if used_cache:
            return configured_model
    
    # No cache available or force retuning - perform hyperparameter tuning
    if verbose:
        action = "Re-tuning" if retune_hyperparameters else "Tuning"
        print(f"🔧 {action} hyperparameters for {model_type}...")
    
    from models.model_selection import perform_hyperparameter_tuning
    
    # Create model wrapper and extract underlying estimator for hyperparameter tuning
    model_wrapper = model_func()
    
    # Extract the underlying sklearn estimator for GridSearchCV
    if hasattr(model_wrapper, 'get_model'):
        underlying_estimator = model_wrapper.get_model()
        # For TransformedTargetRegressor, we need the actual regressor
        if hasattr(underlying_estimator, 'regressor'):
            base_estimator = underlying_estimator.regressor
        else:
            base_estimator = underlying_estimator
    else:
        # If it's already a raw estimator
        base_estimator = model_wrapper
        underlying_estimator = model_wrapper
    
    # Perform hyperparameter tuning on the base estimator
    tuned_base_estimator, best_params, best_score, tuning_results = perform_hyperparameter_tuning(
        base_estimator, model_type, X_train_encoded, y_train, 
        verbose=verbose, use_cache=True, force_retune=retune_hyperparameters
    )
    
    # Apply the tuned parameters to the full model structure
    if hasattr(model_wrapper, 'get_model'):
        # Update the underlying model with tuned hyperparameters
        if hasattr(underlying_estimator, 'regressor'):
            # For TransformedTargetRegressor, update the regressor
            underlying_estimator.regressor = tuned_base_estimator
        else:
            # Direct replacement
            model_wrapper.model = tuned_base_estimator
        return model_wrapper
    else:
        # Return the tuned estimator directly
        return tuned_base_estimator


def _save_model_predictions_1301_format(model_cache, data, validation_ids, training_ids):
    """Save predictions for each model in 1301 format."""
    print("\n" + "="*60)
    print("SAVING MODEL PREDICTIONS IN 1301 FORMAT")
    print("="*60)

    os.makedirs('model_comparison/current_model_results', exist_ok=True)

    all_predictions = []

    for model_key, cache_entry in model_cache.items():
        if cache_entry['target'] != 'director_efc_usd':
            continue

        print(f"\nSaving predictions for {model_key}...")

        # Get predictions
        y_train = cache_entry['y_train']
        y_test = cache_entry['y_test']
        model = cache_entry['model']

        # Get training and test predictions
        if isinstance(model, CatBoostWrapper):
            train_pred = model.predict(cache_entry['X_train_catboost'])
            test_pred = model.predict(cache_entry['X_test_catboost'])
        else:
            train_pred = model.predict(cache_entry['X_train_encoded'])
            test_pred = model.predict(cache_entry['X_test_encoded'])

        # Get indices
        train_indices = y_train.index
        test_indices = y_test.index

        # Create dataframe for this model
        results = []

        # Add training predictions
        for idx, (actual, pred) in zip(train_indices, zip(y_train, train_pred)):
            season_id = data.loc[idx, 'season_production_id']
            buying_org = data.loc[idx, 'gravity_buying_organization_desc']
            abs_error = abs(actual - pred)
            pct_error = (abs_error / actual * 100) if actual != 0 else 0

            results.append({
                'season_production_id': season_id,
                'buying_org_name': buying_org,
                'dataset_type': 'training',
                'target': actual,
                'prediction': pred,
                'absolute_error': abs_error,
                'percentage_error': pct_error
            })

        # Add validation predictions
        for idx, (actual, pred) in zip(test_indices, zip(y_test, test_pred)):
            season_id = data.loc[idx, 'season_production_id']
            buying_org = data.loc[idx, 'gravity_buying_organization_desc']
            abs_error = abs(actual - pred)
            pct_error = (abs_error / actual * 100) if actual != 0 else 0

            results.append({
                'season_production_id': season_id,
                'buying_org_name': buying_org,
                'dataset_type': 'validation',
                'target': actual,
                'prediction': pred,
                'absolute_error': abs_error,
                'percentage_error': pct_error
            })

        # Save to CSV
        results_df = pd.DataFrame(results)
        model_name = cache_entry['model_func'].__name__
        feature_set = cache_entry['set_name']
        filename = f'model_comparison/current_model_results/{model_name}_{feature_set}.csv'
        results_df.to_csv(filename, index=False)
        print(f"   Saved {len(results_df)} predictions to {filename}")

        # Store for comparison summary
        all_predictions.append({
            'model_key': model_key,
            'model': model_name,
            'feature_set': feature_set,
            'results_df': results_df
        })

    return all_predictions


def _create_comparison_summary(all_predictions):
    """Create a comprehensive comparison summary across all models."""
    print("\n" + "="*60)
    print("CREATING MODEL COMPARISON SUMMARY")
    print("="*60)

    summary_rows = []

    for pred_info in all_predictions:
        model = pred_info['model']
        feature_set = pred_info['feature_set']
        results_df = pred_info['results_df']

        # Calculate metrics for both training and validation
        for dataset_type in ['training', 'validation']:
            subset = results_df[results_df['dataset_type'] == dataset_type]

            if len(subset) == 0:
                continue

            actual = subset['target']
            predicted = subset['prediction']

            # Calculate metrics
            mae = np.median(np.abs(actual - predicted))
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            r2 = adjusted_r2_score(actual, predicted, subset[['target']].values)

            # MAPE
            non_zero_mask = actual != 0
            if non_zero_mask.sum() > 0:
                mape = np.median(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]) * 100)
            else:
                mape = np.nan

            # Error rate metrics
            pct_error = subset['percentage_error']
            pct_under_20 = (pct_error < 20).sum() / len(pct_error) * 100
            pct_under_6 = (pct_error < 6).sum() / len(pct_error) * 100

            summary_rows.append({
                'model': model,
                'feature_set': feature_set,
                'dataset_type': dataset_type,
                'n_samples': len(subset),
                'mdae': mae,
                'rmse': rmse,
                'adjusted_r2_score': r2,
                'mdape': mape,
                'pct_titles_under_20_error': pct_under_20,
                'pct_titles_under_6_error': pct_under_6
            })

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_rows)

    # Save summary
    summary_file = 'model_comparison/reports/model_comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Saved comparison summary to {summary_file}")

    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)

    for dataset_type in ['training', 'validation']:
        print(f"\n{dataset_type.upper()} SET RESULTS:")
        print("-" * 60)
        subset = summary_df[summary_df['dataset_type'] == dataset_type]

        for _, row in subset.iterrows():
            print(f"\n{row['model']} - {row['feature_set']}:")
            print(f"  Samples: {row['n_samples']}")
            print(f"  MdAPE: {row['mdape']:.1f}%")
            print(f"  MdAE: ${row['mdae']:,.0f}")
            print(f"  RMSE: ${row['rmse']:,.0f}")
            print(f"  Adjusted R²: {row['adjusted_r2_score']:.3f}")
            print(f"  Titles <20% error: {row['pct_titles_under_20_error']:.1f}%")
            print(f"  Titles <6% error: {row['pct_titles_under_6_error']:.1f}%")

    return summary_df


def _load_1301_validation_ids():
    """Load validation season_production_ids from the 1301 predictions file."""
    validation_file = 'model_comparison/original_model_results/1301_predictions_results.csv'
    df = pd.read_csv(validation_file)
    validation_ids = df[df['dataset_type'] == 'validation']['season_production_id'].tolist()
    training_ids = df[df['dataset_type'] == 'training']['season_production_id'].tolist()
    print(f"\n📋 Loaded 1301 dataset split:")
    print(f"   Validation IDs: {len(validation_ids)}")
    print(f"   Training IDs: {len(training_ids)}")
    return validation_ids, training_ids


def _create_dataset_comparison_report(data, validation_ids, training_ids, target):
    """Create a comparison report between training and validation sets."""
    # Filter data to only include IDs from 1301 file
    validation_mask = data['season_production_id'].isin(validation_ids)
    training_mask_1301 = data['season_production_id'].isin(training_ids)

    validation_data = data[validation_mask]
    training_data_1301 = data[training_mask_1301]

    # ACTUAL training data = all data NOT in validation set
    actual_training_mask = ~validation_mask
    actual_training_data = data[actual_training_mask]

    report = []
    report.append("="*80)
    report.append("TRAINING VS VALIDATION SET COMPARISON")
    report.append("="*80)
    report.append(f"\nDataset Sizes:")
    report.append(f"  Validation set (matching 1301): {len(validation_data)} titles")
    report.append(f"  ACTUAL training set (all non-validation): {len(actual_training_data)} titles")
    report.append(f"  Original 1301 training set overlap: {len(training_data_1301)} titles")
    report.append(f"  Total in current scope: {len(data)} titles")
    report.append(f"  Total in original 1301 file: {len(training_data_1301) + len(validation_data)} titles")

    # Target variable statistics
    report.append(f"\nTarget Variable ({target}) Statistics:")
    report.append(f"  Training (actual) - Mean: ${actual_training_data[target].mean():,.0f}, Std: ${actual_training_data[target].std():,.0f}, Min: ${actual_training_data[target].min():,.0f}, Max: ${actual_training_data[target].max():,.0f}")
    report.append(f"  Validation (1301) - Mean: ${validation_data[target].mean():,.0f}, Std: ${validation_data[target].std():,.0f}, Min: ${validation_data[target].min():,.0f}, Max: ${validation_data[target].max():,.0f}")

    # Organization distribution
    report.append(f"\nOrganization Distribution:")
    train_orgs = actual_training_data['gravity_buying_organization_desc'].value_counts()
    val_orgs = validation_data['gravity_buying_organization_desc'].value_counts()
    all_orgs = set(train_orgs.index) | set(val_orgs.index)
    for org in sorted(all_orgs):
        train_count = train_orgs.get(org, 0)
        val_count = val_orgs.get(org, 0)
        report.append(f"  {org}: Training={train_count}, Validation={val_count}")

    report.append("="*80)

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    os.makedirs('model_comparison/reports', exist_ok=True)
    with open('model_comparison/reports/dataset_comparison_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\n💾 Saved dataset comparison report to model_comparison/reports/dataset_comparison_report.txt")

    return validation_data, actual_training_data


def main(retune_hyperparameters=False, use_1301_validation=False):
    """Main execution function."""
    import time
    from datetime import datetime

    main_start_time = time.time()
    print(f"🚀 Starting model training and evaluation at {datetime.now().strftime('%H:%M:%S')}...")
    if retune_hyperparameters:
        print(f"🔄 Force retuning hyperparameters (ignoring cache)")
    else:
        print(f"🔄 Using cached hyperparameters when available")


    if use_1301_validation:
        print(f"📊 Using 1301 validation set for director models comparison")

    # Load datasets - separate optimized datasets for each target
    print("Loading separate optimized datasets...")
    atl_data = pd.read_csv('data/processed_data/atl_title_data.csv')
    director_data = pd.read_csv('data/processed_data/director_title_data.csv')

    print(f"ATL dataset: {len(atl_data)} titles")
    print(f"Director dataset: {len(director_data)} titles")

    # Load 1301 validation IDs if requested
    validation_ids_1301 = None
    training_ids_1301 = None
    if use_1301_validation:
        validation_ids_1301, training_ids_1301 = _load_1301_validation_ids()
        # Create comparison report
        _create_dataset_comparison_report(director_data, validation_ids_1301, training_ids_1301, 'director_efc_usd')

    # Initialize lists to store all results
    model_results = []
    all_org_metrics = []
    all_team_metrics = []
    model_cache = {}  # Cache trained models for later use

    # Define targets and models to test
    if use_1301_validation:
        targets = ['director_efc_usd']  # Only run director models when using 1301 validation
    else:
        targets = ['atl_efc_usd', 'director_efc_usd']
    models = [linear_regression, random_forest, catboost]

    # Define feature set names based on target (only active sets)
    feature_set_names = {
        'atl_efc_usd': ['set_1', 'set_2', 'set_3', 'set_5'],  # All ATL sets (excluding set_4 due to data leakage)
        'director_efc_usd': ['set_1', 'set_2', 'set_3', 'set_4', 'set_5', 'set_6', 'set_7', 'set_8', 'set_9', 
        'set_10', 'set_11', 'set_12', 'set_13', 'set_14', 'set_15', 'set_16','set_17']  # Active sets including new director profile features
    }

    # Define intelligent director feature sets (with/without fee history)
    intelligent_director_sets = {
        'director_efc_usd': ['set_14', 'set_15', 'set_16']  # Base sets - variants generated dynamically
    }
    
    # Iterate over each combination of target, feature set and model
    for target in targets:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}\n")
        
        # Select appropriate dataset based on target
        data = atl_data if target == 'atl_efc_usd' else director_data
        
        for set_name in feature_set_names[target]:
            try:
                # Get feature set description
                set_description = get_feature_set_description(target, set_name)
                print(f"\nProcessing {set_name}: {set_description}")
                
                features = select_features(data, set_name, target=target)  # Select feature set for specific target
                
                # Convert features to DataFrame for easy manipulation, preserving the original index
                features_df = pd.DataFrame(features, index=data.index)
                
                # Keep original features for CatBoost (no encoding needed)
                features_original = features_df.copy()
                
                # Use encode_rare_categories to handle rare categories and one-hot encoding for other models
                features_encoded = encode_rare_categories(features_df, one_hot_encode=True)
                
                # Get the target values (using encoded index as it might have dropped duplicates)
                y = data.loc[features_encoded.index, target]
                
                # For CatBoost, use original features aligned with the encoded index
                features_catboost = features_original.loc[features_encoded.index]
                
                # TODO: Evaluate if this sophisticated cleaning catches additional issues missed in feature_engineering.py
                # Original simple approach (commented out for comparison):
                # # Handle data quality issues before model training
                # Note: All data cleaning (NaN handling, $0 cost removal) now handled in feature_engineering.py
                # using comprehensive hierarchical imputation (categorical mode, numerical median)
                
                # Handle data quality issues before model training
                print(f"    Data cleaning for {target}...")
                print(f"      Target {target}: {len(y)} samples")
                print(f"      Target NaN values: {y.isnull().sum()}")
                print(f"      Target negative/zero values: {(y <= 0).sum()}")
                
                # Remove records with target values <= 0
                valid_target_mask = y > 0
                if valid_target_mask.sum() < len(y):
                    removed_count = len(y) - valid_target_mask.sum()
                    print(f"      Removing {removed_count} records with target <= 0")
                    features_encoded = features_encoded[valid_target_mask]
                    features_catboost = features_catboost[valid_target_mask]
                    y = y[valid_target_mask]
                
                print(f"      After target cleaning: {len(y)} samples")
                print(f"      Feature matrix shape: {features_encoded.shape}")
                
                # Check for NaN values in features and target
                feature_nan_count = features_encoded.isnull().sum().sum()
                target_nan_count = y.isnull().sum()
                
                print(f"      Feature NaN values: {feature_nan_count}")
                print(f"      Target NaN values: {target_nan_count}")
                
                # Remove records with NaN values 
                if feature_nan_count > 0 or target_nan_count > 0:
                    # Create mask for valid (non-NaN) records
                    feature_valid_mask = ~features_encoded.isnull().any(axis=1)
                    target_valid_mask = ~y.isnull()
                    valid_mask = feature_valid_mask & target_valid_mask
                    
                    removed_nan_count = len(y) - valid_mask.sum()
                    if removed_nan_count > 0:
                        print(f"      Removing {removed_nan_count} records with NaN values")
                        features_encoded = features_encoded[valid_mask]
                        features_catboost = features_catboost[valid_mask]
                        y = y[valid_mask]
                    else:
                        print(f"      No records removed for NaN values")
                else:
                    print(f"      No NaN values found - no records removed")
                    
                print(f"      Final clean dataset: {len(y)} samples, {features_encoded.shape[1]} features")

                # Use cleaned data
                y_clean = y
                features_encoded_clean = features_encoded
                features_catboost_clean = features_catboost
                
                # Skip if too few samples
                if len(y_clean) < 100:
                    print(f"      ⚠️  Skipping {target} - {set_name}: too few samples ({len(y_clean)})")
                    continue

                # Split the data into training and testing sets
                if use_1301_validation and target == 'director_efc_usd' and validation_ids_1301 is not None:
                    # Use 1301 validation set for split
                    print(f"      Using 1301 validation set for train/test split")

                    # Get season_production_ids for clean data
                    clean_ids = data.loc[y_clean.index, 'season_production_id']

                    # Create masks for validation and training
                    validation_mask = clean_ids.isin(validation_ids_1301)
                    # Train on ALL data that's NOT in the validation set (not just 1301 training titles)
                    training_mask = ~validation_mask

                    # Get indices
                    indices_test = np.where(validation_mask)[0]
                    indices_train = np.where(training_mask)[0]

                    print(f"      Training samples: {len(indices_train)} (using all available data)")
                    print(f"      Validation samples: {len(indices_test)} (matching 1301 validation set)")
                else:
                    # Standard train/test split
                    indices = np.arange(len(y_clean))
                    indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=42)
                
                # Prepare encoded data for sklearn models
                X_train_encoded = features_encoded_clean.iloc[indices_train].values
                X_test_encoded = features_encoded_clean.iloc[indices_test].values

                # Also prepare DataFrame versions for SHAP analysis (preserves feature names)
                X_train_encoded_df = features_encoded_clean.iloc[indices_train]
                X_test_encoded_df = features_encoded_clean.iloc[indices_test]

                # Prepare original data for CatBoost (use cleaned data)
                X_train_catboost = features_catboost_clean.iloc[indices_train]
                X_test_catboost = features_catboost_clean.iloc[indices_test]

                # Split target (use cleaned data)
                y_train = y_clean.iloc[indices_train]
                y_test = y_clean.iloc[indices_test]

                # Get grouping columns for test set (use cleaned indices)
                test_orgs = data.loc[y_clean.index, 'gravity_buying_organization_desc'].iloc[indices_test]
                test_teams = data.loc[y_clean.index, 'gravity_buying_team'].iloc[indices_test]
                
                for model_func in models:
                    # Apply cached hyperparameters if available
                    model = _apply_cached_hyperparameters_to_model(
                        model_func, X_train_encoded, y_train, 
                        retune_hyperparameters=retune_hyperparameters, verbose=True
                    )
                    
                    # Check if this is CatBoost and use appropriate features
                    if isinstance(model, CatBoostWrapper):
                        model.train(X_train_catboost, y_train)
                    else:
                        model.train(X_train_encoded, y_train)
                    
                    # Predict on the test set using appropriate features
                    if isinstance(model, CatBoostWrapper):
                        y_pred = model.predict(X_test_catboost)
                    else:
                        y_pred = model.predict(X_test_encoded)
                    
                    # Evaluate model accuracy
                    mse = mean_squared_error(y_test, y_pred)
                    mae = np.median(np.abs(y_test - y_pred))  # Median Absolute Error for consistency with MAPE
                    
                    # Calculate Median Absolute Percentage Error (MAPE)
                    try:
                        # Avoid division by zero by filtering out zero actual values
                        non_zero_mask = y_test != 0
                        if non_zero_mask.sum() > 0:
                            mape = np.median(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask]) * 100)
                        else:
                            mape = np.nan  # All actual values are zero
                    except:
                        mape = np.nan
                    
                    # Use adjusted R² instead of regular R²
                    if isinstance(model, CatBoostWrapper):
                        r2 = adjusted_r2_score(y_test, y_pred, X_test_catboost)
                    else:
                        r2 = adjusted_r2_score(y_test, y_pred, X_test_encoded)
                    rmse = np.sqrt(mse)

                    # Calculate error rate metrics
                    error_rate_metrics = calculate_error_rate_metrics(y_test, y_pred)
                    
                    # Create unique key for caching
                    model_key = f"{target}_{model_func.__name__}_{set_name}"
                    
                    # Determine subset_type for proper model categorization
                    subset_type = None
                    # Note: For regular model training, subset_type is always None
                    # Only intelligent director models get subset_type assignments
                    # The creates_variants flag is used later by intelligent training
                    
                    # Store overall metrics
                    model_results.append({
                        'target': target,
                        'model': model_func.__name__,
                        'feature_set': set_name,
                        'feature_set_description': set_description,
                        'mse': mse,
                        'mdae': mae,
                        'mdape': mape,
                        'rmse': rmse,
                        'adjusted_r2_score': r2,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'model_key': model_key,
                        'subset_type': subset_type,
                        **error_rate_metrics
                    })
                    
                    # Cache model and data for potential feature importance/SHAP analysis
                    model_cache[model_key] = {
                        'model': model,
                        'model_func': model_func,
                        'X_train_encoded': X_train_encoded,
                        'X_test_encoded': X_test_encoded,
                        'X_train_encoded_df': X_train_encoded_df,  # DataFrame versions for SHAP
                        'X_test_encoded_df': X_test_encoded_df,    # DataFrame versions for SHAP
                        'X_train_catboost': X_train_catboost,
                        'X_test_catboost': X_test_catboost,
                        'y_train': y_train,
                        'y_test': y_test,
                        'features_encoded': features_encoded,
                        'features_original': features_original,
                        'target': target,
                        'set_name': set_name,
                        'set_description': set_description,
                        'adjusted_r2_score': r2,  # Add R² score to model cache for feature analysis
                        'indices_test': indices_test,  # Store test indices for prediction output
                        'subset_type': subset_type  # Add subset_type for intelligent routing
                    }
                    
                    print(f"\nModel: {model_func.__name__}")
                    print(f"Target: {target}")
                    print(f"Feature Set: {set_name} - {set_description}")
                    print(f"MdAPE (%): {mape:.1f}%")
                    print(f"MAE: ${mae:,.0f}")
                    print(f"RMSE: ${rmse:,.0f}")
                    print(f"Adjusted R² Score: {r2:.3f}")
                    print(f"Mean Squared Error: {mse:.2f}")
                    print(f"MAE: {mae:.2f}")
                    print(f"RMSE: {rmse:.2f}")
                    print(f"R^2 Score: {r2:.2f}")
                    print(f"MdAPE: {error_rate_metrics['mdape']:.1f}%")
                    print(f"Titles with <20% error: {error_rate_metrics['pct_titles_under_20_error']:.1f}%")
                    print(f"Titles with <6% error: {error_rate_metrics['pct_titles_under_6_error']:.1f}%")
                    
                    # Calculate metrics by organization
                    org_metrics = calculate_metrics_by_group(y_test, y_pred, test_orgs, 'gravity_buying_organization_desc', model_key)
                    org_metrics['target'] = target
                    org_metrics['model'] = model_func.__name__
                    org_metrics['feature_set'] = set_name
                    org_metrics['feature_set_description'] = set_description
                    org_metrics['model_key'] = model_key
                    all_org_metrics.append(org_metrics)
                    
                    # Calculate metrics by team
                    team_metrics = calculate_metrics_by_group(y_test, y_pred, test_teams, 'gravity_buying_team', model_key)
                    team_metrics['target'] = target
                    team_metrics['model'] = model_func.__name__
                    team_metrics['feature_set'] = set_name
                    team_metrics['feature_set_description'] = set_description
                    team_metrics['model_key'] = model_key
                    all_team_metrics.append(team_metrics)
                
            except Exception as e:
                print(f"❌ Error processing {set_name}: {e}")
                continue

    # Train intelligent director models (separate models for with/without fee history)
    if 'director_efc_usd' in targets and not use_1301_validation:
        print("\n" + "="*60)
        print("TRAINING INTELLIGENT DIRECTOR MODELS")
        print("="*60)
        intelligent_results = train_intelligent_director_models(director_data, models, intelligent_director_sets['director_efc_usd'], retune_hyperparameters)

        # Add intelligent results to main results for reporting
        model_results.extend(intelligent_results['model_results'])
        all_org_metrics.extend(intelligent_results['org_metrics'])
        all_team_metrics.extend(intelligent_results['team_metrics'])
        model_cache.update(intelligent_results['model_cache'])

        # Demonstrate intelligent prediction routing
        print("\n" + "="*60)
        print("INTELLIGENT PREDICTION ROUTING DEMO")
        print("="*60)
        create_intelligent_prediction_demo(model_cache, director_data, n_samples=5)

    # Process results and generate reports (after all models including intelligent models)
    _process_results(model_results, all_org_metrics, all_team_metrics, model_cache, director_data)

    if use_1301_validation:
        # Generate 1301 comparison outputs
        all_predictions = _save_model_predictions_1301_format(model_cache, director_data, validation_ids_1301, training_ids_1301)
        _create_comparison_summary(all_predictions)

    # Calculate and display total execution time
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"⏱️  Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"🏁 Completed at: {datetime.now().strftime('%H:%M:%S')}")


def train_intelligent_director_models(data, models, intelligent_set_names, retune_hyperparameters=False):
    """
    Train separate models for directors with and without previous fee history.

    Returns dictionary with:
    - model_results: List of model performance metrics
    - org_metrics: Organization-level metrics
    - team_metrics: Team-level metrics
    - model_cache: Cached models for analysis
    - combined_metrics: Overall performance combining both models
    """
    target = 'director_efc_usd'
    results = {
        'model_results': [],
        'org_metrics': [],
        'team_metrics': [],
        'model_cache': {},
        'combined_metrics': {}
    }

    # Determine which titles have previous fee data vs not
    has_fee_mask = data['mean_director_previous_director_efc_usd'].notna()
    no_fee_mask = ~has_fee_mask

    print(f"Data split:")
    print(f"  Titles with fee history: {has_fee_mask.sum()} ({has_fee_mask.sum()/len(data)*100:.1f}%)")
    print(f"  Titles without fee history: {no_fee_mask.sum()} ({no_fee_mask.sum()/len(data)*100:.1f}%)")

    # Process each intelligent feature set - generate fee variants dynamically
    for base_set_name in intelligent_set_names:
        # Check if this set supports fee features
        feature_set_config = FEATURE_SETS[target].get(base_set_name, {})
        supports_fees = isinstance(feature_set_config, dict) and 'fee_features' in feature_set_config

        if not supports_fees:
            print(f"⚠️  Skipping {base_set_name}: doesn't support fee-based variants")
            continue

        # Generate both fee variants for each base set
        for include_fees, subset_mask, subset_name in [
            (False, no_fee_mask, "no_fee_history"),
            (True, has_fee_mask, "with_fee_history")
        ]:
            # Generate dynamic set name
            suffix = "_with_fees" if include_fees else "_no_fees"
            set_name = f"{base_set_name}{suffix}"

            # Get subset data
            subset_data = data[subset_mask].copy()

            if len(subset_data) < 50:
                print(f"⚠️  Skipping {set_name}: insufficient data ({len(subset_data)} samples)")
                continue

            print(f"\n{'='*50}")
            print(f"TRAINING: {set_name} ({subset_name})")
            print(f"Data size: {len(subset_data)} titles")
            print(f"{'='*50}")

            try:
                # Get feature set description
                set_description = get_feature_set_description(target, set_name)
                print(f"Processing {set_name}: {set_description}")

                features = select_features(subset_data, set_name, target=target)

                # Convert features to DataFrame for easy manipulation
                features_df = pd.DataFrame(features, index=subset_data.index)

                # Keep original features for CatBoost
                features_original = features_df.copy()

                # Use encode_rare_categories for other models
                features_encoded = encode_rare_categories(features_df, one_hot_encode=True)

                # Get the target values
                y = subset_data.loc[features_encoded.index, target]

                # For CatBoost, use original features aligned with encoded index
                features_catboost = features_original.loc[features_encoded.index]

                # Handle data quality issues before model training
                print(f"    Data cleaning for {target} ({subset_name})...")
                print(f"      Target {target}: {len(y)} samples")
                print(f"      Target NaN values: {y.isnull().sum()}")
                print(f"      Target negative/zero values: {(y <= 0).sum()}")
                
                # Remove records with target values <= 0
                valid_target_mask = y > 0
                if valid_target_mask.sum() < len(y):
                    removed_count = len(y) - valid_target_mask.sum()
                    print(f"      Removing {removed_count} records with target <= 0")
                    features_encoded = features_encoded[valid_target_mask]
                    features_catboost = features_catboost[valid_target_mask]
                    y = y[valid_target_mask]
                
                print(f"      After target cleaning: {len(y)} samples")
                print(f"      Feature matrix shape: {features_encoded.shape}")
                
                # Check for NaN values in features and target
                feature_nan_count = features_encoded.isnull().sum().sum()
                target_nan_count = y.isnull().sum()
                
                print(f"      Feature NaN values: {feature_nan_count}")
                print(f"      Target NaN values: {target_nan_count}")
                
                # Remove records with NaN values 
                if feature_nan_count > 0 or target_nan_count > 0:
                    # Create mask for valid (non-NaN) records
                    feature_valid_mask = ~features_encoded.isnull().any(axis=1)
                    target_valid_mask = ~y.isnull()
                    valid_mask = feature_valid_mask & target_valid_mask
                    
                    removed_nan_count = len(y) - valid_mask.sum()
                    if removed_nan_count > 0:
                        print(f"      Removing {removed_nan_count} records with NaN values")
                        features_encoded = features_encoded[valid_mask]
                        features_catboost = features_catboost[valid_mask]
                        y = y[valid_mask]
                    else:
                        print(f"      No records removed for NaN values")
                else:
                    print(f"      No NaN values found - no records removed")
                    
                print(f"      Final clean dataset: {len(y)} samples, {features_encoded.shape[1]} features")

                # Use cleaned data
                y_clean = y
                features_encoded_clean = features_encoded
                features_catboost_clean = features_catboost
                
                # Skip if too few samples
                if len(y_clean) < 30:
                    print(f"      ⚠️  Skipping {subset_name}: too few samples ({len(y_clean)})")
                    continue

                # Split data (same random state for consistency)
                indices = np.arange(len(y_clean))
                indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=42)

                # Prepare data splits
                X_train_encoded = features_encoded_clean.iloc[indices_train].values
                X_test_encoded = features_encoded_clean.iloc[indices_test].values
                
                # Also prepare DataFrame versions for SHAP analysis (preserves feature names)
                X_train_encoded_df = features_encoded_clean.iloc[indices_train]
                X_test_encoded_df = features_encoded_clean.iloc[indices_test]
                
                X_train_catboost = features_catboost_clean.iloc[indices_train]
                X_test_catboost = features_catboost_clean.iloc[indices_test]
                y_train = y_clean.iloc[indices_train]
                y_test = y_clean.iloc[indices_test]

                # Get grouping columns for test set
                test_orgs = subset_data.loc[y_clean.index, 'gravity_buying_organization_desc'].iloc[indices_test]
                test_teams = subset_data.loc[y_clean.index, 'gravity_buying_team'].iloc[indices_test]

                # Train each model
                for model_func in models:
                    # Apply cached hyperparameters if available
                    model = _apply_cached_hyperparameters_to_model(
                        model_func, X_train_encoded, y_train, 
                        retune_hyperparameters=retune_hyperparameters, verbose=True
                    )

                    # Train model with appropriate features
                    if isinstance(model, CatBoostWrapper):
                        model.train(X_train_catboost, y_train)
                        y_pred = model.predict(X_test_catboost)
                    else:
                        model.train(X_train_encoded, y_train)
                        y_pred = model.predict(X_test_encoded)

                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = np.median(np.abs(y_test - y_pred))  # Median Absolute Error for consistency with MAPE
                    
                    # Calculate Median Absolute Percentage Error (MAPE)
                    try:
                        # Avoid division by zero by filtering out zero actual values
                        non_zero_mask = y_test != 0
                        if non_zero_mask.sum() > 0:
                            mape = np.median(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask]) * 100)
                        else:
                            mape = np.nan  # All actual values are zero
                    except:
                        mape = np.nan
                        
                    # Use adjusted R² instead of regular R² 
                    r2 = adjusted_r2_score(y_test, y_pred, X_test_encoded)
                    rmse = np.sqrt(mse)

                    # Calculate error rate metrics
                    error_metrics = calculate_error_rate_metrics(y_test, y_pred)

                    # Create unique model key
                    model_key = f"{target}_{model_func.__name__}_{set_name}_intelligent"

                    # Store results
                    results['model_results'].append({
                        'target': target,
                        'model': model_func.__name__,
                        'feature_set': set_name,
                        'feature_set_description': f"{set_description} ({subset_name})",
                        'mse': mse,
                        'mdae': mae,
                        'mdape': mape,
                        'rmse': rmse,
                        'adjusted_r2_score': r2,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'model_key': model_key,
                        'subset_type': subset_name,
                        **error_metrics
                    })

                    # Cache model
                    results['model_cache'][model_key] = {
                        'model': model,
                        'model_func': model_func,
                        'X_train_encoded': X_train_encoded,
                        'X_test_encoded': X_test_encoded,
                        'X_train_encoded_df': X_train_encoded_df,  # DataFrame versions for SHAP
                        'X_test_encoded_df': X_test_encoded_df,    # DataFrame versions for SHAP
                        'X_train_catboost': X_train_catboost,
                        'X_test_catboost': X_test_catboost,
                        'y_train': y_train,
                        'y_test': y_test,
                        'features_encoded': features_encoded,
                        'features_original': features_original,
                        'target': target,
                        'set_name': set_name,
                        'set_description': set_description,
                        'adjusted_r2_score': r2,  # Add R² score to model cache for feature analysis
                        'indices_test': indices_test,
                        'subset_type': subset_name
                    }

                    print(f"\nModel: {model_func.__name__} ({subset_name})")
                    print(f"Feature Set: {set_name}")
                    print(f"MdAPE: {mape:.1f}%, MAE: ${mae:,.0f}, RMSE: ${rmse:,.0f}, Adjusted R²: {r2:.3f}")
                    print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
                    print(f"MdAPE: {error_metrics['mdape']:.1f}%")
                    print(f"Titles with <20% error: {error_metrics['pct_titles_under_20_error']:.1f}%")
                    print(f"Titles with <6% error: {error_metrics['pct_titles_under_6_error']:.1f}%")

                    # Calculate metrics by organization
                    org_metrics = calculate_metrics_by_group(y_test, y_pred, test_orgs, 'gravity_buying_organization_desc', model_key)
                    org_metrics['target'] = target
                    org_metrics['model'] = model_func.__name__
                    org_metrics['feature_set'] = set_name
                    org_metrics['feature_set_description'] = f"{set_description} ({subset_name})"
                    org_metrics['model_key'] = model_key
                    org_metrics['subset_type'] = subset_name
                    results['org_metrics'].append(org_metrics)

                    # Calculate metrics by team
                    team_metrics = calculate_metrics_by_group(y_test, y_pred, test_teams, 'gravity_buying_team', model_key)
                    team_metrics['target'] = target
                    team_metrics['model'] = model_func.__name__
                    team_metrics['feature_set'] = set_name
                    team_metrics['feature_set_description'] = f"{set_description} ({subset_name})"
                    team_metrics['model_key'] = model_key
                    team_metrics['subset_type'] = subset_name
                    results['team_metrics'].append(team_metrics)

            except Exception as e:
                print(f"Error with intelligent model {set_name}: {e}")
                continue

    # Calculate combined metrics across both models
    print(f"\n{'='*60}")
    print("INTELLIGENT DIRECTOR MODEL SUMMARY")
    print(f"{'='*60}")
    _summarize_intelligent_models(results['model_results'])

    # Calculate feature importance and SHAP for intelligent models
    if results['model_cache']:
        print(f"\n{'='*60}")
        print("CALCULATING FEATURE IMPORTANCE AND SHAP FOR INTELLIGENT MODELS")
        print(f"{'='*60}")
        
        print(f"DEBUG: Found {len(results['model_cache'])} cached intelligent models")
        for model_key in results['model_cache'].keys():
            print(f"  - {model_key}")
        
        # Create a flat set of model keys (not nested dictionary)
        # This matches the format expected by calculate_feature_importance_and_shap
        intelligent_model_keys = set(results['model_cache'].keys())
        
        print(f"DEBUG: Created flat model keys set with {len(intelligent_model_keys)} models")
        
        # Calculate feature importance for intelligent models
        try:
            calculate_feature_importance_and_shap(intelligent_model_keys, results['model_cache'])
            print("DEBUG: Feature importance calculation completed successfully")
        except Exception as e:
            print(f"DEBUG: Feature importance calculation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n{'='*60}")
        print("DEBUG: No intelligent models cached - skipping feature importance calculation")
        print(f"{'='*60}")

    return results


def _summarize_intelligent_models(intelligent_results):
    """Summarize performance of intelligent director models."""
    if not intelligent_results:
        print("No intelligent model results to summarize")
        return

    # Group by base model type (ignoring _no_fees/_with_fees suffix)
    base_sets = {}
    for result in intelligent_results:
        base_set = result['feature_set'].replace('_no_fees', '').replace('_with_fees', '')
        model_name = result['model']
        key = f"{base_set}_{model_name}"

        if key not in base_sets:
            base_sets[key] = {'no_fee': None, 'with_fee': None}

        if 'no_fee_history' in result['subset_type']:
            base_sets[key]['no_fee'] = result
        else:
            base_sets[key]['with_fee'] = result

    # Print summary for each model combination
    for key, models in base_sets.items():
        if models['no_fee'] and models['with_fee']:
            print(f"\nIntelligent Model: {key}")
            print(f"  Without fee history: MdAPE={models['no_fee']['mdape']:.1f}%, MdAE=${models['no_fee']['mdae']:,.0f}, RMSE=${models['no_fee']['rmse']:,.0f}, Adjusted R²={models['no_fee']['adjusted_r2_score']:.3f} (n={models['no_fee']['n_test']})")
            print(f"  With fee history:    MdAPE={models['with_fee']['mdape']:.1f}%, MdAE=${models['with_fee']['mdae']:,.0f}, RMSE=${models['with_fee']['rmse']:,.0f}, Adjusted R²={models['with_fee']['adjusted_r2_score']:.3f} (n={models['with_fee']['n_test']})")

            # Calculate combined weighted performance
            total_samples = models['no_fee']['n_test'] + models['with_fee']['n_test']
            combined_r2 = (models['no_fee']['adjusted_r2_score'] * models['no_fee']['n_test'] +
                          models['with_fee']['adjusted_r2_score'] * models['with_fee']['n_test']) / total_samples
            combined_rmse = (models['no_fee']['rmse'] * models['no_fee']['n_test'] +
                           models['with_fee']['rmse'] * models['with_fee']['n_test']) / total_samples
            combined_mape = (models['no_fee']['mdape'] * models['no_fee']['n_test'] +
                           models['with_fee']['mdape'] * models['with_fee']['n_test']) / total_samples
            combined_mae = (models['no_fee']['mdae'] * models['no_fee']['n_test'] +
                          models['with_fee']['mdae'] * models['with_fee']['n_test']) / total_samples

            print(f"  Combined (weighted):  MdAPE={combined_mape:.1f}%, MdAE=${combined_mae:,.0f}, RMSE=${combined_rmse:,.0f}, Adjusted R²={combined_r2:.3f} (n={total_samples})")


def _process_results(model_results, all_org_metrics, all_team_metrics, model_cache, data):
    """Process model results and generate comprehensive reports."""

    # Identify best performing models
    best_models = identify_best_models(model_results, all_org_metrics, all_team_metrics)

    # Calculate feature importance and SHAP for best models only
    if best_models and model_cache:
        print("\n" + "="*60)
        print("CALCULATING FEATURE IMPORTANCE AND SHAP FOR BEST MODELS")
        print("="*60)
        calculate_feature_importance_and_shap(best_models, model_cache)

        print("\n" + "="*60)
        print("SAVING PREDICTIONS FOR BEST MODELS")
        print("="*60)
        save_best_model_predictions(best_models, model_cache, data)

    # Save metrics and create visualizations
    if model_results:
        _save_metrics_and_create_visualizations(model_results, all_org_metrics, all_team_metrics, best_models)

    # Generate intelligent routing report if applicable
    print("\n" + "="*60)
    print("GENERATING INTELLIGENT ROUTING REPORT")
    print("="*60)
    generate_intelligent_routing_report(model_results)


def _save_metrics_and_create_visualizations(model_results, all_org_metrics, all_team_metrics, best_models):
    """Save model metrics and create summary visualizations."""
    
    print("\n" + "="*60)
    print("SAVING MODEL ACCURACY METRICS")
    print("="*60)
    
    # Create and save overall metrics DataFrame
    metrics_df = pd.DataFrame(model_results)
    metrics_output_path = os.path.join('data', 'model_outputs','model_metrics', 'model_accuracy_metrics.csv')
    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"\nSaved overall metrics to {metrics_output_path}")
    
    # Process organization metrics
    best_org_metrics_df = _process_organization_metrics(all_org_metrics, best_models)
    
    # Process team metrics  
    best_team_metrics_df = _process_team_metrics(all_team_metrics, best_models)
    
    # Filter metrics_df to only include best models for summary charts
    best_metrics_df = metrics_df[metrics_df['model_key'].isin(best_models)]
    
    # Create summary visualizations (using filtered best model data)
    # Note: Unified charts removed - using separate ATL/Director charts instead
    
    # Generate director segments performance chart
    from utils.model_performance_analyzer import ModelPerformanceAnalyzer
    analyzer = ModelPerformanceAnalyzer()
    try:
        analyzer.generate_director_segments_chart()
        print("✅ Director segments performance chart generated")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate director segments chart: {e}")
    
    # Generate individual director performance chart with winning models
    try:
        analyzer.generate_director_winners_performance_chart()
        print("✅ Director performance summary chart generated")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate director performance summary chart: {e}")
    
    # Generate detailed team performance charts for winning models
    try:
        team_charts = analyzer.generate_detailed_team_charts()
        if team_charts:
            print(f"✅ Generated {len(team_charts)} detailed team performance charts:")
            for chart in team_charts:
                print(f"   - {os.path.basename(chart)}")
        else:
            print("⚠️  No detailed team charts generated")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate detailed team charts: {e}")
    
    # Generate detailed organization performance charts for winning models
    try:
        org_charts = analyzer.generate_organization_charts()
        if org_charts:
            print(f"✅ Generated {len(org_charts)} detailed organization performance charts:")
            for chart in org_charts:
                print(f"   - {os.path.basename(chart)}")
        else:
            print("⚠️  No detailed organization charts generated")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate detailed organization charts: {e}")
    
    # Generate team and organization performance charts
    create_summary_charts(best_metrics_df, best_org_metrics_df, best_team_metrics_df)
    
    # Generate separate feature importance charts for ATL and Director
    from utils.create_model_feature_importance_summary import create_separate_feature_importance_charts
    try:
        create_separate_feature_importance_charts()
        print("✅ Separate feature importance charts generated")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate separate feature importance charts: {e}")


def _process_organization_metrics(all_org_metrics, best_models):
    """Process and save organization-level metrics for ALL models, but return filtered data for visualizations."""
    if all_org_metrics:
        org_metrics_df = pd.concat(all_org_metrics, ignore_index=True)
        
        # Save ALL organization metrics (unfiltered)
        org_metrics_path = os.path.join('data', 'model_outputs', 'model_metrics', 'organization_metrics_summary_all.csv')
        org_metrics_df.to_csv(org_metrics_path, index=False)
        print(f"Saved ALL organization metrics to {org_metrics_path} ({len(org_metrics_df)} records across all models)")
        
        # Filter to best models for visualizations and summary statistics
        if best_models:
            best_org_metrics_df = org_metrics_df[org_metrics_df['model_key'].isin(best_models)]
            
            if not best_org_metrics_df.empty:
                # Save filtered version for visualization compatibility
                org_metrics_viz_path = os.path.join('data', 'model_outputs', 'model_metrics', 'organization_metrics_summary.csv')
                best_org_metrics_df.to_csv(org_metrics_viz_path, index=False)
                print(f"Saved organization metrics for visualizations to {org_metrics_viz_path} (filtered to {len(best_models)} best models)")
                
                # Create organization summary statistics for best models only
                org_summary = best_org_metrics_df.groupby('gravity_buying_organization_desc').agg({
                    'adjusted_r2_score': ['mean', 'std', 'min', 'max'],
                    'mse': ['mean', 'std', 'min', 'max'],
                    'rmse': ['mean', 'std', 'min', 'max'],
                    'mdae': ['mean', 'std', 'min', 'max'],
                    'count': 'mean'
                }).round(3)
                org_summary_path = os.path.join('data', 'model_outputs', 'model_metrics', 'organization_performance_summary.csv')
                org_summary.to_csv(org_summary_path)
                print(f"Saved organization summary to {org_summary_path} (best models only)")
                return best_org_metrics_df
        

            # Create organization summary statistics
            agg_dict = {
                'adjusted_r2_score': ['mean', 'std', 'min', 'max'],
                'mse': ['mean', 'std', 'min', 'max'],
                'rmse': ['mean', 'std', 'min', 'max'],
                'mdae': ['mean', 'std', 'min', 'max'],
                'count': 'mean'
            }

            # Add error rate metrics if they exist
            error_rate_columns = ['pct_titles_under_20_error', 'pct_titles_under_6_error', 'mdape', 'mean_percentage_error']
            for col in error_rate_columns:
                if col in best_org_metrics_df.columns:
                    agg_dict[col] = ['mean', 'std', 'min', 'max']

            org_summary = best_org_metrics_df.groupby('gravity_buying_organization_desc').agg(agg_dict).round(3)
            org_summary_path = os.path.join('data', 'model_outputs', 'model_metrics', 'organization_performance_summary.csv')
            org_summary.to_csv(org_summary_path)
            print(f"Saved organization summary to {org_summary_path}")
            return best_org_metrics_df
        else:
            print("No organization metrics data for best models")
    
    return None


def _process_team_metrics(all_team_metrics, best_models):
    """Process and save team-level metrics for ALL models, but return filtered data for visualizations."""
    if all_team_metrics:
        team_metrics_df = pd.concat(all_team_metrics, ignore_index=True)
        
        # Save ALL team metrics (unfiltered)
        team_metrics_path = os.path.join('data', 'model_outputs', 'model_metrics', 'team_metrics_summary_all.csv')
        team_metrics_df.to_csv(team_metrics_path, index=False)
        print(f"Saved ALL team metrics to {team_metrics_path} ({len(team_metrics_df)} records across all models)")
        
        # For visualizations, use the best available model for each team PER TARGET (not globally)
        # This ensures ALL teams appear in charts for BOTH targets using their optimal model
        team_best_metrics = []
        
        # Process each target separately to avoid target mixing bug
        for target in ['atl_efc_usd', 'director_efc_usd']:
            target_data = team_metrics_df[team_metrics_df['target'] == target]
            print(f"Processing {target}: {len(target_data)} records from {target_data['gravity_buying_team'].nunique()} teams")
            
            for team in target_data['gravity_buying_team'].unique():
                team_data = target_data[target_data['gravity_buying_team'] == team]
                
                # Always use the best available model for this specific team and target
                # This ensures ALL teams appear in visualizations for both targets
                if not team_data.empty:
                    best_for_team = team_data.loc[team_data['mdape'].idxmin()]
                    team_best_metrics.append(best_for_team)
                    
                    # Debug: show what model was selected for first few teams per target
                    if len([m for m in team_best_metrics if m['target'] == target]) <= 3:  # First 3 per target
                        model_info = best_for_team['model_key']
                        mdape = best_for_team['mdape']
                        print(f"    {target} - Team '{team}': Using {model_info} (MdAPE: {mdape:.1f}%)")
        
        if team_best_metrics:
            best_team_metrics_df = pd.DataFrame(team_best_metrics)
            
            # Save version for visualization compatibility (includes ALL teams)
            team_metrics_viz_path = os.path.join('data', 'model_outputs', 'model_metrics', 'team_metrics_summary.csv')
            best_team_metrics_df.to_csv(team_metrics_viz_path, index=False)
            print(f"Saved team metrics for visualizations to {team_metrics_viz_path} (ALL {len(best_team_metrics_df)} teams included)")
            
            # Create team summary statistics
            agg_dict = {
                'adjusted_r2_score': ['mean', 'std', 'min', 'max'],
                'mse': ['mean', 'std', 'min', 'max'],
                'rmse': ['mean', 'std', 'min', 'max'],
                'mdae': ['mean', 'std', 'min', 'max'],
                'count': 'mean'
            }

            # Add error rate metrics if they exist
            error_rate_columns = ['pct_titles_under_20_error', 'pct_titles_under_6_error', 'mdape', 'mean_percentage_error']
            for col in error_rate_columns:
                if col in best_team_metrics_df.columns:
                    agg_dict[col] = ['mean', 'std', 'min', 'max']

            team_summary = best_team_metrics_df.groupby('gravity_buying_team').agg(agg_dict).round(3)
            team_summary_path = os.path.join('data', 'model_outputs', 'model_metrics', 'team_performance_summary.csv')
            team_summary.to_csv(team_summary_path)
            print(f"Saved team summary to {team_summary_path} (ALL {len(team_summary)} teams)")
            return best_team_metrics_df
        
        print("No team metrics available for processing")
        return None
    
    return None


def compare_all_models(model_results):
    """
    Compare all models using error rate metrics and traditional metrics.
    This function can be called after main() to get a comprehensive comparison.
    """
    from model_test_metrics import print_model_comparison

    # Convert to DataFrame if needed
    if isinstance(model_results, list):
        comparison_df = pd.DataFrame(model_results)
    else:
        comparison_df = model_results

    # Filter to only include models with error rate metrics
    if 'pct_titles_under_20_error' in comparison_df.columns:
        # Sort by R² score and error rate metrics
        comparison_df = comparison_df.sort_values(
            ['adjusted_r2_score', 'pct_titles_under_20_error'],
            ascending=[False, False]
        )

        print_model_comparison(comparison_df, top_n=15)
        return comparison_df
    else:
        print("Error rate metrics not available in model results")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate machine learning models for cost prediction")
    parser.add_argument(
        '--retune-hyperparameters',
        action='store_true',
        help='Force retuning of hyperparameters (ignore cached values)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached hyperparameters before starting'
    )
    parser.add_argument(
        '--list-cache',
        action='store_true',
        help='List all cached hyperparameters and exit'
    )
    parser.add_argument(
        '--use-1301-validation',
        action='store_true',
        help='Use 1301 validation set for director models comparison (only runs director models)'
    )

    args = parser.parse_args()

    # Handle cache-related commands
    if args.list_cache:
        print("📋 Cached Hyperparameters:")
        cached_files = list_cached_hyperparameters()
        if cached_files:
            for cache_info in cached_files:
                print(f"  {cache_info['cache_key']}")
                print(f"    Cached: {cache_info['cached_at']}")
                print(f"    MAPE Score: {-cache_info['best_score']:.2f}%" if cache_info['best_score'] else "Unknown")
                print()
        else:
            print("  No cached hyperparameters found")
        exit(0)

    if args.clear_cache:
        print("🗑️  Clearing hyperparameter cache...")
        cleared_count = clear_hyperparameter_cache()
        print(f"Cleared {cleared_count} cache files")
        print("✅ Cache cleared successfully!")
        exit(0)

    # Run main training pipeline
    main(retune_hyperparameters=args.retune_hyperparameters, use_1301_validation=args.use_1301_validation)
