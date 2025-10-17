"""
Feature importance and SHAP analysis utilities.
"""

import pandas as pd
import numpy as np
import os
import shap
from shap.utils._exceptions import ExplainerError
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def _get_top_2_models(best_models, model_cache):
    """
    Identify the top 2 models: best ATL and best Director.
    Returns list of model_keys for top performers only.
    """
    # Group models by target type
    atl_models = {}
    director_models = {}
    
    for model_key in best_models:
        if model_key not in model_cache:
            continue
            
        cache_entry = model_cache[model_key]
        target = cache_entry.get('target', '')
        r2_score = cache_entry.get('r2_score', 0)
        feature_set = cache_entry.get('set_name', '')
        
        # Skip ATL set_4 due to data leakage
        if target == 'atl_efc_usd' and feature_set == 'set_4':
            continue
            
        if target == 'atl_efc_usd':
            atl_models[model_key] = r2_score
        elif target == 'director_efc_usd':
            director_models[model_key] = r2_score
    
    # Find best of each type
    top_models = []
    
    if atl_models:
        best_atl_key = max(atl_models, key=atl_models.get)
        top_models.append(best_atl_key)
        print(f"Top ATL model: {best_atl_key} (Adjusted R² = {atl_models[best_atl_key]:.3f})")
    
    if director_models:
        best_director_key = max(director_models, key=director_models.get)
        top_models.append(best_director_key)
        print(f"Top Director model: {best_director_key} (Adjusted R² = {director_models[best_director_key]:.3f})")
    
    return top_models


def calculate_feature_importance_and_shap(best_models, model_cache):
    """
    Calculate feature importance and SHAP values only for TOP 2 models (best ATL + best Director).
    
    For each top model, generates:
    - Feature importance CSV
    - SHAP values CSV (for summary visualizations only)
    """
    
    # Identify top 2 models only (best ATL + best Director)
    top_models = _get_top_2_models(best_models, model_cache)
    
    if not top_models:
        print("No top models identified for SHAP analysis")
        return
    
    print(f"Generating SHAP analysis for {len(top_models)} top performers only:")
    for model_key in top_models:
        cache_entry = model_cache.get(model_key, {})
        target = cache_entry.get('target', 'unknown')
        print(f"  - {model_key} (Target: {target})")
    
    for model_key in top_models:
        if model_key not in model_cache:
            continue
            
        print(f"\n{'='*40}")
        print(f"Processing: {model_key}")
        print(f"{'='*40}")
        
        cache_entry = model_cache[model_key]
        model = cache_entry['model']
        model_func = cache_entry['model_func']
        target = cache_entry['target']
        set_name = cache_entry['set_name']
        
        # Feature Importance
        importances = None
        underlying_model = model.get_model()
        
        # Check if it's a TransformedTargetRegressor and get the actual regressor
        if hasattr(underlying_model, 'regressor_'):
            actual_model = underlying_model.regressor_
        else:
            actual_model = underlying_model
        
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, 'get_feature_importance'):
            importances = actual_model.get_feature_importance()
        
        if importances is not None:
            # Use appropriate feature names based on model type
            from models.model_wrappers import CatBoostWrapper
            if isinstance(model, CatBoostWrapper):
                feature_names = cache_entry['features_original'].columns
            else:
                feature_names = cache_entry['features_encoded'].columns
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            
            print("Top 10 Feature Importances:")
            print(feature_importance_df.head(10))
            
            # Save feature importance
            output_path = os.path.join('data', 'model_outputs', 'feature_importance_sets', 'feature_importances_{model_key}.csv')
            feature_importance_df.to_csv(output_path, index=False)
            print(f"Saved feature importances to {output_path}")
        
        # SHAP Values
        try:
            shap_model = actual_model if hasattr(underlying_model, 'regressor_') else underlying_model
            
            # Use appropriate test data for SHAP
            from models.model_wrappers import CatBoostWrapper
            if isinstance(model, CatBoostWrapper):
                X_test_for_shap = cache_entry['X_test_catboost']
                X_train_for_shap = cache_entry['X_train_catboost']
                shap_columns = cache_entry['X_test_catboost'].columns
            else:
                # Use DataFrame versions to preserve feature names for SHAP
                X_test_for_shap = cache_entry['X_test_encoded_df']
                X_train_for_shap = cache_entry['X_train_encoded_df']
                shap_columns = cache_entry['X_test_encoded_df'].columns
            
            if model_func.__name__ in ('catboost', 'random_forest'):
                explainer = shap.TreeExplainer(shap_model)
                shap_values = explainer.shap_values(X_test_for_shap)
            elif model_func.__name__ in ('linear_regression'):
                explainer = shap.LinearExplainer(shap_model, X_train_for_shap)
                shap_values = explainer.shap_values(X_test_for_shap)
            else:
                print(f"SHAP not implemented for {model_func.__name__}")
                continue
            
            # Convert SHAP values to DataFrame
            try:
                shap_values_df = pd.DataFrame(shap_values, columns=shap_columns)
                print("SHAP values calculated successfully")
                
                # Save SHAP values
                output_path = os.path.join('data', 'model_outputs', 'feature_importance_sets', f'SHAP_values_{model_key}.csv')
                shap_values_df.to_csv(output_path, index=False)
                print(f"Saved SHAP values to {output_path}")
                
                # Display mean absolute SHAP values (feature importance via SHAP)
                mean_shap = pd.DataFrame({
                    'feature': shap_columns,
                    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
                }).sort_values(by='mean_abs_shap', ascending=False)
                print("\nTop 10 Features by Mean Absolute SHAP:")
                print(mean_shap.head(10))
                
                # Generate SHAP summary plots
                _generate_shap_plots(shap_values, X_test_for_shap, mean_shap, model_key, shap_columns)
                
            except ValueError as ve:
                print(f"Warning: Could not create SHAP DataFrame - {ve}")
                
        except Exception as e:
            print(f"Error calculating SHAP for {model_key}: {e}")


def _generate_shap_plots(shap_values, X_test_for_shap, mean_shap, model_key, feature_names):
    """Generate SHAP visualization plots - DISABLED: Only for top performers in summary charts."""
    print(f"SHAP plot generation disabled for individual feature sets. Model: {model_key}")
    print("SHAP plots only generated for top performing models in summary visualizations.")
    return
    
    # DISABLED - Uncomment below if individual SHAP plots needed for debugging
    try:
        # Create summary plot showing feature importance (bar plot)
        try:
            plt.figure(figsize=(10, 8))
            # Create a DataFrame with proper feature names for SHAP plotting
            X_test_named = pd.DataFrame(X_test_for_shap, columns=feature_names)
            shap.summary_plot(shap_values, X_test_named, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {model_key}')
            plt.tight_layout()
            # Create SHAP visuals directory if it doesn't exist
            shap_visuals_dir = 'data/model_outputs/visuals/SHAP_visuals'
            os.makedirs(shap_visuals_dir, exist_ok=True)
            bar_plot_path = os.path.join(shap_visuals_dir, f'SHAP_importance_{model_key}.png')
            plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP importance plot to {bar_plot_path}")
        except Exception as e:
            print(f"Warning: Could not create bar plot - {e}")
            plt.close()
        
        # Create summary plot showing feature impact (beeswarm plot)
        try:
            # Use explicit figure and axis to avoid colorbar issues
            fig, ax = plt.subplots(figsize=(10, 8))
            # Create a DataFrame with proper feature names for SHAP plotting
            X_test_named = pd.DataFrame(X_test_for_shap, columns=feature_names)
            shap.summary_plot(shap_values, X_test_named, show=False, plot_size=None)
            plt.suptitle(f'SHAP Summary Plot - {model_key}', y=1.02)
            summary_plot_path = os.path.join('data', 'model_outputs', f'SHAP_summary_{model_key}.png')
            plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP summary plot to {summary_plot_path}")
        except Exception as e:
            print(f"Warning: Could not create summary plot - {e}")
            plt.close()
        
        # For top features, create dependence plots
        try:
            # Create a DataFrame with proper feature names for dependence plots
            X_test_named = pd.DataFrame(X_test_for_shap, columns=feature_names)
            top_features = mean_shap.head(3)['feature'].values
            plots_saved = 0
            for i, feature in enumerate(top_features):
                try:
                    if feature in feature_names:
                        feature_idx = list(feature_names).index(feature)
                    else:
                        continue

                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(feature_idx, shap_values, X_test_named, show=False)
                    plt.title(f'SHAP Dependence - {feature} ({model_key})')
                    plt.tight_layout()
                    # Clean feature name for filename
                    clean_feature = feature.replace('/', '_').replace('\\', '_')[:20]
                    dep_plot_path = os.path.join('data', 'model_outputs', f'SHAP_dependence_{model_key}_{i+1}_{clean_feature}.png')
                    plt.savefig(dep_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    plots_saved += 1
                except Exception as dep_error:
                    print(f"Warning: Could not create dependence plot for {feature} - {dep_error}")
                    plt.close()
            
            if plots_saved > 0:
                print(f"Saved {plots_saved} SHAP dependence plots")
        except Exception as e:
            print(f"Warning: Could not create dependence plots - {e}")
        
    except Exception as plot_error:
        print(f"Warning: Could not create SHAP plots - {plot_error}")
