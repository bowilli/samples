"""
Model evaluation and selection utilities.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
try:
    from utils.adjusted_r2 import adjusted_r2_score_legacy, calculate_adjusted_r2_from_r2, estimate_feature_count_from_context
except ImportError:
    from adjusted_r2 import adjusted_r2_score_legacy, calculate_adjusted_r2_from_r2, estimate_feature_count_from_context
import os


def calculate_adjusted_r2_for_model_result(model_result):
    """Return the adjusted R² for a model result (already calculated in main.py)."""
    return model_result['adjusted_r2_score']

def get_top_models_by_mape(model_metrics_file=None, top_n=2, exclude_sets=None):
    """
    Consolidated function to identify top performing models by MAPE (primary) and MAE (tie-breaker).
    
    Replaces duplicate logic from dynamic_mape_analysis.py and feature_analysis.py
    Uses the same selection criteria as get_best_models_by_target() for consistency.
    
    Args:
        model_metrics_file (str): Path to model metrics CSV file
        top_n (int): Number of top models per target to return
        exclude_sets (list): Feature sets to exclude (e.g., ['set_4'] for ATL data leakage)
        
    Returns:
        list: List of top model configurations with model info
    """
    if model_metrics_file is None:
        model_metrics_file = 'data/model_outputs/model_metrics/model_accuracy_metrics.csv'
    
    if exclude_sets is None:
        exclude_sets = ['set_4']  # Default exclude ATL set_4 due to data leakage
    
    print(f"📊 Identifying top {top_n} models from {model_metrics_file}")
    print("using top 2 to ensure we capture downstream metrics for more than one model")
    
    if not os.path.exists(model_metrics_file):
        print(f"Warning: {model_metrics_file} not found")
        return []
    
    # Read model metrics
    df = pd.read_csv(model_metrics_file)
    
    # Calculate adjusted R² for all models
    df['adjusted_r2_score'] = df.apply(lambda row: calculate_adjusted_r2_for_model_result(row.to_dict()), axis=1)
    
    # Exclude problematic feature sets
    for exclude_set in exclude_sets:
        initial_count = len(df)
        df = df[~((df['target'] == 'atl_efc_usd') & (df['feature_set'] == exclude_set))]
        excluded_count = initial_count - len(df)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} models using {exclude_set} (data leakage)")
    
    top_models = []
    
    # Get top models for each target type using MAPE-based selection
    for target in ['atl_efc_usd', 'director_efc_usd']:
        target_df = df[df['target'] == target]
        target_name = 'ATL' if target == 'atl_efc_usd' else 'Director'
        
        # Use MdAPE as primary criterion (lower is better)
        valid_mape_df = target_df.dropna(subset=['mdape'])
        
        if not valid_mape_df.empty:
            # Sort by MdAPE (ascending - lower is better), then by MAE (tie-breaker)
            sorted_df = valid_mape_df.sort_values(['mdape', 'mdae'], ascending=[True, True])
            top_target_models = sorted_df.head(top_n)
            
            print(f"✅ {target_name} models ranked by MdAPE (lower is better):")
            for _, model in top_target_models.iterrows():
                print(f"   {model['model_key']}: MdAPE={model['mdape']:.1f}%, MdAE=${model['mdae']:,.0f}")
        else:
            # Fallback to adjusted R² if no MdAPE available
            sorted_df = target_df.sort_values('adjusted_r2_score', ascending=False)
            top_target_models = sorted_df.head(top_n)
            
            print(f"✅ {target_name} models ranked by Adjusted R² (MdAPE not available):")
            for _, model in top_target_models.iterrows():
                print(f"   {model['model_key']}: R²={model['adjusted_r2_score']:.3f}")
        
        # Convert to our standard format
        for _, model in top_target_models.iterrows():
            model_info = {
                'model_key': model['model_key'],
                'target': model['target'],
                'model_type': model['model'],
                'feature_set': model['feature_set'],
                'feature_description': model.get('feature_set_description', ''),
                'adjusted_r2': model['adjusted_r2_score'],
                'adjusted_r2_score': model['adjusted_r2_score'],
                'mdape': model['mdape'],
                'mdae': model['mdae'],
                'rmse': model['rmse'],
                'n_train': model['n_train'],
                'n_test': model['n_test'],
                'name': f"{target_name} {model['model'].title()} {model['feature_set'].title()}",
                'display_name': f"{target_name} {model['model'].title()} {model['feature_set'].title()} ({model.get('feature_set_description', '')})",
                'predictions_file': f"predictions_{model['model_key']}.csv"
            }
            top_models.append(model_info)
    
    print(f"🎯 Selected {len(top_models)} models for analysis (MdAPE-based selection)\n")
    return top_models


def get_best_models_by_target(model_results_or_file=None, verbose=True):
    """
    Get the single best performing model per target using MdAPE as primary criterion, MdAE as tie-breaker.
    
    This function now uses the new subset-aware logic internally but returns the simplified format
    for backward compatibility with existing code.
    
    Selection Logic:
    1. Primary: Minimum Median Absolute Percentage Error (MdAPE) - lower is better
    2. Tie-breaker: If multiple models have identical MdAPE, use Minimum Median Absolute Error (MdAE) 
    3. Fallback: If no MdAPE available, use maximum Adjusted R²
    
    Args:
        model_results_or_file: Either a list of model results or path to CSV file. 
                              If None, uses default model_accuracy_metrics.csv
        verbose: Whether to print selection details
    
    Returns:
        dict: {'atl_efc_usd': model_record, 'director_efc_usd': model_record}
    """
    # Use the new subset-aware logic internally
    best_models_subset, _ = get_best_models_by_subset(model_results_or_file, verbose=verbose)
    
    # Convert to the simplified format for backward compatibility
    best_models = {}
    
    for target, models in best_models_subset.items():
        if target == 'atl_efc_usd':
            # ATL has single overall model
            best_models[target] = models['overall']
        elif target == 'director_efc_usd':
            # Director uses overall best model (not intelligent routing)
            best_models[target] = models['overall']
    
    return best_models

def get_best_models_by_subset(model_results_or_file=None, verbose=True):
    """
    Get the best performing models per target with subset-aware selection for intelligent director models.
    
    For ATL: Returns single best model (same as get_best_models_by_target)
    For Director: Returns best overall model + best fee subset model + best no-fee subset model
    
    Selection Logic:
    1. Primary: Minimum Median Absolute Percentage Error (MdAPE) - lower is better
    2. Tie-breaker: If multiple models have identical MdAPE, use Minimum Median Absolute Error (MdAE) 
    3. Fallback: If no MdAPE available, use maximum Adjusted R²
    
    Args:
        model_results_or_file: Either a list of model results or path to CSV file. 
                              If None, uses default model_accuracy_metrics.csv
        verbose: Whether to print selection details
    
    Returns:
        tuple: (best_models_dict, results_df) where:
            best_models_dict: {
                'atl_efc_usd': {'overall': model_record},
                'director_efc_usd': {
                    'overall': model_record,
                    'fee_subset': model_record,
                    'no_fee_subset': model_record
                }
            }
            results_df: pandas DataFrame with loaded model results
    """
    # Load data
    if model_results_or_file is None:
        model_results_or_file = 'data/model_outputs/model_metrics/model_accuracy_metrics.csv'
    
    if isinstance(model_results_or_file, str):
        # Load from CSV file
        if not os.path.exists(model_results_or_file):
            if verbose:
                print(f"Error: {model_results_or_file} not found")
            return {}, pd.DataFrame()
        results_df = pd.read_csv(model_results_or_file)
    else:
        # Use provided list
        results_df = pd.DataFrame(model_results_or_file)
    
    if results_df.empty:
        return {}, results_df
    
    # Calculate adjusted R² for all models
    results_df['adjusted_r2_score'] = results_df.apply(lambda row: calculate_adjusted_r2_for_model_result(row), axis=1)
    
    best_models = {}
    
    def select_best_model(df, description=""):
        """Helper function to select best model from a dataframe using MdAPE/MdAE logic"""
        if df.empty:
            return None
            
        # Filter out models with NaN MdAPE, then find minimum MdAPE (best prediction accuracy)
        valid_mape_df = df.dropna(subset=['mdape'])
        if not valid_mape_df.empty:
            # Find minimum MdAPE (primary criterion)
            min_mape = valid_mape_df['mdape'].min()
            
            # Find all models with the minimum MdAPE (could be ties)
            best_mape_models = valid_mape_df[valid_mape_df['mdape'] == min_mape]
            
            if len(best_mape_models) > 1:
                # Tie-breaker: Use MdAE (lower is better) among models with same MdAPE
                tied_models_with_mae = best_mape_models.dropna(subset=['mdae'])
                if not tied_models_with_mae.empty:
                    best_idx = tied_models_with_mae['mdae'].idxmin()  # Minimum MdAE breaks the tie
                    tie_breaker_used = True
                else:
                    # If no MdAE available, just take the first one
                    best_idx = best_mape_models.index[0]
                    tie_breaker_used = False
            else:
                # No tie, single best MdAPE model
                best_idx = best_mape_models.index[0]
                tie_breaker_used = False
            
            best_model = valid_mape_df.loc[best_idx]
            
            if verbose and description:
                print(f"\nBest {description}: {best_model['model_key']}")
                print(f"  MdAPE: {best_model['mdape']:.1f}% (primary criterion)")
                if tie_breaker_used:
                    print(f"  MdAE: ${best_model['mdae']:,.0f} (tie-breaker - multiple models had same MdAPE)")
                else:
                    print(f"  MdAE: ${best_model['mdae']:,.0f}")
                print(f"  Adjusted R²: {best_model['adjusted_r2_score']:.3f}, RMSE: ${best_model['rmse']:,.0f}")
                print(f"  Feature Set: {best_model.get('feature_set', 'unknown')} - {best_model.get('feature_set_description', '')}")
            
            return best_model.to_dict()
        else:
            # Fallback to adjusted R² if no MdAPE available
            best_idx = df['adjusted_r2_score'].idxmax()
            best_model = df.loc[best_idx]
            
            if verbose and description:
                print(f"\nBest {description}: {best_model['model_key']}")
                print(f"  Adjusted R²: {best_model['adjusted_r2_score']:.3f} (MdAPE not available, using R² fallback)")
                print(f"  MdAE: ${best_model['mdae']:,.0f}, RMSE: ${best_model['rmse']:,.0f}")
                print(f"  Feature Set: {best_model.get('feature_set', 'unknown')} - {best_model.get('feature_set_description', '')}")
            
            return best_model.to_dict()
    
    # Process each target
    for target in results_df['target'].unique():
        target_df = results_df[results_df['target'] == target]
        target_name = 'ATL' if target == 'atl_efc_usd' else 'Director'
        
        if target == 'atl_efc_usd':
            # ATL: Single best model approach (same as before)
            best_overall = select_best_model(target_df, f"{target_name} overall model")
            if best_overall:
                best_models[target] = {'overall': best_overall}
        
        elif target == 'director_efc_usd':
            # Director: Subset-aware selection
            
            # 1. Best overall director model (only from general feature sets)
            regular_models = target_df[~target_df['model_key'].str.contains('_intelligent', na=False)]
            
            # Filter out models from fee-based feature sets (those with creates_variants=True)
            from features.feature_selection import FEATURE_SETS
            
            def is_general_feature_set(row):
                """Check if the feature set is general (not fee-based)"""
                target = row['target']
                feature_set = row['feature_set']
                
                if target in FEATURE_SETS:
                    feature_config = FEATURE_SETS[target].get(feature_set, {})
                    creates_variants = feature_config.get('creates_variants', False)
                    
                    # Only include in "overall" if it doesn't create variants
                    return not creates_variants
                
                return True  # Default to include if config not found
            
            # Apply filter to exclude fee-based feature sets from overall selection
            general_models = regular_models[regular_models.apply(is_general_feature_set, axis=1)]
            best_overall = select_best_model(general_models, f"{target_name} overall model")
            
            # 2. Best fee subset model (only intelligent models with fee history)
            if 'subset_type' in target_df.columns:
                fee_models = target_df[
                    (target_df['model_key'].str.contains('_intelligent', na=False)) & 
                    (target_df['subset_type'].str.contains('with_fee_history', na=False))
                ]
            else:
                fee_models = pd.DataFrame()  # Empty if no subset_type column
            best_fee = select_best_model(fee_models, f"{target_name} fee subset model")
            
            # 3. Best no-fee subset model (only intelligent models without fee history)
            if 'subset_type' in target_df.columns:
                no_fee_models = target_df[
                    (target_df['model_key'].str.contains('_intelligent', na=False)) & 
                    (target_df['subset_type'].str.contains('no_fee_history', na=False))
                ]
            else:
                no_fee_models = pd.DataFrame()  # Empty if no subset_type column
            best_no_fee = select_best_model(no_fee_models, f"{target_name} no-fee subset model")
            
            # Store results
            director_results = {}
            if best_overall:
                director_results['overall'] = best_overall
            if best_fee:
                director_results['fee_subset'] = best_fee
            if best_no_fee:
                director_results['no_fee_subset'] = best_no_fee
            
            if director_results:
                best_models[target] = director_results
    
    if verbose:
        total_models = sum(len(models) if isinstance(models, dict) else 1 for models in best_models.values())
        print(f"\nTotal models selected: {total_models}")
        print("Note: Selection uses MdAPE as primary criterion, MdAE as tie-breaker")
        print("Director models include subset-specific selections for intelligent routing analysis")
    
    return best_models, results_df

def _compare_models(model1, model2, model1_name, model2_name):
    """
    Shared helper function to compare two models using MdAPE (primary) and MdAE (tie-breaker).
    
    Args:
        model1, model2: Model dictionaries with 'mdape' and 'mdae' keys
        model1_name, model2_name: Names for the models being compared
    
    Returns:
        tuple: (winner_name, performance_difference)
    """
    # Primary comparison: MdAPE (lower is better)
    if model1['mdape'] < model2['mdape']:
        return model1_name, model1['mdape'] - model2['mdape']
    elif model2['mdape'] < model1['mdape']:
        return model2_name, model2['mdape'] - model1['mdape']
    else:
        # Tie-breaker: MdAE (lower is better)
        if model1['mdae'] < model2['mdae']:
            return model1_name, 0  # Tie broken by MdAE
        elif model2['mdae'] < model1['mdae']:
            return model2_name, 0  # Tie broken by MdAE
        else:
            return model2_name, 0  # Perfect tie, prefer general (model2)

def _compare_models_robust(model1, model2, model1_name, model2_name):
    """
    Robust model comparison using multi-metric scoring for edge cases.
    
    Uses MdAPE as primary metric, but if the difference is small (<2%), 
    uses a weighted multi-metric score to make more nuanced decisions.
    
    Scoring weights:
    - MdAPE: 40% (lower is better)
    - MdAE: 20% (lower is better) 
    - RMSE: 20% (lower is better)
    - Adjusted R²: 20% (higher is better)
    
    Args:
        model1, model2: Model dictionaries with metric keys
        model1_name, model2_name: Names for the models being compared
    
    Returns:
        tuple: (winner_name, performance_difference, reason)
    """
    # Primary comparison: MdAPE (lower is better)
    mdape_diff = model1['mdape'] - model2['mdape']
    mdape_threshold = 2.0  # 2% threshold for "close" performance
    
    if abs(mdape_diff) <= mdape_threshold:
        # Use multi-metric scoring for close performance
        def calculate_score(model):
            # Normalize metrics (lower is better for all except R²)
            mdape_score = model['mdape'] * 0.4
            mdae_score = (model['mdae'] / 1000) * 0.2  # Scale to reasonable range
            rmse_score = (model.get('rmse', model['mdae'] * 1.5) / 1000) * 0.2  # Fallback if no RMSE
            r2_score = (1 - model['adjusted_r2_score']) * 0.2  # Invert R² (lower is better for consistency)
            
            return mdape_score + mdae_score + rmse_score + r2_score
        
        score1 = calculate_score(model1)
        score2 = calculate_score(model2)
        
        if score1 < score2:
            reason = f"Multi-metric scoring (MdAPE diff: {mdape_diff:.2f}% < {mdape_threshold}%)"
            return model1_name, score2 - score1, reason
        else:
            reason = f"Multi-metric scoring (MdAPE diff: {mdape_diff:.2f}% < {mdape_threshold}%)"
            return model2_name, score1 - score2, reason
    else:
        # Clear MdAPE winner
        if mdape_diff < 0:
            reason = f"Clear MdAPE advantage ({abs(mdape_diff):.2f}%)"
            return model1_name, abs(mdape_diff), reason
        else:
            reason = f"Clear MdAPE advantage ({abs(mdape_diff):.2f}%)"
            return model2_name, abs(mdape_diff), reason

def determine_optimal_routing_strategy(best_models_subset, results_df=None, verbose=True):
    """
    Determine optimal director routing strategy using head-to-head model comparisons.
    
    Compares fee-based, no-fee-based, and general models to determine whether to use:
    - Single model approach (general model for all directors)
    - Dual model approach (separate models for fee/no-fee subsets)
    
    Logic:
    1. Fee wins + No-fee wins → Dual model approach
    2. General wins both → Single general model  
    3. Fee wins + General wins → Dual model approach
    4. General wins + No-fee wins → Single model (general preferred)
    
    Args:
        best_models_subset: Output from get_best_models_by_subset() containing fee/no-fee models
        results_df: Pre-loaded pandas DataFrame with model results (avoids redundant loading)
        verbose: Whether to print decision details
    
    Returns:
        dict: {
            'director_efc_usd': {
                'strategy': 'dual_model' | 'single_model',
                'fee_subset_model': model_info,
                'no_fee_subset_model': model_info,
                'reasoning': str,
                'performance_comparison': {...}
            }
        }
    """
    if results_df is None or results_df.empty:
        if verbose:
            print("Error: No model results data provided for routing strategy determination")
        return {}
    
    routing_results = {}
    
    # Only process director models (ATL doesn't have intelligent routing)
    if 'director_efc_usd' in best_models_subset:
        director_models = best_models_subset['director_efc_usd']
        
        # Check if we have all required models
        if ('overall' not in director_models or 
            'fee_subset' not in director_models or 
            'no_fee_subset' not in director_models):
            if verbose:
                print("Warning: Missing required models for routing strategy determination")
            return {}
        
        general_model = director_models['overall']
        fee_model = director_models['fee_subset']
        no_fee_model = director_models['no_fee_subset']
        
        def compare_models(model1, model2, model1_name, model2_name):
            """Compare two models using robust multi-metric scoring"""
            winner, diff, reason = _compare_models_robust(model1, model2, model1_name, model2_name)
            return winner, diff
        
        # Head-to-head comparisons
        fee_winner, fee_diff = compare_models(fee_model, general_model, 'fee', 'general')
        no_fee_winner, no_fee_diff = compare_models(no_fee_model, general_model, 'no_fee', 'general')
        
        # Decision logic
        if fee_winner == 'fee' and no_fee_winner == 'no_fee':
            # Scenario 1: Fee + No-Fee win → Dual model approach
            strategy = 'dual_model'
            fee_subset_model = fee_model
            no_fee_subset_model = no_fee_model
            reasoning = f"Fee model beats general by {abs(fee_diff):.1f}% MdAPE, No-fee model beats general by {abs(no_fee_diff):.1f}% MdAPE"
            
        elif fee_winner == 'general' and no_fee_winner == 'general':
            # Scenario 2: General wins both → Single general model
            strategy = 'single_model'
            fee_subset_model = general_model
            no_fee_subset_model = general_model
            reasoning = f"General model beats fee by {abs(fee_diff):.1f}% MdAPE and no-fee by {abs(no_fee_diff):.1f}% MdAPE"
            
        elif fee_winner == 'fee' and no_fee_winner == 'general':
            # Scenario 3: Fee + General win → Dual model approach
            strategy = 'dual_model'
            fee_subset_model = fee_model
            no_fee_subset_model = general_model
            reasoning = f"Fee model beats general by {abs(fee_diff):.1f}% MdAPE, General beats no-fee by {abs(no_fee_diff):.1f}% MdAPE"
            
        else:  # fee_winner == 'general' and no_fee_winner == 'no_fee'
            # Scenario 4: General + No-Fee win → Dual model approach
            strategy = 'dual_model'
            fee_subset_model = general_model  # General wins vs fee
            no_fee_subset_model = no_fee_model  # No-fee wins vs general
            reasoning = f"General beats fee by {abs(fee_diff):.1f}% MdAPE, No-fee beats general by {abs(no_fee_diff):.1f}% MdAPE"
        
        routing_results['director_efc_usd'] = {
            'strategy': strategy,
            'fee_subset_model': {
                'model_key': fee_subset_model['model_key'],
                'mdape': fee_subset_model['mdape'],
                'mdae': fee_subset_model['mdae'],
                'rmse': fee_subset_model['rmse'],
                'adjusted_r2_score': fee_subset_model['adjusted_r2_score']
            },
            'no_fee_subset_model': {
                'model_key': no_fee_subset_model['model_key'],
                'mdape': no_fee_subset_model['mdape'],
                'mdae': no_fee_subset_model['mdae'],
                'rmse': no_fee_subset_model['rmse'],
                'adjusted_r2_score': no_fee_subset_model['adjusted_r2_score']
            },
            'reasoning': reasoning,
            'routing_type': 'full_intelligent',
            'performance_comparison': {
                'fee_vs_general': {
                    'winner': fee_winner,
                    'fee_model': f"{fee_model['model_key']} (MdAPE: {fee_model['mdape']:.1f}%, MdAE: ${fee_model['mdae']:,.0f})",
                    'general_model': f"{general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%, MdAE: ${general_model['mdae']:,.0f})",
                    'difference': fee_diff
                },
                'no_fee_vs_general': {
                    'winner': no_fee_winner,
                    'no_fee_model': f"{no_fee_model['model_key']} (MdAPE: {no_fee_model['mdape']:.1f}%, MdAE: ${no_fee_model['mdae']:,.0f})",
                    'general_model': f"{general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%, MdAE: ${general_model['mdae']:,.0f})",
                    'difference': no_fee_diff
                }
            }
        }
        
        if verbose:
            print(f"\n🎯 Optimal Director Routing Strategy Analysis:")
            print(f"  ")
            print(f"  📊 Head-to-Head Comparisons:")
            print(f"    Fee Subset: {fee_model['model_key']} (MdAPE: {fee_model['mdape']:.1f}%) vs")
            print(f"                {general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%)")
            print(f"    Winner: {fee_winner.title()} model")
            print(f"  ")
            print(f"    No-Fee Subset: {no_fee_model['model_key']} (MdAPE: {no_fee_model['mdape']:.1f}%) vs")
            print(f"                   {general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%)")
            print(f"    Winner: {no_fee_winner.title()} model")
            print(f"  ")
            print(f"  🏆 Optimal Strategy: {strategy.upper().replace('_', ' ')}")
            print(f"    Reasoning: {reasoning}")
            print(f"  ")
            if strategy == 'dual_model':
                print(f"  📋 Deployment Configuration:")
                print(f"    Directors WITH fee history → {fee_subset_model['model_key']}")
                print(f"    Directors WITHOUT fee history → {no_fee_subset_model['model_key']}")
            else:
                print(f"  📋 Deployment Configuration:")
                print(f"    ALL directors → {fee_subset_model['model_key']}")
    
    return routing_results

def determine_fee_vs_general_strategy(best_models_subset, results_df=None, verbose=True):
    """
    Determine routing strategy when only fee subset models are available.
    
    Compares best fee subset model vs best general model to decide:
    - If fee model is better: Use fee model for directors with fee history, general for others
    - If general model is better or tied: Use general model for all directors
    
    Args:
        best_models_subset: Output from get_best_models_by_subset()
        results_df: Pre-loaded pandas DataFrame with model results
        verbose: Whether to print decision details
    
    Returns:
        dict: Routing strategy information with same structure as determine_optimal_routing_strategy
    """
    if results_df is None or results_df.empty:
        if verbose:
            print("Error: No model results data provided for fee vs general strategy determination")
        return {}
    
    routing_results = {}
    
    if 'director_efc_usd' in best_models_subset:
        director_models = best_models_subset['director_efc_usd']
        
        if 'overall' not in director_models or 'fee_subset' not in director_models:
            if verbose:
                print("Warning: Missing required models for fee vs general strategy determination")
            return {}
        
        general_model = director_models['overall']
        fee_model = director_models['fee_subset']
        
        # Compare fee model vs general model
        fee_winner, fee_diff, fee_reason = _compare_models_robust(fee_model, general_model, 'fee', 'general')
        
        if fee_winner == 'fee':
            # Fee model is better - use hybrid approach
            strategy = 'dual_model'
            fee_subset_model = fee_model
            no_fee_subset_model = general_model  # Use general for directors without fee history
            reasoning = f"Fee model beats general by {abs(fee_diff):.1f}% MdAPE. Using fee model for directors with fee history, general model for others"
        else:
            # General model is better or tied - use single approach
            strategy = 'single_model'
            fee_subset_model = general_model
            no_fee_subset_model = general_model
            reasoning = f"General model beats or ties fee model by {abs(fee_diff):.1f}% MdAPE. Using general model for all directors"
        
        routing_results['director_efc_usd'] = {
            'strategy': strategy,
            'fee_subset_model': fee_subset_model,
            'no_fee_subset_model': no_fee_subset_model,
            'reasoning': reasoning,
            'routing_type': 'fee_vs_general',
            'performance_comparison': {
                'fee_vs_general': {
                    'winner': fee_winner,
                    'fee_model': f"{fee_model['model_key']} (MdAPE: {fee_model['mdape']:.1f}%, MdAE: ${fee_model['mdae']:,.0f})",
                    'general_model': f"{general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%, MdAE: ${general_model['mdae']:,.0f})",
                    'difference': fee_diff
                }
            }
        }
        
        if verbose:
            print(f"\n🎯 Fee vs General Routing Strategy Analysis:")
            print(f"  📊 Head-to-Head Comparison:")
            print(f"    Fee Subset: {fee_model['model_key']} (MdAPE: {fee_model['mdape']:.1f}%) vs")
            print(f"    General: {general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%)")
            print(f"    Winner: {fee_winner.title()} model")
            print(f"  ")
            print(f"  🏆 Optimal Strategy: {strategy.upper().replace('_', ' ')}")
            print(f"    Reasoning: {reasoning}")
            print(f"  ")
            if strategy == 'dual_model':
                print(f"  📋 Deployment Configuration:")
                print(f"    Directors WITH fee history → {fee_subset_model['model_key']}")
                print(f"    Directors WITHOUT fee history → {no_fee_subset_model['model_key']}")
            else:
                print(f"  📋 Deployment Configuration:")
                print(f"    ALL directors → {fee_subset_model['model_key']}")
    
    return routing_results

def determine_no_fee_vs_general_strategy(best_models_subset, results_df=None, verbose=True):
    """
    Determine routing strategy when only no-fee subset models are available.
    
    Compares best no-fee subset model vs best general model to decide:
    - If no-fee model is better: Use no-fee model for directors without fee history, general for others
    - If general model is better or tied: Use general model for all directors
    
    Args:
        best_models_subset: Output from get_best_models_by_subset()
        results_df: Pre-loaded pandas DataFrame with model results
        verbose: Whether to print decision details
    
    Returns:
        dict: Routing strategy information with same structure as determine_optimal_routing_strategy
    """
    if results_df is None or results_df.empty:
        if verbose:
            print("Error: No model results data provided for no-fee vs general strategy determination")
        return {}
    
    routing_results = {}
    
    if 'director_efc_usd' in best_models_subset:
        director_models = best_models_subset['director_efc_usd']
        
        if 'overall' not in director_models or 'no_fee_subset' not in director_models:
            if verbose:
                print("Warning: Missing required models for no-fee vs general strategy determination")
            return {}
        
        general_model = director_models['overall']
        no_fee_model = director_models['no_fee_subset']
        
        # Compare no-fee model vs general model
        no_fee_winner, no_fee_diff, no_fee_reason = _compare_models_robust(no_fee_model, general_model, 'no_fee', 'general')
        
        if no_fee_winner == 'no_fee':
            # No-fee model is better - use hybrid approach
            strategy = 'dual_model'
            fee_subset_model = general_model  # Use general for directors with fee history
            no_fee_subset_model = no_fee_model
            reasoning = f"No-fee model beats general by {abs(no_fee_diff):.1f}% MdAPE. Using general model for directors with fee history, no-fee model for others"
        else:
            # General model is better or tied - use single approach
            strategy = 'single_model'
            fee_subset_model = general_model
            no_fee_subset_model = general_model
            reasoning = f"General model beats or ties no-fee model by {abs(no_fee_diff):.1f}% MdAPE. Using general model for all directors"
        
        routing_results['director_efc_usd'] = {
            'strategy': strategy,
            'fee_subset_model': fee_subset_model,
            'no_fee_subset_model': no_fee_subset_model,
            'reasoning': reasoning,
            'routing_type': 'no_fee_vs_general',
            'performance_comparison': {
                'no_fee_vs_general': {
                    'winner': no_fee_winner,
                    'no_fee_model': f"{no_fee_model['model_key']} (MdAPE: {no_fee_model['mdape']:.1f}%, MdAE: ${no_fee_model['mdae']:,.0f})",
                    'general_model': f"{general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%, MdAE: ${general_model['mdae']:,.0f})",
                    'difference': no_fee_diff
                }
            }
        }
        
        if verbose:
            print(f"\n🎯 No-Fee vs General Routing Strategy Analysis:")
            print(f"  📊 Head-to-Head Comparison:")
            print(f"    No-Fee Subset: {no_fee_model['model_key']} (MdAPE: {no_fee_model['mdape']:.1f}%) vs")
            print(f"    General: {general_model['model_key']} (MdAPE: {general_model['mdape']:.1f}%)")
            print(f"    Winner: {no_fee_winner.title()} model")
            print(f"  ")
            print(f"  🏆 Optimal Strategy: {strategy.upper().replace('_', ' ')}")
            print(f"    Reasoning: {reasoning}")
            print(f"  ")
            if strategy == 'dual_model':
                print(f"  📋 Deployment Configuration:")
                print(f"    Directors WITH fee history → {fee_subset_model['model_key']}")
                print(f"    Directors WITHOUT fee history → {no_fee_subset_model['model_key']}")
            else:
                print(f"  📋 Deployment Configuration:")
                print(f"    ALL directors → {fee_subset_model['model_key']}")
    
    return routing_results

def identify_best_models(model_results, org_metrics, team_metrics, mape_threshold=20.0, verbose=True):
    """
    Identify best performing models for detailed analysis, including intelligent routing models.
    
    For ATL: Returns single best model
    For Director: Returns models based on optimal routing strategy (single or dual model approach)
    
    Returns a set of model_keys for models that should have feature importance/SHAP calculated.
    Uses MdAPE (lower is better) as primary metric, MdAE as tie-breaker, adjusted R² as fallback.
    """
    model_keys = set()
    
    # Get subset-aware model selection and shared data
    best_models_subset, results_df = get_best_models_by_subset(model_results, verbose=verbose)
    
    # Add ATL models (always single model)
    if 'atl_efc_usd' in best_models_subset:
        atl_model = best_models_subset['atl_efc_usd']['overall']
        model_keys.add(atl_model['model_key'])
    
    # Add Director models based on routing strategy
    if 'director_efc_usd' in best_models_subset:
        director_models = best_models_subset['director_efc_usd']
        
        # Check what intelligent models are available
        has_fee_subset = 'fee_subset' in director_models
        has_no_fee_subset = 'no_fee_subset' in director_models
        
        if has_fee_subset and has_no_fee_subset:
            # FULL INTELLIGENT ROUTING - Both subsets available
            routing_strategy = determine_optimal_routing_strategy(best_models_subset, results_df, verbose=False)
            
            if routing_strategy and 'director_efc_usd' in routing_strategy:
                strategy_info = routing_strategy['director_efc_usd']
                
                if strategy_info['strategy'] == 'dual_model':
                    # Add both models for dual model approach
                    model_keys.add(strategy_info['fee_subset_model']['model_key'])
                    model_keys.add(strategy_info['no_fee_subset_model']['model_key'])
                    
                    if verbose:
                        print(f"\n🎯 Intelligent Routing: DUAL MODEL approach selected")
                        print(f"   Fee model: {strategy_info['fee_subset_model']['model_key']}")
                        print(f"   No-fee model: {strategy_info['no_fee_subset_model']['model_key']}")
                else:
                    # Add single model for single model approach
                    model_keys.add(strategy_info['fee_subset_model']['model_key'])
                    
                    if verbose:
                        print(f"\n🎯 Intelligent Routing: SINGLE MODEL approach selected")
                        print(f"   All directors: {strategy_info['fee_subset_model']['model_key']}")
            else:
                # Fallback to overall best if routing strategy determination fails
                model_keys.add(director_models['overall']['model_key'])
                if verbose:
                    print(f"\n⚠️  Full routing strategy determination failed, using overall best director model")
                    
        elif has_fee_subset and not has_no_fee_subset:
            # PARTIAL INTELLIGENT ROUTING - Fee vs General
            routing_strategy = determine_fee_vs_general_strategy(best_models_subset, results_df, verbose=False)
            
            if routing_strategy and 'director_efc_usd' in routing_strategy:
                strategy_info = routing_strategy['director_efc_usd']
                
                if strategy_info['strategy'] == 'dual_model':
                    # Add both models for hybrid approach
                    model_keys.add(strategy_info['fee_subset_model']['model_key'])
                    model_keys.add(strategy_info['no_fee_subset_model']['model_key'])
                    
                    if verbose:
                        print(f"\n🎯 Partial Intelligent Routing: FEE vs GENERAL approach selected")
                        print(f"   Fee model: {strategy_info['fee_subset_model']['model_key']}")
                        print(f"   No-fee fallback: {strategy_info['no_fee_subset_model']['model_key']}")
                else:
                    # Add single model for single model approach
                    model_keys.add(strategy_info['fee_subset_model']['model_key'])
                    
                    if verbose:
                        print(f"\n🎯 Partial Intelligent Routing: SINGLE MODEL approach selected")
                        print(f"   All directors: {strategy_info['fee_subset_model']['model_key']}")
            else:
                # Fallback to overall best if routing strategy determination fails
                model_keys.add(director_models['overall']['model_key'])
                if verbose:
                    print(f"\n⚠️  Fee vs general routing strategy determination failed, using overall best director model")
                    
        elif not has_fee_subset and has_no_fee_subset:
            # PARTIAL INTELLIGENT ROUTING - No-fee vs General
            routing_strategy = determine_no_fee_vs_general_strategy(best_models_subset, results_df, verbose=False)
            
            if routing_strategy and 'director_efc_usd' in routing_strategy:
                strategy_info = routing_strategy['director_efc_usd']
                
                if strategy_info['strategy'] == 'dual_model':
                    # Add both models for hybrid approach
                    model_keys.add(strategy_info['fee_subset_model']['model_key'])
                    model_keys.add(strategy_info['no_fee_subset_model']['model_key'])
                    
                    if verbose:
                        print(f"\n🎯 Partial Intelligent Routing: NO-FEE vs GENERAL approach selected")
                        print(f"   Fee fallback: {strategy_info['fee_subset_model']['model_key']}")
                        print(f"   No-fee model: {strategy_info['no_fee_subset_model']['model_key']}")
                else:
                    # Add single model for single model approach
                    model_keys.add(strategy_info['fee_subset_model']['model_key'])
                    
                    if verbose:
                        print(f"\n🎯 Partial Intelligent Routing: SINGLE MODEL approach selected")
                        print(f"   All directors: {strategy_info['fee_subset_model']['model_key']}")
            else:
                # Fallback to overall best if routing strategy determination fails
                model_keys.add(director_models['overall']['model_key'])
                if verbose:
                    print(f"\n⚠️  No-fee vs general routing strategy determination failed, using overall best director model")
                    
        else:
            # FALLBACK - No intelligent models available, use overall best
            model_keys.add(director_models['overall']['model_key'])
            if verbose:
                print(f"\n📊 No intelligent models available, using overall best director model")
    
    if verbose:
        print(f"\n🎯 Selected {len(model_keys)} models for detailed analysis:")
        for model_key in sorted(model_keys):
            print(f"   - {model_key}")
    
    return model_keys


def save_best_model_predictions(best_models, model_cache, data):
    """
    Save predictions for best models with organization, team, and season information.
    
    For each best model, saves a CSV with:
    - gravity_buying_organization_desc
    - gravity_buying_team
    - season_production_id
    - prediction (in actual dollars, not log-transformed)
    - actual (for comparison)
    """
    
    for model_key in best_models:
        if model_key not in model_cache:
            continue
            
        print(f"\nSaving predictions for: {model_key}")
        
        cache_entry = model_cache[model_key]
        model = cache_entry['model']
        target = cache_entry['target']
        
        # Get test indices and features
        from models.model_wrappers import CatBoostWrapper
        if isinstance(model, CatBoostWrapper):
            X_test = cache_entry['X_test_catboost']
        else:
            X_test = cache_entry['X_test_encoded']
        
        y_test = cache_entry['y_test']
        features_encoded = cache_entry['features_encoded']
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Load the appropriate dataset based on the model's target
        if 'atl_efc_usd' in model_key:
            model_data = pd.read_csv('data/processed_data/atl_title_data.csv')
        else:  # director_efc_usd
            model_data = pd.read_csv('data/processed_data/director_title_data.csv')
        
        # Get the test indices from the original data
        # We need to trace back through the encoding process
        test_indices = y_test.index
        
        # Create prediction dataframe
        predictions_df = pd.DataFrame({
            'gravity_buying_organization_desc': model_data.loc[test_indices, 'gravity_buying_organization_desc'].values,
            'gravity_buying_team': model_data.loc[test_indices, 'gravity_buying_team'].values,
            'season_production_id': model_data.loc[test_indices, 'season_production_id'].values,
            'prediction': y_pred,
            'actual': y_test.values
        })
        
        # Calculate error for reference
        predictions_df['error'] = predictions_df['prediction'] - predictions_df['actual']
        predictions_df['error_pct'] = (predictions_df['error'] / predictions_df['actual'] * 100).round(2)
        
        # Save full predictions with error metrics
        full_output_path = os.path.join('data', 'model_outputs', 'predictions', f'predictions_full_{model_key}.csv')
        predictions_df.to_csv(full_output_path, index=False)
        print(f"  Saved full predictions to {full_output_path}")
        
        # Save simplified version with key columns including actual values
        simple_predictions_df = predictions_df[['gravity_buying_organization_desc', 
                                               'gravity_buying_team', 
                                               'season_production_id', 
                                               'prediction',
                                               'actual']]
        
        simple_output_path = os.path.join('data', 'model_outputs', 'predictions', f'predictions_{model_key}.csv')
        simple_predictions_df.to_csv(simple_output_path, index=False)
        print(f"  Saved simplified predictions to {simple_output_path}")
        
        # Print summary statistics
        mse = mean_squared_error(predictions_df['actual'], predictions_df['prediction'])
        print(f"  Prediction Summary:")
        print(f"    Mean prediction: ${predictions_df['prediction'].mean():,.0f}")
        print(f"    Median prediction: ${predictions_df['prediction'].median():,.0f}")
        print(f"    Mean actual: ${predictions_df['actual'].mean():,.0f}")
        print(f"    Median actual: ${predictions_df['actual'].median():,.0f}")
        print(f"    Mean squared error: {mse:,.0f}")
        print(f"    Mean absolute error: ${np.abs(predictions_df['error']).mean():,.0f}")
        print(f"    Adjusted R² score: {adjusted_r2_score_legacy(predictions_df['actual'], predictions_df['prediction']):.3f}")


def calculate_metrics_by_group(y_true, y_pred, groups, group_name, model_key=None):
    """Calculate metrics grouped by a specific column."""
    results = []
    unique_groups = groups.unique()
    
    for group in unique_groups:
        mask = groups == group
        group_size = mask.sum()
        
        if group_size >= 2:  # Need at least 2 samples for meaningful metrics
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            mse = mean_squared_error(group_y_true, group_y_pred)
            mae = np.median(np.abs(group_y_true - group_y_pred))  # Median Absolute Error for consistency with MAPE
            
            # Calculate Median Absolute Percentage Error (MAPE)
            try:
                # Avoid division by zero by filtering out zero actual values
                non_zero_mask = group_y_true != 0
                if non_zero_mask.sum() > 0:
                    mape = np.median(np.abs((group_y_true[non_zero_mask] - group_y_pred[non_zero_mask]) / group_y_true[non_zero_mask]) * 100)
                else:
                    mape = np.nan  # All actual values are zero
            except:
                mape = np.nan
            
            try:
                # Determine minimum sample size needed for adjusted R²
                from utils.adjusted_r2 import estimate_feature_count_from_context
                estimated_features = estimate_feature_count_from_context(model_key, group_size)
                min_samples_for_adjusted = estimated_features + 2  # Need n > k + 1
                
                # Use regular R² for groups that are too small for adjusted R²
                if group_size < min_samples_for_adjusted:
                    r2 = r2_score(group_y_true, group_y_pred)
                    if group_size > 20:  # Log for groups that should be big enough but aren't
                        logging.info(f"Using regular R² for group '{group}' (size={group_size}, needs >={min_samples_for_adjusted} for adjusted R²)")
                else:  # Large enough groups - use adjusted R²
                    r2 = adjusted_r2_score_legacy(group_y_true, group_y_pred, model_key)
            except Exception as e:
                # Log the specific error for debugging
                logging.warning(f"R² calculation failed for group '{group}' (size={group_size}): {e}")
                r2 = np.nan  # Handle edge cases
            
            results.append({
                group_name: group,
                'count': group_size,
                'mse': mse,
                'mdae': mae,
                'mdape': mape,
                'adjusted_r2_score': r2,
                'rmse': np.sqrt(mse)
            })
        # Single sample teams (group_size == 1) are intentionally excluded
        # as they cannot provide meaningful statistical evaluation
    
    return pd.DataFrame(results)


def calculate_error_rate_metrics(y_true, y_pred):
    """
    Calculate error rate metrics for model evaluation.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        dict: Dictionary containing error rate metrics including:
            - pct_titles_under_20_error: Percentage of predictions within 20% error
            - pct_titles_under_6_error: Percentage of predictions within 6% error
            - mean_percentage_error: Mean absolute percentage error
            - mape: Mean Absolute Percentage Error (same as mean_percentage_error)
            - median_percentage_error: Median absolute percentage error
            - max_percentage_error: Maximum percentage error
            - min_percentage_error: Minimum percentage error
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate percentage errors
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100

    # Calculate metrics
    pct_under_20_error = (percentage_errors < 20).mean() * 100
    pct_under_6_error = (percentage_errors < 6).mean() * 100

    # Additional error statistics
    mean_pct_error = percentage_errors.mean()
    median_pct_error = np.median(percentage_errors)

    # Mean Absolute Percentage Error (MAPE) - same as mean_pct_error but explicitly named
    mape = mean_pct_error
    
    # Median Absolute Percentage Error (MdAPE)
    mdape = median_pct_error

    return {
        'pct_titles_under_20_error': pct_under_20_error,
        'pct_titles_under_6_error': pct_under_6_error,
        'mean_percentage_error': mean_pct_error,
        'mdape': mdape,
        'median_percentage_error': median_pct_error,
        'max_percentage_error': percentage_errors.max(),
        'min_percentage_error': percentage_errors.min()
    }


def generate_intelligent_routing_report(model_results, output_path='data/model_outputs/model_metrics'):
    """
    Generate a text report showing performance of intelligent routing models.

    Compares models with/without fee history and shows which routing strategy is optimal.

    Args:
        model_results: List of model result dictionaries
        output_path: Directory to save the report

    Returns:
        str: Path to generated report file, or None if no intelligent models found
    """
    from datetime import datetime
    import os

    # Convert to DataFrame for easier filtering
    results_df = pd.DataFrame(model_results)

    # Find intelligent routing models (those with _with_fees or _no_fees suffix)
    intelligent_models = results_df[
        results_df['model_key'].str.contains('_with_fees|_no_fees', regex=True, na=False)
    ]

    if intelligent_models.empty:
        print("⚠️  No intelligent routing models found. Skipping intelligent routing report.")
        return None

    # Group by base model name (without _with_fees/_no_fees and any subsequent suffixes)
    # Pattern matches _with_fees or _no_fees followed by optional additional text
    intelligent_models['base_model'] = intelligent_models['model_key'].str.replace(
        r'_(with_fees|no_fees).*$', '', regex=True
    )

    # Create report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_path, f"intelligent_routing_analysis_{timestamp}.txt")

    os.makedirs(output_path, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("INTELLIGENT ROUTING MODEL PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report compares models trained separately on titles with and without\n")
        f.write("previous director fee history to determine the optimal routing strategy.\n\n")

        # Analyze each base model
        base_models = intelligent_models['base_model'].unique()

        for base_model in sorted(base_models):
            subset_models = intelligent_models[intelligent_models['base_model'] == base_model]

            # Models may have additional suffixes after _with_fees/_no_fees (e.g., _intelligent)
            with_fees = subset_models[subset_models['model_key'].str.contains('_with_fees', regex=False)]
            no_fees = subset_models[subset_models['model_key'].str.contains('_no_fees', regex=False)]

            f.write("=" * 80 + "\n")
            f.write(f"MODEL: {base_model}\n")
            f.write("=" * 80 + "\n\n")

            # WITH FEES MODEL
            if not with_fees.empty:
                wf = with_fees.iloc[0]
                f.write("WITH FEE HISTORY MODEL\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model Key: {wf['model_key']}\n")
                f.write(f"Algorithm: {wf.get('model_type', 'N/A')}\n")
                f.write(f"Feature Set: {wf.get('feature_set', 'N/A')}\n\n")

                f.write("Performance Metrics:\n")
                f.write(f"  Sample Size: {wf.get('sample_size', 'N/A')} titles\n")
                f.write(f"  MdAPE: {wf.get('mdape', 'N/A'):.2f}%\n")
                f.write(f"  R² Score: {wf.get('adjusted_r2_score', 'N/A'):.3f}\n")
                f.write(f"  Adjusted R²: {wf.get('adjusted_r2_score', 'N/A'):.3f}\n")
                f.write(f"  RMSE: ${wf.get('rmse', 0):,.0f}\n")
                f.write(f"  MdAE: ${wf.get('mdae', 0):,.0f}\n\n")

                f.write("Error Rate Distribution:\n")
                f.write(f"  Excellent (<6% error): {wf.get('pct_titles_under_6_error', 'N/A'):.1f}%\n")
                f.write(f"  Good (<20% error): {wf.get('pct_titles_under_20_error', 'N/A'):.1f}%\n")
                f.write(f"  Challenging (≥20% error): {100 - wf.get('pct_titles_under_20_error', 0):.1f}%\n\n")
            else:
                f.write("WITH FEE HISTORY MODEL: Not available\n\n")

            # NO FEES MODEL
            if not no_fees.empty:
                nf = no_fees.iloc[0]
                f.write("WITHOUT FEE HISTORY MODEL\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model Key: {nf['model_key']}\n")
                f.write(f"Algorithm: {nf.get('model_type', 'N/A')}\n")
                f.write(f"Feature Set: {nf.get('feature_set', 'N/A')}\n\n")

                f.write("Performance Metrics:\n")
                f.write(f"  Sample Size: {nf.get('sample_size', 'N/A')} titles\n")
                f.write(f"  MdAPE: {nf.get('mdape', 'N/A'):.2f}%\n")
                f.write(f"  R² Score: {nf.get('adjusted_r2_score', 'N/A'):.3f}\n")
                f.write(f"  Adjusted R²: {nf.get('adjusted_r2_score', 'N/A'):.3f}\n")
                f.write(f"  RMSE: ${nf.get('rmse', 0):,.0f}\n")
                f.write(f"  MdAE: ${nf.get('mdae', 0):,.0f}\n\n")

                f.write("Error Rate Distribution:\n")
                f.write(f"  Excellent (<6% error): {nf.get('pct_titles_under_6_error', 'N/A'):.1f}%\n")
                f.write(f"  Good (<20% error): {nf.get('pct_titles_under_20_error', 'N/A'):.1f}%\n")
                f.write(f"  Challenging (≥20% error): {100 - nf.get('pct_titles_under_20_error', 0):.1f}%\n\n")
            else:
                f.write("WITHOUT FEE HISTORY MODEL: Not available\n\n")

            # COMPARISON
            if not with_fees.empty and not no_fees.empty:
                wf = with_fees.iloc[0]
                nf = no_fees.iloc[0]

                f.write("SIDE-BY-SIDE COMPARISON\n")
                f.write("-" * 40 + "\n")

                mape_diff = wf.get('mdape', 0) - nf.get('mdape', 0)
                r2_diff = wf.get('adjusted_r2_score', 0) - nf.get('adjusted_r2_score', 0)

                f.write(f"                           With Fees    No Fees      Difference\n")
                f.write(f"  MdAPE:                  {wf.get('mdape', 0):7.2f}%    {nf.get('mdape', 0):7.2f}%    {mape_diff:+7.2f}%\n")
                f.write(f"  Adjusted R²:            {wf.get('adjusted_r2_score', 0):7.3f}     {nf.get('adjusted_r2_score', 0):7.3f}     {r2_diff:+7.3f}\n")
                f.write(f"  Good Predictions:       {wf.get('pct_titles_under_20_error', 0):7.1f}%    {nf.get('pct_titles_under_20_error', 0):7.1f}%    {wf.get('pct_titles_under_20_error', 0) - nf.get('pct_titles_under_20_error', 0):+7.1f}%\n\n")

                # Recommendation
                f.write("RECOMMENDATION\n")
                f.write("-" * 40 + "\n")

                # Use MdAPE as primary decision metric
                if abs(mape_diff) < 2.0:  # Within 2% - marginal difference
                    f.write("✓ SINGLE MODEL RECOMMENDED\n")
                    f.write(f"  The performance difference is marginal (MAPE difference: {abs(mape_diff):.2f}%).\n")
                    f.write(f"  Use the model with more data for stability and simplicity.\n")
                    better_model = "with_fees" if wf.get('sample_size', 0) >= nf.get('sample_size', 0) else "no_fees"
                    f.write(f"  Recommended: {better_model.upper().replace('_', ' ')} model\n")
                else:
                    f.write("✓ DUAL MODEL ROUTING RECOMMENDED\n")
                    f.write(f"  Significant performance difference detected (MAPE difference: {abs(mape_diff):.2f}%).\n")
                    f.write(f"  Use separate models for each subset:\n")
                    f.write(f"    - With fee history: {wf['model_key']}\n")
                    f.write(f"    - Without fee history: {nf['model_key']}\n")

                f.write("\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\n✅ INTELLIGENT ROUTING REPORT: {report_path}")
    return report_path
