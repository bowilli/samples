#from .embeddings_comp_features import generate_comp_embeddings_features
#from .production_features import generate_production_features
from typing import Tuple
import logging
import pandas as pd
import numpy as np
import datetime as dt
import kragle as kg
import argparse
import os
import sys

# Get the absolute path to the script directory and analysis directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)

# Add analysis directory to Python path so imports work from any location
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

def safe_import(relative_module: str, items: list, fallback_module: str = None):
    """
    Consolidated import handling with automatic fallback for relative vs absolute imports.
    
    Args:
        relative_module (str): Module path with relative import (e.g., 'sample_size_reporter' or 'utils.adjusted_r2')  
        items (list): List of items to import from the module (e.g., ['log_stage', 'log_outlier_analysis'])
        fallback_module (str): Fallback module path for absolute import (defaults to relative_module without 'utils.')
        
    Returns:
        dict: Dictionary mapping item names to imported functions/classes
    """
    if fallback_module is None:
        # Default fallback: remove 'utils.' prefix for absolute imports
        fallback_module = relative_module.replace('utils.', '')
    
    imports = {}
    
    try:
        # Try relative import first (when imported as module)
        if relative_module.startswith('.'):
            # Already relative
            module = __import__(relative_module, fromlist=items)
        else:
            # Make relative
            module = __import__(f'.{relative_module}', fromlist=items, level=1)
        
        for item in items:
            imports[item] = getattr(module, item)
            
    except (ImportError, ValueError, KeyError):
        # Fall back to absolute imports (when run as script or imported from outside)
        try:
            module = __import__(fallback_module, fromlist=items)
            for item in items:
                imports[item] = getattr(module, item)
        except ImportError as e:
            # Try with features prefix (when called from main.py)
            try:
                features_module = f"features.{relative_module}"
                module = __import__(features_module, fromlist=items)
                for item in items:
                    imports[item] = getattr(module, item)
            except ImportError:
                # Final fallback: try importing from current directory
                module = __import__(relative_module.split('.')[-1], fromlist=items)
                for item in items:
                    imports[item] = getattr(module, item)
    
    return imports


# ============================================================
# TITLE SCOPE CONFIGURATION & FUNNEL TRACKING
# ============================================================

TITLE_SCOPE_CONFIG = {
    'enabled': True,  # Set to True to apply filters
    'filters': {
        'launch_year_min': None,      # e.g., 2018 - only include titles launched in or after this year
        'launch_year_max': None,      # e.g., 2024 - only include titles launched in or before this year
        'countries': None,             # e.g., ['United States', 'United Kingdom'] - list of countries
        'programming_categories': None, # e.g., ['Series', 'Film'] - list of programming categories
        'buying_orgs': None,           # e.g., ['Netflix US'] - list of buying organizations
        'production_types': None,      # e.g., ['Original'] - list of production types
        'genres': None,                # e.g., ['Comedy', 'Drama'] - list of genres
        'has_director_fee': True,      # True = only titles with non-null/non-zero director_efc_usd, False/None = all titles
    },
    'description': 'Titles with director fees (non-null and >$0 director_efc_usd)'
}

ENABLE_FUNNEL_TRACKING = True  # Set to False to disable funnel tracking


def apply_scope_filters(title_df, config):
    """
    Apply configured filters to define in-scope titles.

    Args:
        title_df: DataFrame with title data
        config: TITLE_SCOPE_CONFIG dictionary

    Returns:
        Filtered DataFrame containing only in-scope titles
    """
    if not config['enabled']:
        print("📊 Scope: No filters applied - using all titles")
        return title_df

    filtered = title_df.copy()
    filters = config['filters']
    initial_count = len(filtered)

    print("📊 Applying scope filters:")

    # Launch year filters
    if filters['launch_year_min'] is not None:
        filtered = filtered[filtered['launch_year'] >= filters['launch_year_min']]
        print(f"  - launch_year >= {filters['launch_year_min']}: {len(filtered)} titles")

    if filters['launch_year_max'] is not None:
        filtered = filtered[filtered['launch_year'] <= filters['launch_year_max']]
        print(f"  - launch_year <= {filters['launch_year_max']}: {len(filtered)} titles")

    # Country filter
    if filters['countries'] is not None:
        if 'gravity_country_of_origin' in filtered.columns:
            filtered = filtered[filtered['gravity_country_of_origin'].isin(filters['countries'])]
            print(f"  - countries in {filters['countries']}: {len(filtered)} titles")

    # Programming category filter
    if filters['programming_categories'] is not None:
        if 'programming_category_desc' in filtered.columns:
            filtered = filtered[filtered['programming_category_desc'].isin(filters['programming_categories'])]
            print(f"  - programming categories in {filters['programming_categories']}: {len(filtered)} titles")

    # Buying org filter
    if filters['buying_orgs'] is not None:
        if 'gravity_buying_organization_desc' in filtered.columns:
            filtered = filtered[filtered['gravity_buying_organization_desc'].isin(filters['buying_orgs'])]
            print(f"  - buying orgs in {filters['buying_orgs']}: {len(filtered)} titles")

    # Production type filter
    if filters['production_types'] is not None:
        if 'os_production_type' in filtered.columns:
            filtered = filtered[filtered['os_production_type'].isin(filters['production_types'])]
            print(f"  - production types in {filters['production_types']}: {len(filtered)} titles")

    # Genre filter
    if filters['genres'] is not None:
        if 'genre_desc' in filtered.columns:
            filtered = filtered[filtered['genre_desc'].isin(filters['genres'])]
            print(f"  - genres in {filters['genres']}: {len(filtered)} titles")

    # Director fee filter (titles with actual director costs)
    if filters['has_director_fee'] is True:
        if 'director_efc_usd' in filtered.columns:
            before_count = len(filtered)
            filtered = filtered[
                (filtered['director_efc_usd'].notna()) &
                (filtered['director_efc_usd'] > 0)
            ]
            filtered_count = before_count - len(filtered)
            print(f"  - has_director_fee (non-null and >$0): {len(filtered)} titles (excluded {filtered_count})")
        else:
            print(f"  - WARNING: has_director_fee filter requested but 'director_efc_usd' column not found in title_df")

    print(f"\n✓ Scope filtering complete: {initial_count} → {len(filtered)} titles ({len(filtered)/initial_count*100:.1f}% retained)")

    return filtered


def _merge_with_logging(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merge_name: str,
    log_stage_func,
    log_key: str = None,
    log_description: str = None,
    on: str = 'season_production_id',
    left_on: str = None,
    right_on: str = None,
    how: str = 'inner',
    drop_columns: list = None,
    print_warning_if_empty: bool = False,
    funnel = None,
    funnel_stage_name: str = None
) -> pd.DataFrame:
    """
    Reusable function for merge operations with consistent logging and printing.

    Args:
        left_df (pd.DataFrame): Left DataFrame to merge
        right_df (pd.DataFrame): Right DataFrame to merge
        merge_name (str): Name for print statements (e.g., 'median_cast_efc')
        log_stage_func: Function to call for logging stage
        log_key (str): Key for log_stage function (defaults to f'after_{merge_name}_merge')
        log_description (str): Description for log_stage (defaults to f'After merging {merge_name}')
        on (str): Column to join on (default: 'season_production_id')
        left_on (str): Left column to join on (overrides 'on')
        right_on (str): Right column to join on (overrides 'on')
        how (str): Type of merge (default: 'inner')
        drop_columns (list): Columns to drop after merge
        print_warning_if_empty (bool): Print warning if right_df is empty
        funnel: TitleFunnel instance for tracking (optional)
        funnel_stage_name (str): Override stage name for funnel tracking

    Returns:
        pd.DataFrame: Merged dataframe
    """
    # Handle empty DataFrame case
    if right_df is None or right_df.empty:
        if print_warning_if_empty:
            print(f"WARNING: {merge_name} is empty, skipping merge")
            print(f"After {merge_name} (skipped): {len(left_df)} titles")
        return left_df

    # Perform merge
    merge_kwargs = {'how': how}
    if left_on and right_on:
        merge_kwargs.update({'left_on': left_on, 'right_on': right_on})
    else:
        merge_kwargs['on'] = on

    result_df = left_df.merge(right_df, **merge_kwargs)

    # Drop specified columns
    if drop_columns:
        result_df = result_df.drop(columns=drop_columns, errors='ignore')

    # Print and log results
    print(f"After {merge_name} merge: {len(result_df)} titles")

    if log_stage_func:
        log_key = log_key or f'after_{merge_name.lower().replace(" ", "_")}_merge'
        log_description = log_description or f'After merging {merge_name}'
        log_stage_func(log_key, result_df, log_description)

    # Track in funnel if provided
    if funnel is not None:
        stage_name = funnel_stage_name or f'After_{merge_name.replace(" ", "_")}_Merge'
        funnel.add_stage(
            stage_name=stage_name,
            current_titles=result_df['season_production_id'].unique(),
            dropout_reason=f'Missing required feature: {merge_name}'
        )

    return result_df

# Import main feature modules using consolidated function
feature_imports = safe_import('embeddings_features', 
    ['generate_embeddings_features', 'find_similar_seasons', 'calculate_median_comp_efc', 'generate_director_fee_comps_batch'])
comps_imports = safe_import('comps_features', 
    ['get_base_similar_seasons', 'calculate_base_median_efc'])  
production_imports = safe_import('production_features', 
    ['generate_director_profile_features'])
# Import reporting functions from dedicated module
reporter_imports = safe_import('utils.feature_reporter',
    ['generate_missing_features_report', 'generate_embedding_completeness_report', 'save_processed_datasets'])

# Extract imports to global scope for backward compatibility
generate_embeddings_features = feature_imports['generate_embeddings_features']
find_similar_seasons = feature_imports['find_similar_seasons'] 
calculate_median_comp_efc = feature_imports['calculate_median_comp_efc']
generate_director_fee_comps_batch = feature_imports['generate_director_fee_comps_batch']
get_base_similar_seasons = comps_imports['get_base_similar_seasons']
calculate_base_median_efc = comps_imports['calculate_base_median_efc']
generate_director_profile_features = production_imports['generate_director_profile_features']
# Extract reporting functions to global scope
generate_missing_features_report = reporter_imports['generate_missing_features_report']
generate_embedding_completeness_report = reporter_imports['generate_embedding_completeness_report']  
save_processed_datasets = reporter_imports['save_processed_datasets']

DATA_FILES = {
    'person': {
        'csv': os.path.join(ANALYSIS_DIR, 'data/raw_data/person_data_raw.csv'), 
        'sql': os.path.join(SCRIPT_DIR, 'sql/person_data_new.sql')
    },
    'title': {
        'csv': os.path.join(ANALYSIS_DIR, 'data/raw_data/title_data_raw.csv'), 
        'sql': os.path.join(SCRIPT_DIR, 'sql/title_data.sql')
    },
    'base_comps': {
        'csv': os.path.join(ANALYSIS_DIR, 'data/raw_data/base_comps_data_raw.csv'), 
        'sql': os.path.join(SCRIPT_DIR, 'sql/base_comps_data.sql')
    },
    'person_scores': {
        'csv': os.path.join(ANALYSIS_DIR, 'data/raw_data/person_scores.csv'), 
        'sql': os.path.join(SCRIPT_DIR, 'sql/person_scores.sql')
    }
}

MISSING_DATA_THRESHOLD = 0.40  # Updated from 0.20 to match actual implementation
EMBEDDING_THRESHOLD_ATL = 40  # Reverted from 50% - higher threshold reduced performance
EMBEDDING_THRESHOLD_DIRECTOR = 40  # Keep consistent threshold for both models
IQR_THRESHOLD = 4  # Increased from 3 to be less aggressive with outliers
ISOLATION_FOREST_CONTAMINATION = 0.02  # Changed to 0.02 after running experiments, best result

def _execute_sql_query(sql_file: str) -> pd.DataFrame:
    """
    Execute SQL query from file using Kragle Presto and return pandas DataFrame.
    
    Reads SQL query from a .sql file, executes it via Kragle's Presto interface,
    waits for completion, and returns results as a pandas DataFrame.
    
    Args:
        sql_file (str): Path to .sql file containing the query
        
    Returns:
        pd.DataFrame: Query results as a DataFrame
        
    Raises:
        Exception: If query execution fails or times out
    """
    with open(sql_file, 'r') as fh:
        sql_query = " ".join(fh.readlines())
    
    job = kg.genie.PrestoJob().script(sql_query).headers().session('query_max_stage_count', 200)
    result = job.execute()
    result.wait()
    result.raise_for_status()
    return result.pandas()

def _load_data_file(name: str, csv_path: str, sql_path: str, force_sql: bool = False) -> pd.DataFrame:
    """
    Load data from CSV cache or execute SQL query with intelligent fallback logic.
    
    Implements a caching strategy where data is first loaded from CSV if available,
    otherwise executes the SQL query and caches results. Supports forced SQL execution
    to refresh cached data.
    
    Args:
        name (str): Dataset name for logging (e.g., 'person', 'title')
        csv_path (str): Path to cached CSV file
        sql_path (str): Path to .sql file containing the query
        force_sql (bool): If True, bypass CSV and execute SQL query (default: False)
        
    Returns:
        pd.DataFrame: Dataset loaded from CSV or SQL execution
        
    Side Effects:
        - Creates CSV file at csv_path if loading from SQL
        - Creates directory structure if it doesn't exist
        - Removes 'Unnamed: 0' column if present from CSV
    """
    if force_sql or not os.path.exists(csv_path):
        if force_sql:
            print(f'Force regenerating {name} from SQL (--regenerate-sql flag used)')
        else:
            print(f'CSV not found, executing SQL query for {name}')
        df = _execute_sql_query(sql_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df
    else:
        df = pd.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        print(f'Loading {name} from existing CSV')
        return df

def load_data(force_sql: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets from CSV files or SQL queries."""
    return tuple(
        _load_data_file(name, config['csv'], config['sql'], force_sql) 
        for name, config in DATA_FILES.items()
    )

NON_NUMERIC_FEATURES = [
    'season_production_id', 'titlename', 'hitch_person_id', 'person_id', 
    'job_title_id', 'cast_number', 'vscore', 'embedding', 'percent_embedding'
]

def _filter_embedding_completeness(df: pd.DataFrame, threshold: int, target_type: str, funnel=None) -> pd.DataFrame:
    """
    Filter titles based on embedding completeness threshold for model quality.

    Calculates embedding completeness percentage for each title and filters to only
    include titles that meet the minimum threshold. This ensures model training data
    has sufficient embedding coverage for reliable predictions.

    Process:
    1. Groups by season_production_id to calculate per-title embedding completeness
    2. Computes percentage: (records with embeddings) / (total records) * 100
    3. Filters to titles meeting threshold (e.g., ≥40% embedding completeness)
    4. Removes any remaining records with missing embeddings
    5. Logs stage for data pipeline tracking

    Args:
        df (pd.DataFrame): Input person+title dataset with embedding column
        threshold (int): Minimum embedding completeness percentage (e.g., 40 for 40%)
        target_type (str): Model type ('ATL' or 'Director') for logging
        funnel: Optional TitleFunnel instance for tracking title dropouts

    Returns:
        pd.DataFrame: Filtered dataset containing only titles with sufficient embeddings

    Side Effects:
        - Logs stage using sample_size_reporter for pipeline tracking
        - Prints count of qualifying titles
        - Tracks in funnel if provided
    """
    # Import log_stage using consolidated function
    reporter_imports = safe_import('utils.sample_size_reporter', ['log_stage'])
    log_stage = reporter_imports['log_stage']

    log_stage(f'before_embedding_filter_{target_type.lower()}', df, f'Combined dataset before {target_type} embedding completeness filter (≥{threshold}%)')

    completeness = df.groupby('season_production_id').agg({
        'person_id': 'count',
        'hitch_person_id': 'count',
        'embedding': 'count'
    }).reset_index()

    completeness['percent_embedding'] = (
        completeness['embedding'] / completeness['person_id'] * 100
    )

    complete_titles = completeness[
        completeness['percent_embedding'] >= threshold
    ][['season_production_id', 'percent_embedding']]

    result = df.merge(complete_titles, on='season_production_id', how='inner')
    result = result.dropna(subset=['embedding'])

    log_stage(f'after_embedding_filter_{target_type.lower()}', result, f'After ≥{threshold}% embedding completeness filter for {target_type}')
    print(f'Titles with ≥{threshold}% embeddings for {target_type}: {result["season_production_id"].nunique()}')

    # Track in funnel if provided (only for Director)
    if funnel is not None and target_type == 'Director':
        funnel.add_stage(
            stage_name='04_Has_Director_Embeddings',
            current_titles=result['season_production_id'].unique(),
            dropout_reason=f'{target_type} embedding completeness <{threshold}% or missing embeddings'
        )

    return result

def _hierarchical_imputation(cleaned: pd.DataFrame, col: str, role: str, 
                             min_sub_team_size: int = 3, min_team_size: int = 5, min_org_size: int = 10,
                             is_categorical: bool = False) -> dict:
    """
    Apply hierarchical imputation to a single column using organizational structure.
    
    Args:
        cleaned: DataFrame containing the data
        col: Column name to impute
        role: Role name for logging ('Cast' or 'Director')
        min_sub_team_size: Minimum size for sub-team imputation
        min_team_size: Minimum size for team imputation  
        min_org_size: Minimum size for org imputation
        is_categorical: Whether to use mode (True) or median (False)
        
    Returns:
        dict: Statistics about imputation methods used
    """
    missing_count = cleaned[col].isnull().sum()
    if missing_count == 0:
        return {'sub_team_imputed': 0, 'team_imputed': 0, 'org_imputed': 0, 'fallback_imputed': 0}
    
    # Initialize imputation statistics
    imputation_stats = {
        'sub_team_imputed': 0, 
        'team_imputed': 0, 
        'org_imputed': 0, 
        'fallback_imputed': 0
    }
    
    # Get indices of missing values
    missing_mask = cleaned[col].isnull()
    
    # Helper function for production-type-aware imputation
    def _try_impute_with_prod_type(group_col, min_size, stat_key):
        if group_col not in cleaned.columns or 'os_production_type' not in cleaned.columns:
            return
        
        nonlocal missing_mask
        for group_val in cleaned[group_col].unique():
            if pd.isna(group_val):
                continue
            for prod_type in cleaned['os_production_type'].unique():
                if pd.isna(prod_type):
                    continue
                
                combined_mask = (cleaned[group_col] == group_val) & (cleaned['os_production_type'] == prod_type)
                combined_missing_mask = missing_mask & combined_mask
                
                if combined_missing_mask.sum() > 0:
                    combined_data = cleaned[combined_mask & ~cleaned[col].isnull()]
                    
                    if len(combined_data) >= min_size and len(combined_data[col].dropna()) > 0:
                        if is_categorical:
                            impute_value = combined_data[col].mode()
                            if len(impute_value) > 0:
                                cleaned.loc[combined_missing_mask, col] = impute_value.iloc[0]
                                imputation_stats[stat_key] += combined_missing_mask.sum()
                                missing_mask = cleaned[col].isnull()
                        else:
                            median_value = combined_data[col].median()
                            if not pd.isna(median_value):
                                cleaned.loc[combined_missing_mask, col] = median_value
                                imputation_stats[stat_key] += combined_missing_mask.sum()
                                missing_mask = cleaned[col].isnull()
    
    # Strategy 1: Production-type-aware imputation (try first)
    _try_impute_with_prod_type('gravity_sub_buying_team', min_sub_team_size, 'sub_team_imputed')
    _try_impute_with_prod_type('gravity_buying_team', min_team_size, 'team_imputed')
    _try_impute_with_prod_type('gravity_buying_organization_desc', min_org_size, 'org_imputed')
    
    # Strategy 2: Standard hierarchical imputation (fallback for remaining missing values)
    # Sub buying team imputation
    if 'gravity_sub_buying_team' in cleaned.columns:
        for sub_team in cleaned['gravity_sub_buying_team'].unique():
            if pd.isna(sub_team):
                continue
            
            sub_team_mask = cleaned['gravity_sub_buying_team'] == sub_team
            sub_team_missing_mask = missing_mask & sub_team_mask
            
            if sub_team_missing_mask.sum() > 0:  # Sub team has missing values
                sub_team_data = cleaned[sub_team_mask & ~cleaned[col].isnull()]
                
                if len(sub_team_data) >= min_sub_team_size and len(sub_team_data[col].dropna()) > 0:
                    if is_categorical:
                        impute_value = sub_team_data[col].mode()
                        if len(impute_value) > 0:
                            mode_value = impute_value.iloc[0]
                            # Handle categorical columns properly
                            if cleaned[col].dtype.name == 'category' and mode_value not in cleaned[col].cat.categories:
                                cleaned[col] = cleaned[col].cat.add_categories([mode_value])
                            cleaned.loc[sub_team_missing_mask, col] = mode_value
                            imputation_stats['sub_team_imputed'] += sub_team_missing_mask.sum()
                    else:
                        median_value = sub_team_data[col].median()
                        if not pd.isna(median_value):
                            cleaned.loc[sub_team_missing_mask, col] = median_value
                            imputation_stats['sub_team_imputed'] += sub_team_missing_mask.sum()
    
    # Strategy 2: Buying team imputation for remaining missing values
    missing_mask = cleaned[col].isnull()  # Recalculate after sub team imputation
    
    if 'gravity_buying_team' in cleaned.columns:
        for team in cleaned['gravity_buying_team'].unique():
            if pd.isna(team):
                continue
            
            team_mask = cleaned['gravity_buying_team'] == team
            team_missing_mask = missing_mask & team_mask
            
            if team_missing_mask.sum() > 0:  # Team has missing values
                team_data = cleaned[team_mask & ~cleaned[col].isnull()]
                
                if len(team_data) >= min_team_size and len(team_data[col].dropna()) > 0:
                    if is_categorical:
                        impute_value = team_data[col].mode()
                        if len(impute_value) > 0:
                            mode_value = impute_value.iloc[0]
                            # Handle categorical columns properly
                            if cleaned[col].dtype.name == 'category' and mode_value not in cleaned[col].cat.categories:
                                cleaned[col] = cleaned[col].cat.add_categories([mode_value])
                            cleaned.loc[team_missing_mask, col] = mode_value
                            imputation_stats['team_imputed'] += team_missing_mask.sum()
                    else:
                        median_value = team_data[col].median()
                        if not pd.isna(median_value):
                            cleaned.loc[team_missing_mask, col] = median_value
                            imputation_stats['team_imputed'] += team_missing_mask.sum()
    
    # Strategy 3: Buying org imputation for remaining missing values
    missing_mask = cleaned[col].isnull()  # Recalculate after team imputation
    
    if 'gravity_buying_organization_desc' in cleaned.columns:
        for org in cleaned['gravity_buying_organization_desc'].unique():
            if pd.isna(org):
                continue
            
            org_mask = cleaned['gravity_buying_organization_desc'] == org
            org_missing_mask = missing_mask & org_mask
            
            if org_missing_mask.sum() > 0:  # Org has missing values
                org_data = cleaned[org_mask & ~cleaned[col].isnull()]
                
                if len(org_data) >= min_org_size and len(org_data[col].dropna()) > 0:
                    if is_categorical:
                        impute_value = org_data[col].mode()
                        if len(impute_value) > 0:
                            mode_value = impute_value.iloc[0]
                            # Handle categorical columns properly
                            if cleaned[col].dtype.name == 'category' and mode_value not in cleaned[col].cat.categories:
                                cleaned[col] = cleaned[col].cat.add_categories([mode_value])
                            cleaned.loc[org_missing_mask, col] = mode_value
                            imputation_stats['org_imputed'] += org_missing_mask.sum()
                    else:
                        median_value = org_data[col].median()
                        if not pd.isna(median_value):
                            cleaned.loc[org_missing_mask, col] = median_value
                            imputation_stats['org_imputed'] += org_missing_mask.sum()
    
    # Strategy 4: Fallback for remaining values
    missing_mask = cleaned[col].isnull()  # Recalculate after org imputation
    remaining_missing = missing_mask.sum()
    
    if remaining_missing > 0:
        if is_categorical:
            # Handle boolean columns - use most frequent valid value
            if cleaned[col].dtype == 'bool':
                # For boolean columns, get the most frequent non-NaN value
                non_nan_values = cleaned[col].dropna()
                if len(non_nan_values) > 0:
                    global_mode = non_nan_values.mode()
                    if len(global_mode) > 0:
                        cleaned.loc[missing_mask, col] = global_mode.iloc[0]
                    else:
                        # If no clear mode, use the first valid value or False as fallback
                        cleaned.loc[missing_mask, col] = non_nan_values.iloc[0] if len(non_nan_values) > 0 else False
                else:
                    # No non-NaN values available, default to False
                    cleaned.loc[missing_mask, col] = False
            # Handle categorical columns properly by adding "Missing" to categories first
            elif cleaned[col].dtype.name == 'category':
                if 'Missing' not in cleaned[col].cat.categories:
                    cleaned[col] = cleaned[col].cat.add_categories(['Missing'])
                cleaned.loc[missing_mask, col] = 'Missing'
            else:
                # Object columns can be filled directly
                cleaned.loc[missing_mask, col] = 'Missing'
        else:
            # Use global median for numerical columns
            global_median = cleaned[col].median()
            if not pd.isna(global_median):
                cleaned.loc[missing_mask, col] = global_median
        
        imputation_stats['fallback_imputed'] = remaining_missing
    
    return imputation_stats


def _clean_target_data(df: pd.DataFrame, target_col: str, role: str, funnel=None) -> Tuple[pd.DataFrame, list]:
    """
    Clean data for a specific target column with comprehensive data quality checks.

    Performs multiple data cleaning operations:
    1. Removes records with missing target values
    2. Removes $0 director costs (data quality issue)
    3. Hierarchical categorical imputation (sub-team → team → org → "Missing")
    4. Drops columns with excessive missing data (>40% threshold)
    5. Identifies and returns numeric feature columns

    Args:
        df (pd.DataFrame): Input dataset with person and title data
        target_col (str): Target column name ('atl_efc_usd' or 'director_efc_usd')
        role (str): Model type ('Cast' or 'Director') for logging
        funnel: Optional TitleFunnel instance for tracking title dropouts

    Returns:
        Tuple[pd.DataFrame, list]: Cleaned dataframe and list of numeric feature names
    """
    # Remove rows without target values
    cleaned = df.dropna(subset=[target_col])
    
    # For director data, remove $0 director costs (data quality issue)
    if target_col == 'director_efc_usd':
        initial_count = len(cleaned)
        zero_cost_mask = cleaned[target_col] == 0
        if zero_cost_mask.sum() > 0:
            print(f"   Warning: Removing {zero_cost_mask.sum()} records with $0 director costs from {initial_count} total")
            cleaned = cleaned[~zero_cost_mask]
            print(f"   After $0 cost removal: {len(cleaned)} records remaining")
    
    # Handle categorical missing data using hierarchical imputation
    print(f"Handling categorical missing data for {role}...")
    
    # Minimum sample size thresholds for reliable imputation
    MIN_SUB_TEAM_SIZE = 5  # Minimum records per sub buying team
    MIN_TEAM_SIZE = 7  # Minimum records per buying team  
    MIN_ORG_SIZE = 10  # Minimum records per buying org
    
    for col in cleaned.columns:
        if cleaned[col].dtype == 'object' or cleaned[col].dtype.name == 'category':
            missing_count = cleaned[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(cleaned) * 100
                
                # RESTORED: Hierarchical imputation to maximize data availability for advanced features
                # This is crucial for director profile features that are sparse but valuable
                imputation_stats = _hierarchical_imputation(
                    cleaned, col, role, MIN_SUB_TEAM_SIZE, MIN_TEAM_SIZE, MIN_ORG_SIZE, is_categorical=True
                )
                
                # Report imputation strategy results
                total_imputed = sum(imputation_stats.values())
                print(f"  {col}: {missing_count} missing values ({missing_pct:.1f}%) imputed:")
                print(f"    Sub Team mode: {imputation_stats['sub_team_imputed']} values")
                print(f"    Team mode: {imputation_stats['team_imputed']} values") 
                print(f"    Org mode: {imputation_stats['org_imputed']} values")
                print(f"    'Missing': {imputation_stats['fallback_imputed']} values")
                
                if total_imputed != missing_count:
                    print(f"    WARNING: Imputation mismatch! Expected {missing_count}, got {total_imputed}")
    
    # Handle numerical missing data using hierarchical median imputation
    print(f"Handling numerical missing data for {role}...")

    # Columns to exclude from imputation (missing data is meaningful for intelligent routing)
    # Include both person-level names (previous_director_efc_*) and title-level names (mean_director_previous_*)
    no_impute_cols = [
        'previous_director_efc_usd',  # Person-level column
        'previous_director_efc_base',  # Person-level column
        'mean_director_previous_director_efc_usd',  # Title-level (aggregated)
        'max_director_previous_director_efc_usd',
        'min_director_previous_director_efc_usd',
        'std_director_previous_director_efc_usd',
        'mean_director_previous_director_efc_base',
        'max_director_previous_director_efc_base',
        'min_director_previous_director_efc_base',
        'std_director_previous_director_efc_base',
        'nos_filmography_strength_rescaled_max', # Person-level column
        'nos_filmography_strength_rescaled_country_origin', # Person-level column
        'director_experience', # Person-level column
        'max_box_office', # Person-level column
        'vscore' # person embedding column
    ]

    for col in cleaned.columns:
        if cleaned[col].dtype in [np.float64, np.float32, np.int64, np.int32] and cleaned[col].isnull().sum() > 0:
            # Skip imputation for fee history columns - missing data indicates no previous Netflix work
            if col in no_impute_cols:
                print(f"  {col}: Skipping imputation (missing data is meaningful for intelligent routing)")
                continue

            missing_count = cleaned[col].isnull().sum()
            missing_pct = missing_count / len(cleaned) * 100

            # RESTORED: Hierarchical imputation to maximize data availability for advanced numerical features
            # This is CRITICAL for director profile features (box office, experience, filmography scores)
            imputation_stats = _hierarchical_imputation(
                cleaned, col, role, MIN_SUB_TEAM_SIZE, MIN_TEAM_SIZE, MIN_ORG_SIZE, is_categorical=False
            )
            
            # Report imputation strategy results  
            total_imputed = sum(imputation_stats.values())
            print(f"  {col}: {missing_count} missing values ({missing_pct:.1f}%) imputed:")
            print(f"    Sub Team median: {imputation_stats['sub_team_imputed']} values")
            print(f"    Team median: {imputation_stats['team_imputed']} values")
            print(f"    Org median: {imputation_stats['org_imputed']} values")
            print(f"    Global median: {imputation_stats['fallback_imputed']} values")
            
            if total_imputed != missing_count:
                print(f"    WARNING: Imputation mismatch! Expected {missing_count}, got {total_imputed}")
    
    # Handle boolean missing data using hierarchical most frequent imputation
    print(f"Handling boolean missing data for {role}...")
    for col in cleaned.columns:
        if cleaned[col].dtype == 'bool' and cleaned[col].isnull().sum() > 0:
            missing_count = cleaned[col].isnull().sum()
            missing_pct = missing_count / len(cleaned) * 100
            
            # RESTORED: Hierarchical imputation for boolean data to maximize data availability  
            # Uses team-level knowledge to make better boolean feature predictions
            imputation_stats = _hierarchical_imputation(
                cleaned, col, role, MIN_SUB_TEAM_SIZE, MIN_TEAM_SIZE, MIN_ORG_SIZE, is_categorical=True
            )
            
            # Report imputation strategy results
            total_imputed = sum(imputation_stats.values())
            print(f"  {col}: {missing_count} missing values ({missing_pct:.1f}%) imputed:")
            print(f"    Sub Team most frequent: {imputation_stats['sub_team_imputed']} values")
            print(f"    Team most frequent: {imputation_stats['team_imputed']} values")
            print(f"    Org most frequent: {imputation_stats['org_imputed']} values")
            print(f"    Global most frequent: {imputation_stats['fallback_imputed']} values")
            
            if total_imputed != missing_count:
                print(f"    WARNING: Imputation mismatch! Expected {missing_count}, got {total_imputed}")
    
    # Identify columns to drop based on missing data threshold
    missing_summary = cleaned.isnull().sum().reset_index()
    missing_summary.columns = ['column', 'num_missing']

    # Use configurable missing data threshold
    drop_threshold = len(cleaned) * MISSING_DATA_THRESHOLD
    cols_to_drop = missing_summary[
        missing_summary['num_missing'] > drop_threshold
    ]['column'].tolist()

    # Don't drop fee history columns - missing data is meaningful for intelligent routing
    cols_to_drop = [col for col in cols_to_drop if col not in no_impute_cols]

    print(f'{role} columns to drop (>{MISSING_DATA_THRESHOLD*100:.0f}% missing): {cols_to_drop}')

    # Remove high-missing columns
    final_df = cleaned.drop(columns=cols_to_drop, errors='ignore')
    print(f'Total cleaned {role} records: {len(final_df)}')
    
    # Get numeric features excluding non-numeric identifiers
    numeric_features = final_df.select_dtypes(
        include=[np.number], exclude=['bool']
    ).columns.tolist()
    
    final_numeric_features = [
        col for col in numeric_features if col not in NON_NUMERIC_FEATURES
    ]
    
    print(f'{role} numeric features: {len(final_numeric_features)}')

    # Track in funnel if provided (only for Director)
    if funnel is not None and role == 'Director':
        funnel.add_stage(
            stage_name='05_Valid_Director_Target',
            current_titles=final_df['season_production_id'].unique(),
            dropout_reason='Missing director_efc_usd or $0 director cost (data quality issue)'
        )

    return final_df, final_numeric_features

def _apply_org_exclusions(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """
    Apply target-specific organization exclusions based on business rules.
    
    Different model types have different organizational data requirements:
    - Director models: Exclude Animation Series, Licensing, and UCAN Live Events  
      (insufficient/unreliable director fee data)
    - ATL models: Only exclude UCAN Live Events (keep Animation/Licensing data)
    - Other models: Keep all organizations
    
    Args:
        df (pd.DataFrame): Input dataset with gravity_buying_organization_desc column
        target_type (str): Model type ('director', 'atl', or other)
        
    Returns:
        pd.DataFrame: Filtered dataset with appropriate organizations excluded
    """
    if 'gravity_buying_organization_desc' not in df.columns:
        return df
    
    original_count = len(df)
    
    if target_type.lower() == 'director':
        # Exclude Animation Series and Licensing for Director models only, and UCAN Live Events
        exclude_orgs = ['Animation Series', 'Licensing','UCAN Live Events']
        df_filtered = df[~df['gravity_buying_organization_desc'].isin(exclude_orgs)]
        excluded_count = original_count - len(df_filtered)
        print(f"  {target_type} model: Excluded {excluded_count} records from Animation Series, Licensing, and UCAN Live Events")
    elif target_type.lower() == 'atl':
        # Exclude UCAN Live Events for ATL models
        exclude_orgs = ['UCAN Live Events']
        df_filtered = df[~df['gravity_buying_organization_desc'].isin(exclude_orgs)]
        excluded_count = original_count - len(df_filtered)
        print(f"  {target_type} model: Excluded {excluded_count} records from UCAN Live Events")
    else:
        # Keep all organizations for other models
        df_filtered = df.copy()
        print(f"  {target_type} model: Keeping all organizations")
    
    print(f"  {target_type} model: {len(df_filtered)} records remaining")
    return df_filtered

def clean_data(person_title_data: pd.DataFrame, funnel=None) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """Clean and prepare data for cast and director modeling."""
    # Filter for titles with sufficient embedding completeness - use different thresholds for each target
    print("Applying separate embedding thresholds:")
    print(f"  ATL model threshold: {EMBEDDING_THRESHOLD_ATL}%")
    print(f"  Director model threshold: {EMBEDDING_THRESHOLD_DIRECTOR}%")

    # Filter data separately for each target type
    atl_embeddings_cleaned = _filter_embedding_completeness(
        person_title_data.copy(), EMBEDDING_THRESHOLD_ATL, 'ATL', funnel=None
    )

    director_embeddings_cleaned = _filter_embedding_completeness(
        person_title_data.copy(), EMBEDDING_THRESHOLD_DIRECTOR, 'Director', funnel=funnel
    )

    # Apply organization exclusions separately for each target
    print("\nApplying organization exclusions:")
    atl_embeddings_cleaned = _apply_org_exclusions(atl_embeddings_cleaned, 'ATL')
    director_embeddings_cleaned = _apply_org_exclusions(director_embeddings_cleaned, 'Director')

    # Log after organization exclusions
    reporter_imports = safe_import('utils.sample_size_reporter', ['log_stage'])
    log_stage = reporter_imports['log_stage']

    log_stage('after_org_exclusions_atl', atl_embeddings_cleaned, 'ATL data after organization exclusions')
    log_stage('after_org_exclusions_director', director_embeddings_cleaned, 'Director data after organization exclusions')

    # Process cast and director data separately
    cast_df, cast_features = _clean_target_data(
        atl_embeddings_cleaned.copy(), 'atl_efc_usd', 'Cast', funnel=None
    )

    director_df, director_features = _clean_target_data(
        director_embeddings_cleaned.copy(), 'director_efc_usd', 'Director', funnel=funnel
    )

    # Log after target data cleaning
    log_stage('after_target_cleaning_cast', cast_df, 'Cast data after target cleaning (missing target removal + data quality)')
    log_stage('after_target_cleaning_director', director_df, 'Director data after target cleaning (missing target removal + data quality)')

    return cast_df, director_df, cast_features, director_features

def handle_vfx_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle VFX tier fields by applying logic for missing values and converting to categorical.
    
    Logic:
    - If vfx_efc_usd is missing, set it to 0
    - If vfx_efc_usd is 0, set vfx_efc_perc_gross_prod to 0
    - If vfx_efc_usd is 0 and vfx_tier is missing, set vfx_tier to 0
    - If vfx_efc_usd is 0 and vfx_touch_tier is missing, set vfx_touch_tier to 0
    - Convert vfx_tier and vfx_touch_tier to categorical
    
    Args:
        df: DataFrame with VFX-related columns
        
    Returns:
        DataFrame with properly handled VFX tier fields
    """
    print("Handling VFX tier fields...")
    
    df_copy = df.copy()
    
    # Convert missing vfx cost data to 0
    if 'vfx_efc_usd' in df_copy.columns:
        df_copy['vfx_efc_usd'] = np.where(
            df_copy['vfx_efc_usd'].isna(), 
            0.0, 
            df_copy['vfx_efc_usd']
        )
        
        # Set vfx percentage to 0 if vfx cost is 0
        if 'vfx_efc_perc_gross_prod' in df_copy.columns:
            df_copy['vfx_efc_perc_gross_prod'] = np.where(
                df_copy['vfx_efc_usd'] == 0.0,
                0.0,
                df_copy['vfx_efc_perc_gross_prod']
            )
    
    # Assign 0 to vfx tier fields if no vfx costs
    if 'vfx_tier' in df_copy.columns:
        df_copy['vfx_tier'] = np.where(
            (df_copy['vfx_efc_usd'] == 0.0) & (df_copy['vfx_tier'].isna()),
            0,
            df_copy['vfx_tier']
        )
        # Convert to categorical
        df_copy['vfx_tier'] = df_copy['vfx_tier'].astype('category')
        print(f"  vfx_tier unique values: {sorted(df_copy['vfx_tier'].dropna().unique())}")
    
    if 'vfx_touch_tier' in df_copy.columns:
        df_copy['vfx_touch_tier'] = np.where(
            (df_copy['vfx_efc_usd'] == 0.0) & (df_copy['vfx_touch_tier'].isna()),
            0,
            df_copy['vfx_touch_tier']
        )
        # Convert to categorical
        df_copy['vfx_touch_tier'] = df_copy['vfx_touch_tier'].astype('category')
        print(f"  vfx_touch_tier unique values: {sorted(df_copy['vfx_touch_tier'].dropna().unique())}")
    
    print(f"VFX tier handling complete. Non-null vfx_tier: {df_copy['vfx_tier'].notna().sum() if 'vfx_tier' in df_copy.columns else 0}")
    
    return df_copy

def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the dataset, specifically relevant_title_age.
    
    Args:
        df: DataFrame with production_start_year, launch_year, and os_production_type columns
        
    Returns:
        DataFrame with added relevant_title_age feature
    """
    print("Adding time-based features...")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Get current year
    current_year = dt.datetime.now().year
    
    # Ensure production_start_year is integer type (handle missing values)
    if 'production_start_year' in df_copy.columns:
        df_copy['production_start_year'] = pd.to_numeric(df_copy['production_start_year'], errors='coerce')
        
        # Handle missing production_start_year by using launch_year
        df_copy['production_start_year'] = np.where(
            df_copy['production_start_year'].isna(), 
            df_copy['launch_year'],
            df_copy['production_start_year']
        )
    
    # Calculate title ages
    df_copy['title_age_yrs'] = current_year - df_copy['launch_year']
    df_copy['title_production_age_yrs'] = current_year - df_copy['production_start_year']
    
    # Create relevant_title_age based on os_production_type
    # For non-FEATURE: use title_age_yrs (launch-based)
    # For FEATURE: use title_production_age_yrs (production-based)
    df_copy['relevant_title_age'] = np.where(
        df_copy['os_production_type'] != 'FEATURE', 
        df_copy['title_age_yrs'],
        np.where(
            df_copy['os_production_type'] == 'FEATURE', 
            df_copy['title_production_age_yrs'], 
            np.nan
        )
    )
    
    # Drop intermediate columns to keep data clean
    df_copy.drop(columns=['title_age_yrs', 'title_production_age_yrs'], inplace=True, errors='ignore')
    
    print(f"Added relevant_title_age feature. Non-null values: {df_copy['relevant_title_age'].notna().sum()}")
    
    return df_copy

def find_iqr_outliers(df: pd.DataFrame, model: str, features: list, threshold: float = IQR_THRESHOLD) -> pd.DataFrame:
    """Detect outliers using IQR method, grouped by buying organization."""
    target_features = features.copy()
    
    if model == 'cast':
        target_features.append('atl_efc_usd')
    elif model == 'director':
        target_features.append('director_efc_usd')
    else:
        raise ValueError("Model must be 'cast' or 'director'")
    
    print(f"Detecting outliers for {len(target_features)} features")
    
    # Initialize outlier column
    df_copy = df.copy()
    df_copy['iqr_outlier'] = False
    
    # Process each buying organization
    for org_name, group_idx in df_copy.groupby('gravity_buying_organization_desc').groups.items():
        group = df_copy.loc[group_idx]
        print(f"Processing {org_name}: {len(group)} records")
        
        for feature in target_features:
            if feature not in group.columns:
                continue
                
            Q1, Q3 = group[feature].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            
            if IQR == 0:
                mean, std = group[feature].mean(), group[feature].std()
                lower, upper = mean - threshold * std, mean + threshold * std
            else:
                lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            
            # Mark outliers for this feature in this org
            outliers_mask = (group[feature] < lower) | (group[feature] > upper)
            df_copy.loc[group_idx[outliers_mask], 'iqr_outlier'] = True
        
        outlier_count = df_copy.loc[group_idx, 'iqr_outlier'].sum()
        print(f"  Outliers flagged: {outlier_count}")
    
    total_outliers = df_copy['iqr_outlier'].sum()
    print(f"Total outliers flagged: {total_outliers}")
    
    # Return unique season_production_id entries only
    result = df_copy[['season_production_id', 'iqr_outlier']].drop_duplicates(subset='season_production_id')
    print(f"IQR outliers result: {len(result)} unique records")
    return result

RARE_CATEGORY_THRESHOLDS = {
    'genre_desc': 20,
    'primary_language': 50,
    'primary_country_shooting_location': 50,
    'gravity_country_of_origin': 50,
    'overall_deal_name': 10,
    'gravity_buying_team': 15,  # moved from 3 to 15 after running experiments, best result
    'gravity_sub_buying_team': 10,  # exploring this for director model
}

ADDITIONAL_OHE_COLS = [
    'gravity_deal_structure', 'os_production_type', 'gravity_buying_organization_desc',
    'ownership_structure', 'management_structure', 'nonfiction_series_buying_org',
    'returning_season', 'genre_documentary_standup', 
    'vfx_tier', 'vfx_touch_tier'
]

def encode_rare_categories(df: pd.DataFrame, one_hot_encode: bool = False) -> pd.DataFrame:
    """Group rare categories and optionally apply one-hot encoding."""
    print("Encoding rare categories...")
    
    df_encoded = df.copy()
    
    # Group rare categories more efficiently
    for col, threshold in RARE_CATEGORY_THRESHOLDS.items():
        if col in df_encoded.columns:
            value_counts = df_encoded[col].value_counts()
            rare_values = set(value_counts[value_counts < threshold].index)
            if rare_values:
                # Use .replace() instead of .map() to handle NaN values properly
                df_encoded[col] = df_encoded[col].replace(rare_values, 'Other')
                print(f"  {col}: {len(rare_values)} rare categories grouped")
    
    # One-hot encode if requested
    if one_hot_encode:
        ohe_cols = [
            col for col in list(RARE_CATEGORY_THRESHOLDS.keys()) + ADDITIONAL_OHE_COLS
            if col in df_encoded.columns
        ]
        print(f"One-hot encoding {len(ohe_cols)} columns")
        
        # Use sparse=False to avoid memory issues and dtype consistency
        df_encoded = pd.get_dummies(
            df_encoded, 
            columns=ohe_cols, 
            drop_first=False, 
            sparse=False,
            dtype='uint8'  # Use uint8 to save memory
        )
        
        # Remove duplicates more efficiently
        df_encoded = df_encoded.drop_duplicates()
    
    return df_encoded

def find_isolationforest_outliers(
    df: pd.DataFrame, 
    feature_cols: list, 
    contamination: float = ISOLATION_FOREST_CONTAMINATION, 
    random_state: int = 42
) -> pd.DataFrame:
    """Detect outliers using Isolation Forest algorithm."""
    from sklearn.ensemble import IsolationForest
    
    print('Starting Isolation Forest outlier detection...')
    
    # Clean data and prepare features
    df_clean = df.dropna(subset=feature_cols).copy()
    X = df_clean[feature_cols]
    
    print(f"Using {len(feature_cols)} features for {len(df_clean)} records")
    
    # Fit Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = iso_forest.fit_predict(X)
    
    # Convert predictions to boolean (True for outliers)
    df_clean['isolationforest_outlier'] = predictions == -1
    
    # Return only the outlier flags - keep original df intact
    outlier_count = df_clean['isolationforest_outlier'].sum()
    print(f"Isolation Forest outliers flagged: {outlier_count}")
    
    return df_clean[['season_production_id', 'isolationforest_outlier']]

def outlier_label(row: pd.Series) -> str:
    """Label outliers based on IQR and Isolation Forest detection methods."""
    if row['iqr_outlier'] and row['isolationforest_outlier']:
        return 'Both'
    elif row['iqr_outlier']:
        return 'IQR Only'
    elif row['isolationforest_outlier']:
        return 'Isolation Forest Only'
    else:
        return 'Neither'

def _impute_title_level_features(data: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """
    Impute title-level features that come from SQL LEFT JOINs and may have NaN values.
    
    Args:
        data: DataFrame containing the data
        target_type: 'ATL' or 'Director' for logging purposes
        
    Returns:
        DataFrame with imputed title-level features
    """
    print(f"\n🔧 Imputing title-level features for {target_type} dataset...")
    
    # Title-level features that may have NaNs from LEFT JOINs
    title_level_features = ['total_cast_count', 'num_shooting_locations', 'est_shooting_days']
    
    for col in title_level_features:
        if col in data.columns and data[col].isnull().sum() > 0:
            missing_count = data[col].isnull().sum()
            print(f"  Imputing {col}: {missing_count} missing values")
            
            # Use hierarchical imputation for numerical features
            imputation_stats = _hierarchical_imputation(
                data, col, target_type, 
                min_sub_team_size=3, min_team_size=5, min_org_size=10,
                is_categorical=False
            )
            
            # Report results
            total_imputed = sum(imputation_stats.values())
            if total_imputed > 0:
                print(f"    Sub-team: {imputation_stats['sub_team_imputed']}, "
                      f"Team: {imputation_stats['team_imputed']}, "
                      f"Org: {imputation_stats['org_imputed']}, "
                      f"Fallback: {imputation_stats['fallback_imputed']}")
    
    return data

def create_director_and_atl_title_dataset(
    director_data: pd.DataFrame,
    cast_data: pd.DataFrame,
    title_df: pd.DataFrame,
    base_comps_df: pd.DataFrame,
    director_profile_features: pd.DataFrame = None,
    funnel = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate Director-specific title-level dataset with director-focused features."""
    
    # Generate Director-specific embedding features from cleaned director data
    director_embeddings = generate_embeddings_features(director_data, 'Directors')

    # Generate ATL-specific embedding features from cleaned cast data
    cast_embeddings = generate_embeddings_features(cast_data, 'Cast')
    
    # Find similar seasons for ATL comparisons
    similar_cast = find_similar_seasons(
        cast_embeddings, measure='median_cast_embedding', role='cast', top_n=5
    )
    
    # Find similar seasons for Director comparisons
    similar_directors = find_similar_seasons(
        director_embeddings, measure='median_directors_embedding', role='director', top_n=5
    )
    # Find similar seasons for ATL comparisons
    similar_cast = find_similar_seasons(
        cast_embeddings, measure='median_cast_embedding', role='cast', top_n=5
    )

    # Get Director base comparisons
    try:
        if base_comps_df.empty:
            logging.warning("Base comparisons DataFrame is empty - director base median features will be unavailable")
            similar_base_director = pd.DataFrame()
        elif 'director_efc_usd' not in base_comps_df.columns:
            logging.warning("Column 'director_efc_usd' not found in base_comps_df - skipping base_median_director_efc_usd feature")
            similar_base_director = pd.DataFrame()
        else:
            similar_base_director = get_base_similar_seasons(base_comps_df, metric='director_efc_usd')
            logging.info(f"Generated {len(similar_base_director)} director base comparison records")
    except Exception as e:
        logging.error(f"Error generating director base comparisons: {e}")
        similar_base_director = pd.DataFrame()

    # Get ATL base comparisons
    try:
        if base_comps_df.empty:
            logging.warning("Base comparisons DataFrame is empty - base median features will be unavailable")
            similar_base_atl = pd.DataFrame()
        elif 'atl_efc_usd' not in base_comps_df.columns:
            logging.error("Column 'atl_efc_usd' not found in base_comps_df - cannot generate ATL base comparisons")
            similar_base_atl = pd.DataFrame()
        else:
            similar_base_atl = get_base_similar_seasons(base_comps_df, metric='atl_efc_usd')
            logging.info(f"Generated {len(similar_base_atl)} ATL base comparison records")
    except Exception as e:
        logging.error(f"Error generating ATL base comparisons: {e}")
        similar_base_atl = pd.DataFrame()
    
    # Calculate median costs from Director comparisons
    median_director_efc = calculate_median_comp_efc(
        similar_directors, title_df, metric='director_efc_usd', role='director'
    )
    
    # Calculate median costs from ATL comparisons
    median_cast_efc = calculate_median_comp_efc(
        similar_cast, title_df, metric='atl_efc_usd', role='cast'
    )

    # Calculate Director base median
    if not similar_base_director.empty:
        base_median_director_efc = calculate_base_median_efc(similar_base_director, metric='director_efc_usd')
    else:
        # Create a dummy DataFrame with NaN values for base_median_director_efc_usd
        base_median_director_efc = pd.DataFrame({
            'target_ptp_id': title_df['season_production_id'],
            'base_median_director_efc_usd': np.nan
        })
    
    # Calculate ATL base median
    base_median_atl_efc = calculate_base_median_efc(similar_base_atl, metric='atl_efc_usd')
    
    # Import log_stage function
    reporter_imports = safe_import('utils.sample_size_reporter', ['log_stage'])
    log_stage = reporter_imports['log_stage']
    
    # Build Director title features using inner joins for complete data
    director_title_features = title_df.copy()
    print(f"Starting Director dataset with title_df: {len(director_title_features)} titles")
    log_stage('director_title_merge_start', director_title_features, 'Starting Director title-level feature creation with base title data')
    
    # Build ATL title features using inner joins for complete data
    atl_title_features = title_df.copy()
    print(f"Starting ATL dataset with title_df: {len(atl_title_features)} titles")
    log_stage('atl_title_merge_start', atl_title_features, 'Starting ATL title-level feature creation with base title data')
    
    # Merge Director-specific features using inner joins for complete data
    director_title_features = _merge_with_logging(director_title_features, director_embeddings, 'director_embeddings', log_stage,
                                                 log_key='director_after_director_embeddings_merge',
                                                 log_description='Director dataset: After merging director_embeddings')
    
    # Merge ATL-specific features using inner joins for complete data
    atl_title_features = _merge_with_logging(atl_title_features, median_cast_efc, 'median_cast_efc', log_stage,
                                           log_key='atl_after_median_cast_efc_merge',
                                           log_description='ATL dataset: After merging median_cast_efc')
    atl_title_features = _merge_with_logging(atl_title_features, cast_embeddings, 'cast_embeddings', log_stage,
                                           log_key='atl_after_cast_embeddings_merge', 
                                           log_description='ATL dataset: After merging cast_embeddings')
    
    # Base median Director EFC merge
    director_title_features = _merge_with_logging(
        director_title_features, base_median_director_efc, 'base_median_director_efc', log_stage,
        left_on='season_production_id', right_on='target_ptp_id',
        drop_columns=['target_ptp_id'],
        log_key='director_after_base_median_director_efc_merge',
        log_description='Director dataset: After merging base_median_director_efc'
    )
    
    director_title_features = _merge_with_logging(
        director_title_features, median_director_efc, 'median_director_efc', log_stage,
        log_key='director_after_median_director_efc_merge',
        log_description='Director dataset: After merging median director EFC features'
    )

    # Base median ATL EFC merge
    atl_title_features = _merge_with_logging(
        atl_title_features, base_median_atl_efc, 'base_median_atl_efc', log_stage,
        left_on='season_production_id', right_on='target_ptp_id',
        drop_columns=['target_ptp_id'],
        log_key='atl_after_base_median_atl_efc_merge',
        log_description='ATL dataset: After merging base_median_atl_efc'
    )

    # Generate director fee comp features
    print("Generating director fee comp features...")
    director_fee_comps = generate_director_fee_comps_batch(
        director_data, title_df, 
        role='Director',
        fee_column='director_efc_usd',
        min_titles=1,           # Reduced from 2
        similarity_threshold=0.05,  # Reduced from 0.1
        top_n=15,              # Increased from 10
        weighting_strategies=['equal', 'experience'],
        enable_fallbacks=True,   # Enable fallback strategies
        use_cache=True
    )
    print(f"Generated director fee comps for {len(director_fee_comps)} titles")
    
    # Merge director fee comps
    director_title_features = _merge_with_logging(
        director_title_features, director_fee_comps, 'director_fee_comps',
        log_stage_func=None,  # Skip logging for this conditional merge
        print_warning_if_empty=True
    )

    # 🔧 MERGE ORDER FIX: Move director profile features merge to AFTER fee merges
    # This prevents subsequent inner joins from corrupting the profile features
    print("Merging director profile features (moved to safer position)...")
    director_title_features = _merge_with_logging(
        director_title_features, director_profile_features, 'director_profile_features',
        log_stage,
        log_key='director_after_profile_features_merge',
        log_description='Director dataset: After merging director profile features (FIXED position)',
        print_warning_if_empty=True,
        funnel=funnel,
        funnel_stage_name='07_After_Director_Profile_Features'
    )
    
    # Handle VFX tier fields
    print("Handling VFX tiers in Director title-level data...")
    director_title_features = handle_vfx_tiers(director_title_features)
    log_stage('director_after_vfx_handling', director_title_features, 'After handling VFX tiers in Director title-level data')

    print("Handling VFX tiers in ATL title-level data...")
    atl_title_features = handle_vfx_tiers(atl_title_features)
    log_stage('atl_after_vfx_handling', atl_title_features, 'After handling VFX tiers in ATL title-level data')
    
    # Impute title-level features that may have NaNs from SQL LEFT JOINs
    director_title_features = _impute_title_level_features(director_title_features, 'Director')
    atl_title_features = _impute_title_level_features(atl_title_features, 'ATL')
    
    # Add time-based features
    print("Adding time-based features to Director title-level data...")
    director_title_features = add_time_based_features(director_title_features)
    log_stage('director_final_title_data', director_title_features, 'Final Director title-level dataset after all processing')

    print("Adding time-based features to ATL title-level data...")
    atl_title_features = add_time_based_features(atl_title_features)
    log_stage('atl_final_title_data', atl_title_features, 'Final ATL title-level dataset after all processing')

    # Final funnel tracking
    if funnel:
        final_director_titles = director_title_features['season_production_id'].unique()
        funnel.add_stage(
            stage_name='08_Final_Director_Dataset',
            current_titles=final_director_titles,
            dropout_reason='N/A (successfully processed through all stages)'
        )

    return director_title_features, similar_directors, similar_base_director, atl_title_features, similar_cast, similar_base_atl


ISOLATION_FOREST_EXCLUDE_COLS = [
    'season_production_id', 'titlename', 'person_id', 'job_title_name', 
    'job_title_id', 'person_name', 'department_name', 'hitch_person_id', 
    'hitch_person_desc', 'embedding'
]

NUMERIC_DTYPES = ['int64', 'float64', 'bool', 'uint8', 'int32', 'float32']

def _process_outlier_detection(df: pd.DataFrame, model: str, numeric_features: list, funnel=None) -> pd.DataFrame:
    """
    Process comprehensive outlier detection using dual methodology approach.

    Implements a robust two-method outlier detection pipeline:
    1. IQR (Interquartile Range) method on original data
    2. Isolation Forest method on encoded categorical features
    3. Combines results and creates outlier labels ('IQR Only', 'Isolation Forest Only', 'Both', 'Neither')
    4. Excludes titles flagged by both methods as potential outliers
    5. Logs detailed analysis for transparency

    Args:
        df (pd.DataFrame): Input dataset for outlier detection
        model (str): Model type ('ATL' or 'Director') for logging
        numeric_features (list): List of numeric feature column names for IQR analysis
        funnel: Optional TitleFunnel instance for tracking title dropouts

    Returns:
        pd.DataFrame: Original dataset with outlier analysis completed (for logging/transparency)
    """
    print(f"Starting outlier detection for {model} with {len(df)} records")
    
    # IQR outlier detection on original data
    print("Step 1: Running IQR outlier detection...")
    iqr_outliers = find_iqr_outliers(df, model=model, features=numeric_features)
    print(f"IQR outlier detection complete: {len(iqr_outliers)} records processed")
    
    # Encode categorical features for Isolation Forest
    print("Step 2: Encoding categorical features for Isolation Forest...")
    encoded_df = encode_rare_categories(df, one_hot_encode=True)
    print(f"Categorical encoding complete: {len(encoded_df)} records, {len(encoded_df.columns)} columns")
    
    # Prepare features for Isolation Forest (excluding specified columns)
    print("Step 3: Preparing features for Isolation Forest...")
    feature_cols = [
        col for col in encoded_df.columns 
        if col not in ISOLATION_FOREST_EXCLUDE_COLS and encoded_df[col].dtype in NUMERIC_DTYPES
    ]
    print(f"Feature preparation complete: {len(feature_cols)} features selected for Isolation Forest")
    
    # Isolation Forest outlier detection
    print("Step 4: Running Isolation Forest outlier detection...")
    isolation_outliers = find_isolationforest_outliers(encoded_df, feature_cols)
    print(f"Isolation Forest detection complete: {len(isolation_outliers)} records processed")
    
    # Merge outlier results and create labels
    print("Step 5: Merging outlier results and creating labels...")
    print(f"Isolation outliers: {len(isolation_outliers)} records, unique IDs: {isolation_outliers['season_production_id'].nunique()}")
    print(f"IQR outliers: {len(iqr_outliers)} records, unique IDs: {iqr_outliers['season_production_id'].nunique()}")
    
    outliers_combined = isolation_outliers.merge(iqr_outliers, on='season_production_id', how='inner')
    print(f"After merge: {len(outliers_combined)} records, unique IDs: {outliers_combined['season_production_id'].nunique()}")
    
    # Check for duplicates and remove them
    if outliers_combined['season_production_id'].duplicated().any():
        print("WARNING: Found duplicate season_production_ids after merge!")
        outliers_combined = outliers_combined.drop_duplicates(subset='season_production_id')
        print(f"After removing duplicates: {len(outliers_combined)} records")
    
    outliers_combined['outlier_label'] = outliers_combined.apply(outlier_label, axis=1)
    print(f"Outlier labeling complete: {len(outliers_combined)} records with labels")
    
    # Log detailed outlier analysis using consolidated function
    reporter_imports = safe_import('utils.sample_size_reporter', ['log_outlier_analysis'])
    log_outlier_analysis = reporter_imports['log_outlier_analysis']
    
    label_counts = outliers_combined['outlier_label'].value_counts()
    initial_titles = df['season_production_id'].nunique()
    
    outlier_results = {
        'initial_titles': initial_titles,
        'iqr_only': label_counts.get('IQR Only', 0),
        'isolation_forest_only': label_counts.get('Isolation Forest Only', 0),
        'both_methods': label_counts.get('Both', 0),
        'neither_method': label_counts.get('Neither', 0),
        'excluded_titles': label_counts.get('Both', 0),  # Only excluding 'Both'
        'retained_titles': initial_titles - label_counts.get('Both', 0)
    }
    
    log_outlier_analysis(model, outlier_results)
    
    print(f"Outlier breakdown - IQR Only: {outlier_results['iqr_only']}, "
          f"Isolation Forest Only: {outlier_results['isolation_forest_only']}, "
          f"Both: {outlier_results['both_methods']}, "
          f"Neither: {outlier_results['neither_method']}")
    
    # Merge outlier results to original dataframe and filter non-outliers
    print("Step 6: Merging results back to original dataframe...")
    print(f"Original df: {len(df)} records, {len(df.columns)} columns")
    print(f"Outliers combined: {len(outliers_combined)} records, {len(outliers_combined.columns)} columns")
    df_with_outliers = df.merge(outliers_combined, on='season_production_id', how='left')
    print(f"Merge complete: {len(df_with_outliers)} records, {len(df_with_outliers.columns)} columns")
    
    print("Step 7: Filtering outliers (excluding only titles flagged by BOTH methods)...")
    result = df_with_outliers[df_with_outliers['outlier_label'] != 'Both']
    print(f"Outlier detection complete: {len(result)} records remaining (excluded 'Both' outliers), {len(result.columns)} columns")

    # Track in funnel if provided (only for Director)
    if funnel is not None and model.lower() == 'director':
        funnel.add_stage(
            stage_name='06_Non_Outlier_Directors',
            current_titles=result['season_production_id'].unique(),
            dropout_reason='Flagged as outlier by both IQR and Isolation Forest methods'
        )

    return result

def main():
    """Main function to run feature engineering pipeline with command line arguments."""
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument('--force-all', action='store_true',
                        help='Force regenerate both SQL data and processed data (overwrite all files)')
    parser.add_argument('--process-only', action='store_true',
                        help='Force regenerate only processed data (keep existing raw SQL data)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear director similarity and fee comp caches then exit')
    
    args = parser.parse_args()
    
    # Handle cache clearing first
    if args.clear_cache:
        embeddings_imports = safe_import('embeddings_features', ['clear_all_director_caches'])
        clear_all_director_caches = embeddings_imports['clear_all_director_caches']
        print("Clearing all director caches...")
        clear_all_director_caches()
        print("Cache clearing complete.")
        return

    # Validate arguments
    if args.force_all and args.process_only:
        print("Error: Cannot use both --force-all and --process-only flags together")
        return
    
    # Determine what to regenerate
    force_sql = args.force_all
    force_processed = args.force_all or args.process_only
    
    if args.force_all:
        print("="*60)
        print("REGENERATING ALL DATA (--force-all flag used)")
        print("This will regenerate both SQL data and processed data")
        print("="*60)
    elif args.process_only:
        print("="*60)
        print("REGENERATING PROCESSED DATA ONLY (--process-only flag used)")
        print("This will keep existing raw SQL data and regenerate processed data")
        print("="*60)
    else:
        print("="*60)
        print("RUNNING FEATURE ENGINEERING PIPELINE")
        print("Using existing data where available")
        print("="*60)
    
    # Check if processed data exists and warn if not forcing regeneration
    processed_exists = (
        os.path.exists(os.path.join(ANALYSIS_DIR, 'data/processed_data/title_data.csv')) and
        os.path.exists(os.path.join(ANALYSIS_DIR, 'data/processed_data/cast_person_data.csv')) and
        os.path.exists(os.path.join(ANALYSIS_DIR, 'data/processed_data/director_person_data.csv'))
    )
    
    if processed_exists and not force_processed:
        print("Note: Processed data files already exist. To regenerate processed data, use --process-only or --force-all")
        print("This run will only regenerate if raw SQL data is missing.\n")
    
    # Load all datasets
    person_data, title_df, base_comps_df, person_scores_df = load_data(force_sql)

    # Check if we should skip processing if files already exist
    if processed_exists and not force_processed:
        print("Processed data files exist and no regeneration flags used. Skipping feature engineering.")
        print("Use --force-all to regenerate everything, or --process-only to regenerate only processed data.")
        return

    # ============================================================
    # INITIALIZE FUNNEL TRACKING
    # ============================================================
    funnel = None
    if ENABLE_FUNNEL_TRACKING:
        print("\n" + "="*80)
        print("INITIALIZING TITLE FUNNEL TRACKING")
        print("="*80)

        # Import TitleFunnel class
        try:
            from utils.title_funnel_tracker import TitleFunnel
        except ImportError:
            try:
                from features.utils.title_funnel_tracker import TitleFunnel
            except ImportError:
                from .utils.title_funnel_tracker import TitleFunnel

        # Apply scope filters to title_df
        title_df = apply_scope_filters(title_df, TITLE_SCOPE_CONFIG)

        # Initialize funnel with scoped titles
        funnel = TitleFunnel(
            initial_titles=title_df['season_production_id'].unique(),
            scope_description=TITLE_SCOPE_CONFIG['description']
        )
        print(f"✓ Funnel initialized with {len(funnel.initial_titles):,} titles in scope\n")

    # Merge person and title data
    person_title_merged = person_data.merge(title_df, on='season_production_id', how='inner')
    print(f'Combined dataset: {len(person_title_merged)} records')

    # Import log_stage using consolidated function
    reporter_imports = safe_import('utils.sample_size_reporter', ['log_stage'])
    log_stage = reporter_imports['log_stage']
    log_stage('after_title_merge', person_title_merged, 'After merging person and title data')

    # Track: After person-title merge
    if funnel:
        funnel.add_stage(
            stage_name='02_Has_Person_Records',
            current_titles=person_title_merged['season_production_id'].unique(),
            dropout_reason='No person records found for title'
        )

    # Track: Early director filter (before cleaning) to understand how many titles have directors
    if funnel:
        director_only = person_title_merged[person_title_merged['department_name'] == 'Directors']
        funnel.add_stage(
            stage_name='03_Has_Director_Records',
            current_titles=director_only['season_production_id'].unique(),
            dropout_reason='No director records in department_name="Directors"'
        )

    # Handle VFX tier fields BEFORE clean_data to preserve VFX columns with high missing rates
    print("\nHandling VFX tiers for combined dataset...")
    person_title_merged = handle_vfx_tiers(person_title_merged)

    # Generate director profile features BEFORE clean_data to preserve director feature columns
    print("\nGenerating director profile features from raw data...")
    director_profile_features_raw = generate_director_profile_features(person_title_merged)
    print(f"Generated director profile features for {len(director_profile_features_raw)} titles")
    
    # Note: Organization exclusions now handled separately for each target in clean_data()
    # This allows ATL models to keep Animation Series and Licensing data while Director models exclude it
    
    # Generate missing features report BEFORE cleaning data
    print("\nGenerating missing features report...")
    missing_report_path = generate_missing_features_report(person_title_merged)
    print(f"Missing features report generated: {missing_report_path}")
    
    # Clean data for cast and director modeling
    cast_cleaned, director_cleaned, cast_features, director_features = clean_data(person_title_merged, funnel=funnel)
    print(f"After cleaning - Cast: {len(cast_cleaned)} records, Director: {len(director_cleaned)} records")

    print(f"Cast cleaned columns: {len(cast_cleaned.columns)}")
    print(f"Director cleaned columns: {len(director_cleaned.columns)}")
    
    # Add time-based features to both datasets
    print("\nAdding time-based features to cast dataset...")
    cast_cleaned = add_time_based_features(cast_cleaned)
    
    print("\nAdding time-based features to director dataset...")
    director_cleaned = add_time_based_features(director_cleaned)
    
    # Add relevant_title_age to numeric features if it was created
    if 'relevant_title_age' in cast_cleaned.columns:
        cast_features.append('relevant_title_age')
    if 'relevant_title_age' in director_cleaned.columns:
        director_features.append('relevant_title_age')
    
    # Log before outlier detection (after all preprocessing is complete)
    log_stage('cast_before_outliers', cast_cleaned, 'Cast data before outlier detection')
    log_stage('director_before_outliers', director_cleaned, 'Director data before outlier detection')
    
    # Process outlier detection and removal for cast
    print("Processing cast outlier detection...")
    cast_final = _process_outlier_detection(cast_cleaned, 'cast', cast_features)
    print(f"Cast processing complete: {len(cast_final)} records, {len(cast_final.columns)} columns")
    print(f"Data reduction - Cast: {len(cast_cleaned)} -> {len(cast_final)} ({len(cast_cleaned) - len(cast_final)} removed)")
    
    log_stage('cast_after_outliers', cast_final, 'Cast data after outlier detection')
    
    # Process outlier detection and removal for director
    print("Processing director outlier detection...")
    director_final = _process_outlier_detection(director_cleaned, 'director', director_features, funnel=funnel)
    print(f"Director processing complete: {len(director_final)} records, {len(director_final.columns)} columns")
    print(f"Data reduction - Director: {len(director_cleaned)} -> {len(director_final)} ({len(director_cleaned) - len(director_final)} removed)")

    log_stage('director_after_outliers', director_final, 'Director data after outlier detection')
    
    # Save cleaned person-level data
    # Ensure processed_data directory exists
    processed_data_dir = os.path.join(ANALYSIS_DIR, 'data/processed_data')
    os.makedirs(processed_data_dir, exist_ok=True)
    
    print("Saving cast person-level data...")
    print(f"Cast final shape: {cast_final.shape}")
    print(f"Cast final columns: {list(cast_final.columns)}")
    cast_final.to_csv(os.path.join(processed_data_dir, 'cast_person_data.csv'), index=False)
    print("Cast person data saved successfully")
    
    print("Saving director person-level data...")
    print(f"Director final shape: {director_final.shape}")
    print(f"Director final columns: {list(director_final.columns)}")
    director_final.to_csv(os.path.join(processed_data_dir, 'director_person_data.csv'), index=False)
    print("Director person data saved successfully")
    
    print("Saved person-level data")
    

    # Create separate title-level datasets for improved sample sizes
    # print("Creating separate ATL and Director title-level datasets...")
    
    # Create model-specific datasets  
    print("\n" + "="*60)
    print("CREATING SEPARATE DIRECTOR AND ATL TITLE DATASETS")
    print("="*60)
    director_title_data, director_comps, director_base_comps, atl_title_data, cast_comps, atl_base_comps = create_director_and_atl_title_dataset(
        director_final, cast_final, title_df, base_comps_df, director_profile_features_raw, funnel=funnel
    )
    print(f"Director and ATL title-level features complete: {len(director_title_data)} director records and {len(atl_title_data)} ATL records")
    

    # Import save function for separate datasets
    try:
        from utils.feature_reporter import save_separate_datasets
    except ImportError:
        # Fallback for when run as script
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        from feature_reporter import save_separate_datasets
    
    # Save separate datasets
    save_separate_datasets(atl_title_data, director_title_data,
                          cast_comps, director_comps, atl_base_comps, director_base_comps)
    
    # Generate and save sample size report using consolidated function
    reporter_imports = safe_import('utils.sample_size_reporter', ['save_report', 'print_summary'])
    save_report = reporter_imports['save_report']
    print_summary = reporter_imports['print_summary']
    print_summary()
    report_path = save_report('data/processed_data')
    print(f"Sample size report generated: {report_path}")


    # ============================================================
    # GENERATE FUNNEL TRACKING REPORT
    # ============================================================
    if funnel:
        print("\n" + "="*80)
        print("TITLE FUNNEL TRACKING COMPLETE")
        print("="*80)

        # Print funnel to console
        funnel.print_funnel()

        # Save detailed reports
        funnel_output_dir = os.path.join(ANALYSIS_DIR, 'data/reports/funnel_analysis')
        funnel.save_report(output_dir=funnel_output_dir)

    print(f'Processing complete. ATL dataset: {len(atl_title_data)} records, Director dataset: {len(director_title_data)} records')
 

if __name__ == "__main__":
    main()
