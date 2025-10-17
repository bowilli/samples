import logging
import numpy as np
import pandas as pd

def get_base_similar_seasons(comps_df, metric='atl_efc_usd', top_n=5):
    """
    Generate comp set size for base comps, max is 10.
    
    Args:
        comps_df: DataFrame with comparison data
        metric: The metric to use ('atl_efc_usd' or 'director_efc_usd')
        top_n: Number of top similar seasons to return
        
    Returns:
        DataFrame with similar seasons or empty DataFrame if error/no data
    """
    
    # Input validation
    if comps_df is None or comps_df.empty:
        logging.warning(f"Empty or None DataFrame provided to get_base_similar_seasons for metric '{metric}'")
        return pd.DataFrame()
    
    required_cols = ['target_ptp_id', 'comp_season_production_id', 'similarity_score', 'rank', metric]
    missing_cols = [col for col in required_cols if col not in comps_df.columns]
    if missing_cols:
        logging.error(f"Missing required columns in base comparisons DataFrame: {missing_cols}")
        return pd.DataFrame()
    
    # First remove all rows without our target metric
    comps_df_cleaned = comps_df.dropna(subset=[metric])
    
    # CRITICAL FIX: Remove self-comparisons to prevent data leakage
    # When rank = -1, the title is comparing to itself - exclude these records
    initial_count = len(comps_df_cleaned)
    comps_df_cleaned = comps_df_cleaned[comps_df_cleaned['rank'] != -1]
    final_count = len(comps_df_cleaned)
    
    if initial_count != final_count:
        logging.info(f"Excluded {initial_count - final_count} self-comparison records from {metric} base comparisons")
    
    # Check if we have any valid comparisons left
    if comps_df_cleaned.empty:
        logging.warning(f"No valid comparison records remaining for metric '{metric}' after cleaning")
        return pd.DataFrame()
    
    # Sort dataframe by target ptp id and similarity score in descending order
    df_sorted = comps_df_cleaned.sort_values(by=['target_ptp_id','similarity_score'], ascending=[True,False])
    
    # Group by target ptp id and take top n
    top_n_df = df_sorted.groupby('target_ptp_id').head(top_n).reset_index(drop=True)
        
    # Only want to return target_ptp_id, comp and sim score and the metric
    relevant_fields = ['target_ptp_id','comp_season_production_id','similarity_score', metric]
    
    similar_base_seasons_df = top_n_df[relevant_fields]
    
    return similar_base_seasons_df
    

def calculate_base_median_efc(similar_base_seasons_df, metric='atl_efc_usd'):
    """
    Calculate the median EFC for similar seasons for base comps.
    
    Args:
        similar_base_seasons_df: DataFrame with similar seasons
        metric: The metric to calculate median for ('atl_efc_usd' or 'director_efc_usd')
    
    Returns:
        DataFrame with target_ptp_id and base_median_{metric}
    """
    
    # Handle empty DataFrame case
    if similar_base_seasons_df.empty:
        logging.warning(f"Empty DataFrame provided to calculate_base_median_efc for metric '{metric}'")
        output_col = f'base_median_{metric}'
        return pd.DataFrame(columns=['target_ptp_id', output_col])
    
    # Group by 'target_ptp_id' and calculate the median of the metric
    base_median_efc = similar_base_seasons_df.groupby('target_ptp_id')[metric].median().reset_index()
    
    # Create output column name based on metric
    output_col = f'base_median_{metric}'
    
    # Rename the column to reflect that it's a median
    base_median_efc.rename(columns={metric: output_col}, inplace=True)
    
    # Subset to columns of interest
    base_median_efc_df = base_median_efc[['target_ptp_id', output_col]]
    
    return base_median_efc_df


# Backward compatibility wrapper functions
def calculate_base_median_atl_efc(similar_base_seasons_df):
    """Calculate the median ATL EFC for similar seasons for base comps - backward compatibility."""
    return calculate_base_median_efc(similar_base_seasons_df, metric='atl_efc_usd')


def calculate_base_median_director_efc(similar_base_seasons_df):
    """Calculate the median director EFC for similar seasons for base comps."""
    return calculate_base_median_efc(similar_base_seasons_df, metric='director_efc_usd')
