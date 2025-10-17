import pandas as pd
import numpy as np


### I didn't implement any of this - no time
def generate_location_hub(data):
    """
    Generate location hub features from the data.
    """
    # Example feature generation
    data['location_hub'] = data['country'].apply(lambda x: x.split(',')[0])
    
    # Add more location-related features as needed
    return data


def generate_production_features(data):
    """
    Generate production features from the data.
    """
    # Example feature generation
    data['production_budget'] = data['budget'] * 1.2
    data['production_year'] = pd.to_datetime(data['release_date']).dt.year
    data['production_country'] = data['country'].apply(lambda x: x.split(',')[0])

    # Add more production-related features as needed
    return data


def generate_director_profile_features(person_data):
    """
    Generate director profile features aggregated at the title level.

    Takes person-level data and aggregates director-specific features
    (filmography strength, box office, experience, previous fees)
    from person-level to title-level.

    Args:
        person_data: DataFrame with person-level data including director features

    Returns:
        DataFrame with season_production_id and aggregated director profile features
    """
    print("Generating director profile features...")

    # Filter to directors only
    director_data = person_data[person_data['department_name'] == 'Directors'].copy()

    if director_data.empty:
        print("No director data found")
        return pd.DataFrame()

    print(f"Processing {len(director_data)} director records across {director_data['season_production_id'].nunique()} titles")

    # Define the director feature columns to aggregate
    director_feature_cols = [
        'nos_filmography_strength_rescaled_max',
        'max_box_office',
        'nos_filmography_strength_rescaled_country_origin',
        'director_experience',
        'previous_director_efc_usd',
        'previous_director_efc_base'
    ]

    # Check which columns actually exist in the data
    available_cols = [col for col in director_feature_cols if col in director_data.columns]
    missing_cols = [col for col in director_feature_cols if col not in director_data.columns]

    if missing_cols:
        print(f"Warning: Missing director feature columns: {missing_cols}")

    if not available_cols:
        print("Error: No director feature columns found in data")
        return pd.DataFrame()

    print(f"Aggregating {len(available_cols)} director features: {available_cols}")

    # Group by title and calculate aggregations
    title_director_features = []

    for season_id in director_data['season_production_id'].unique():
        title_directors = director_data[director_data['season_production_id'] == season_id]

        # Start with the basic info
        feature_dict = {'season_production_id': season_id}

        # Add director count
        feature_dict['director_count'] = len(title_directors)

        # Aggregate each available feature column
        for col in available_cols:
            # Get non-null values for this feature
            values = title_directors[col].dropna()

            if len(values) > 0:
                # Calculate aggregations
                feature_dict[f'mean_director_{col}'] = values.mean()
                feature_dict[f'max_director_{col}'] = values.max()
                feature_dict[f'min_director_{col}'] = values.min()
                feature_dict[f'std_director_{col}'] = values.std() if len(values) > 1 else 0.0
            else:
                # No valid values for this feature
                feature_dict[f'mean_director_{col}'] = np.nan
                feature_dict[f'max_director_{col}'] = np.nan
                feature_dict[f'min_director_{col}'] = np.nan
                feature_dict[f'std_director_{col}'] = np.nan

        title_director_features.append(feature_dict)

    # Convert to DataFrame
    result_df = pd.DataFrame(title_director_features)

    print(f"Generated director profile features for {len(result_df)} titles")
    print(f"Feature columns created: {len(result_df.columns) - 1}")  # -1 for season_production_id

    return result_df
