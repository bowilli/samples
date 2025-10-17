import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import ast  # Import ast to safely evaluate strings of lists

def generate_embeddings_features(person_data, role):
    """Generate summary embeddings for each season production."""
    season_embeddings = []
    for season_id in person_data['season_production_id'].unique():
        season_role_data = person_data[(person_data['season_production_id'] == season_id) & (person_data['department_name'] == role)]
        
        # Convert string embeddings to numpy arrays 
        embeddings = season_role_data['embedding'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))
        
        if embeddings.empty:
            season_embeddings.append({
                'season_production_id': season_id,
                # Full vectors
                f'mean_{role.lower()}_embedding': np.nan,
                f'max_{role.lower()}_embedding': np.nan,
                f'median_{role.lower()}_embedding': np.nan,
                f'std_{role.lower()}_embedding': np.nan,
                # Scalar features
                f'mean_{role.lower()}_embedding_scalar': np.nan,
                f'median_{role.lower()}_embedding_scalar': np.nan,
                f'max_{role.lower()}_embedding_scalar': np.nan,
            })
           
                    
        else:
            # Calculate aggregated embedding vectors across all cast/directors
            stacked_embeddings = np.stack(embeddings.tolist())
            mean_embedding_vector = np.mean(stacked_embeddings, axis=0)
            max_embedding_vector = np.max(stacked_embeddings, axis=0)
            median_embedding_vector = np.median(stacked_embeddings, axis=0)
            std_embedding_vector = np.std(stacked_embeddings, axis=0)
            
            season_embeddings.append({
                'season_production_id': season_id,
                # Keep full vectors for similarity calculations (used by find_similar_seasons)
                f'mean_{role.lower()}_embedding': mean_embedding_vector,
                f'max_{role.lower()}_embedding': max_embedding_vector,
                f'median_{role.lower()}_embedding': median_embedding_vector,
                f'std_{role.lower()}_embedding': std_embedding_vector,
                # Add scalar features for direct use in models
                f'mean_{role.lower()}_embedding_scalar': np.mean(mean_embedding_vector),
                f'median_{role.lower()}_embedding_scalar': np.median(median_embedding_vector),
                f'max_{role.lower()}_embedding_scalar': np.max(max_embedding_vector),
            })
    return pd.DataFrame(season_embeddings)

def find_similar_seasons(embeddings_df, measure='mean_cast_embedding',role='cast', top_n=5):
    """Find the most similar season productions based on a specified embedding measure."""
    # Filter out rows where the embedding is np.nan
    valid_embeddings_df = embeddings_df.dropna(subset=[measure])
    
    # Convert embeddings to numpy arrays
    embeddings_list = valid_embeddings_df[measure].apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)
    
    # Stack embeddings into a matrix
    embeddings = np.stack(embeddings_list.values)
    
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    similar_seasons = []
    for idx, season_id in enumerate(valid_embeddings_df['season_production_id']):
        # Get indices of the most similar seasons, excluding the season itself
        similar_indices = np.argsort(-similarity_matrix[idx])[1:top_n+1]
        similar_ids = valid_embeddings_df.iloc[similar_indices]['season_production_id'].tolist()
        similar_seasons.append({
            'season_production_id': season_id,
            f'{role}_comps': similar_ids
        })
    
    return pd.DataFrame(similar_seasons)

def calculate_median_comp_efc(similar_seasons_df, title_df, metric, role='cast'):
    """Calculate the median metric (e.g. atl_efc_usd or director_ef_usd) for similar season productions."""
    median_efc = []
    for _, row in similar_seasons_df.iterrows():
        season_id = row['season_production_id']
        comps = row[f'{role}_comps']
        comp_efc_values = title_df[title_df['season_production_id'].isin(comps)][metric]
        median = comp_efc_values.mean()
        median_efc.append({
            'season_production_id': season_id,
            f'median_{metric}': median
        })
    return pd.DataFrame(median_efc)
# embeddings_df = generate_embeddings_features(cast_data)
# similar_seasons = find_similar_seasons(embeddings_df, measure='mean_cast_embedding', top_n=2)
# mean_atl_efc = calculate_mean_atl_efc(similar_seasons, atl_efc_data)

# print("Similar Seasons:", similar_seasons)
# print("Mean ATL EFC:", mean_atl_efc)

def find_similar_season_with_filters(embeddings_df, measure='mean_cast_embedding', role='cast', top_n=5, filters=None):
    """Find similar seasons with optional filters."""
    if filters is not None:
        for key, value in filters.items():
            embeddings_df = embeddings_df[embeddings_df[key] == value]
    
    return find_similar_seasons(embeddings_df, measure=measure, role=role, top_n=top_n)

def find_similar_directors(person_data, title_data, target_title_id=None, 
                          role='Director', top_n=10, min_titles=2, 
                          similarity_threshold=0.1, exclude_target_directors=True):
    """
    Find similar directors for a target title using only historical data.
    
    Args:
        person_data: DataFrame with [person_id, department_name, embedding, season_production_id]
        title_data: DataFrame with [season_production_id, launch_date, preproduction_date]
        target_title_id: Title to generate features for (uses its preproduction_date as cutoff)
        role: Role to filter for (default 'Director')
        top_n: Maximum number of similar directors to return per director
        min_titles: Minimum historical titles a director needs to be included
        similarity_threshold: Minimum cosine similarity to consider
        exclude_target_directors: Whether to exclude directors working on target title
    
    Returns:
        DataFrame with director similarity relationships
    """
    if target_title_id is None:
        raise ValueError("target_title_id is required")
    
    # Get temporal cutoff from target title's preproduction date
    target_title_data = title_data[title_data['season_production_id'] == target_title_id]
    if target_title_data.empty:
        raise ValueError(f"Target title {target_title_id} not found in title_data")
    
    cutoff_date = target_title_data['pre_prod_start_date'].iloc[0]
    
    # Filter to historical titles only (launched before target's preproduction)
    historical_titles = title_data[title_data['launch_date'] < cutoff_date]['season_production_id']
    
    if len(historical_titles) == 0:
        print(f"No historical titles found before {cutoff_date}")
        return pd.DataFrame()
    
    # Filter person_data to historical titles and specified role
    # Handle the department name mapping (Director -> Directors)
    if role == 'Director':
        department_filter = (person_data['department_name'] == 'Directors')
    else:
        department_filter = (person_data['department_name'] == role)
    
    historical_person_data = person_data[
        (person_data['season_production_id'].isin(historical_titles)) & 
        department_filter
    ].copy()
    
    if historical_person_data.empty:
        print(f"No historical {role} data found for target {target_title_id}")
        print(f"  Historical titles available: {len(historical_titles)}")
        print(f"  Total {role} records in person_data: {len(person_data[department_filter])}")
        return pd.DataFrame()
    
    # Aggregate embeddings per person (not per title) with better error handling
    def safe_embedding_aggregation(embeddings_series):
        """Safely aggregate embeddings, filtering out invalid ones."""
        valid_embeddings = []
        for emb in embeddings_series:
            try:
                if isinstance(emb, str):
                    parsed_emb = ast.literal_eval(emb)
                else:
                    parsed_emb = emb

                # Convert to numpy array and check dimension
                emb_array = np.array(parsed_emb)
                if emb_array.ndim == 1 and len(emb_array) > 1:  # Valid embedding vector
                    valid_embeddings.append(emb_array)
                else:
                    print(f"Skipping invalid embedding: shape={emb_array.shape}, value={emb_array}")
            except Exception as e:
                print(f"Error parsing embedding: {e}")

        if valid_embeddings:
            return np.mean(valid_embeddings, axis=0)
        else:
            return None

    director_profiles = historical_person_data.groupby('person_id').agg({
        'embedding': safe_embedding_aggregation,
        'season_production_id': lambda x: list(x.unique()),  # Track title history
    }).copy()

    # Filter out directors with no valid embeddings
    valid_embeddings_mask = director_profiles['embedding'].apply(lambda x: x is not None)
    director_profiles = director_profiles[valid_embeddings_mask]

    # Add title count and director names
    director_profiles['title_count'] = director_profiles['season_production_id'].apply(len)

    # Create name mapping for directors
    name_mapping = historical_person_data.groupby('person_id')['person_name'].first().to_dict()
    director_profiles['director_name'] = director_profiles.index.map(name_mapping)
    
    # Quality filter: Only directors with sufficient historical work
    director_profiles = director_profiles[director_profiles['title_count'] >= min_titles]
    
    if director_profiles.empty:
        print(f"No directors found with >= {min_titles} historical titles")
        return pd.DataFrame()
    
    # Exclude directors working on the target title to prevent self-comparison
    if exclude_target_directors:
        # Handle department name mapping for target directors exclusion
        if role == 'Director':
            target_department_filter = (person_data['department_name'] == 'Directors')
        else:
            target_department_filter = (person_data['department_name'] == role)
            
        target_directors = person_data[
            (person_data['season_production_id'] == target_title_id) &
            target_department_filter
        ]['person_id'].unique()
        
        initial_count = len(director_profiles)
        director_profiles = director_profiles[~director_profiles.index.isin(target_directors)]
        excluded_count = initial_count - len(director_profiles)
        
        if excluded_count > 0:
            print(f"Excluded {excluded_count} target directors from similarity pool")
    
    if director_profiles.empty:
        print("No directors remaining after exclusions")
        return pd.DataFrame()
    
    # Get target directors (the ones we want to find similar directors FOR)
    if role == 'Director':
        target_department_filter = (person_data['department_name'] == 'Directors')
    else:
        target_department_filter = (person_data['department_name'] == role)

    target_directors_data = person_data[
        (person_data['season_production_id'] == target_title_id) &
        target_department_filter &
        (person_data['embedding'].notna())
    ]

    if target_directors_data.empty:
        #print(f"No target directors with embeddings found for title {target_title_id}")
        return pd.DataFrame()

    #print(f"Found {len(target_directors_data)} target directors with embeddings")

    # Process target director embeddings
    target_director_profiles = []
    for _, target_dir in target_directors_data.iterrows():
        try:
            if isinstance(target_dir['embedding'], str):
                target_embedding = ast.literal_eval(target_dir['embedding'])
            else:
                target_embedding = target_dir['embedding']

            target_emb_array = np.array(target_embedding)
            if target_emb_array.ndim == 1 and len(target_emb_array) > 1:
                target_director_profiles.append({
                    'person_id': target_dir['person_id'],
                    'person_name': target_dir['person_name'],
                    'embedding': target_emb_array
                })
            else:
                print(f"Skipping target director {target_dir['person_name']}: invalid embedding shape {target_emb_array.shape}")
        except Exception as e:
            print(f"Error processing target director {target_dir['person_name']}: {e}")

    if not target_director_profiles:
        print("No valid target director embeddings found")
        return pd.DataFrame()

    # Convert historical director embeddings to matrix
    historical_person_ids = director_profiles.index.values
    try:
        historical_embedding_matrix = np.vstack(director_profiles['embedding'].values)
    except ValueError as e:
        print(f"Error stacking historical embeddings: {e}")
        return pd.DataFrame()

    # Find similar directors for each target director
    similar_directors = []

    for target_dir in target_director_profiles:
        target_embedding = target_dir['embedding'].reshape(1, -1)  # Shape for cosine_similarity

        # Calculate similarities between this target director and all historical directors
        similarities = cosine_similarity(target_embedding, historical_embedding_matrix)[0]

        # Find top_n most similar
        top_indices = np.argsort(-similarities)[:top_n * 2]  # Get extra for filtering

        rank = 1
        for idx in top_indices:
            sim_score = similarities[idx]
            historical_director_id = historical_person_ids[idx]

            # Skip if it's the same person (shouldn't happen due to exclusion, but safety check)
            if historical_director_id == target_dir['person_id']:
                continue

            if sim_score >= similarity_threshold and rank <= top_n:
                similar_directors.append({
                    'target_title_id': target_title_id,
                    'director_id': target_dir['person_id'],
                    'director_name': target_dir['person_name'],
                    'similar_director_id': historical_director_id,
                    'similar_director_name': director_profiles.loc[historical_director_id, 'director_name'],
                    'similarity_score': sim_score,
                    'rank': rank,
                    'director_title_count': 1,  # Target directors have 1 title (the target title)
                    'similar_title_count': director_profiles.loc[historical_director_id, 'title_count'],
                    'director_titles': [target_title_id],
                    'similar_titles': director_profiles.loc[historical_director_id, 'season_production_id']
                })
                rank += 1
    
    return pd.DataFrame(similar_directors)

def aggregate_director_similarities_for_title(similar_directors_df, target_title_id, 
                                            person_data, role='Director',
                                            weighting_strategy='equal'):
    """
    Aggregate director similarity features for a specific title.
    
    Args:
        similar_directors_df: Output from find_similar_directors()
        target_title_id: Title to generate features for
        person_data: DataFrame with person and title data
        role: Role to filter for
        weighting_strategy: How to weight multiple directors ('equal', 'experience', 'title_count')
    
    Returns:
        Dictionary with aggregated director similarity features
    """
    # Get directors for this title
    # Handle department name mapping
    if role == 'Director':
        department_filter = (person_data['department_name'] == 'Directors')
    else:
        department_filter = (person_data['department_name'] == role)
        
    title_directors = person_data[
        (person_data['season_production_id'] == target_title_id) &
        department_filter
    ]['person_id'].unique()
    
    if len(title_directors) == 0:
        return {
            'season_production_id': target_title_id,
            f'{role.lower()}_similarity_mean': np.nan,
            f'{role.lower()}_similarity_max': np.nan,
            f'{role.lower()}_similarity_std': np.nan,
            f'{role.lower()}_similarity_count': 0,
            f'{role.lower()}_director_count': 0,
            f'{role.lower()}_network_size': np.nan,
        }
    
    # Filter similarities for this title's directors
    title_similarities = similar_directors_df[
        similar_directors_df['director_id'].isin(title_directors)
    ]
    
    if title_similarities.empty:
        return {
            'season_production_id': target_title_id,
            f'{role.lower()}_similarity_mean': np.nan,
            f'{role.lower()}_similarity_max': np.nan,
            f'{role.lower()}_similarity_std': np.nan,
            f'{role.lower()}_similarity_count': 0,
            f'{role.lower()}_director_count': len(title_directors),
            f'{role.lower()}_network_size': np.nan,
        }
    
    # Calculate weights for multiple directors
    if weighting_strategy == 'equal':
        weights = np.ones(len(title_directors)) / len(title_directors)
    elif weighting_strategy == 'experience' or weighting_strategy == 'title_count':
        # Weight by director's historical title count
        director_counts = []
        for director_id in title_directors:
            director_data = title_similarities[title_similarities['director_id'] == director_id]
            if not director_data.empty:
                director_counts.append(director_data['director_title_count'].iloc[0])
            else:
                director_counts.append(1)  # Default weight
        
        weights = np.array(director_counts) / np.sum(director_counts)
    else:
        weights = np.ones(len(title_directors)) / len(title_directors)
    
    # Aggregate similarity features across directors
    all_similarities = title_similarities['similarity_score'].values
    
    # Per-director aggregations
    director_means = []
    director_maxes = []
    network_sizes = []
    
    for i, director_id in enumerate(title_directors):
        director_sims = title_similarities[title_similarities['director_id'] == director_id]
        if not director_sims.empty:
            director_means.append(director_sims['similarity_score'].mean())
            director_maxes.append(director_sims['similarity_score'].max())
            network_sizes.append(len(director_sims))
        else:
            director_means.append(0)
            director_maxes.append(0)
            network_sizes.append(0)
    
    # Weighted aggregation across directors
    weighted_mean = np.average(director_means, weights=weights)
    weighted_max = np.average(director_maxes, weights=weights)
    
    return {
        'season_production_id': target_title_id,
        f'{role.lower()}_similarity_mean': weighted_mean,
        f'{role.lower()}_similarity_max': weighted_max,
        f'{role.lower()}_similarity_std': np.std(all_similarities) if len(all_similarities) > 1 else 0,
        f'{role.lower()}_similarity_count': len(all_similarities),
        f'{role.lower()}_director_count': len(title_directors),
        f'{role.lower()}_network_size': np.mean(network_sizes),
    }

def generate_director_similarity_features_batch(person_data, title_data, role='Director', 
                                               top_n=10, min_titles=2, similarity_threshold=0.1,
                                               weighting_strategies=['equal', 'experience']):
    """
    Generate director similarity features for all titles in batch.
    
    Args:
        person_data: DataFrame with person and embedding data
        title_data: DataFrame with title and date information
        role: Role to process (default 'Director')
        top_n: Number of similar directors to find per director
        min_titles: Minimum titles required for directors
        similarity_threshold: Minimum similarity score threshold
        weighting_strategies: List of weighting methods to apply
    
    Returns:
        DataFrame with director similarity features for all titles
    """
    all_features = []
    
    # Sort titles by preproduction date for efficient processing
    sorted_titles = title_data.sort_values('pre_prod_start_date')['season_production_id'].unique()
    
    for i, title_id in enumerate(sorted_titles):
        if i % 50 == 0:
            print(f"Processing title {i+1}/{len(sorted_titles)}: {title_id}")
        
        try:
            # Find similar directors for this title
            similar_directors = find_similar_directors(
                person_data, title_data, target_title_id=title_id,
                role=role, top_n=top_n, min_titles=min_titles, 
                similarity_threshold=similarity_threshold
            )
            
            # Generate features for each weighting strategy
            for strategy in weighting_strategies:
                features = aggregate_director_similarities_for_title(
                    similar_directors, title_id, person_data, 
                    role=role, weighting_strategy=strategy
                )
                
                # Add strategy suffix to feature names (except season_production_id)
                if strategy != 'equal':
                    strategy_features = {}
                    for key, value in features.items():
                        if key == 'season_production_id':
                            strategy_features[key] = value
                        else:
                            strategy_features[f"{key}_{strategy}"] = value
                    features = strategy_features
                
                all_features.append(features)
        
        except Exception as e:
            print(f"Error processing title {title_id}: {e}")
            # Add empty features for failed titles
            empty_features = {
                'season_production_id': title_id,
                f'{role.lower()}_similarity_mean': np.nan,
                f'{role.lower()}_similarity_max': np.nan,
                f'{role.lower()}_similarity_std': np.nan,
                f'{role.lower()}_similarity_count': 0,
                f'{role.lower()}_director_count': 0,
                f'{role.lower()}_network_size': np.nan,
            }
            all_features.append(empty_features)
    
    # Convert to DataFrame and merge multiple strategies
    if not all_features:
        return pd.DataFrame()
    
    # Group features by title and merge different strategies
    features_df = pd.DataFrame(all_features)
    
    # If multiple strategies, merge on season_production_id
    if len(weighting_strategies) > 1:
        base_features = features_df[features_df.columns[~features_df.columns.str.contains('_experience|_success')]]
        
        merged_df = base_features.copy()
        
        for strategy in weighting_strategies[1:]:  # Skip 'equal' (base)
            strategy_features = features_df[
                features_df.columns[features_df.columns.str.contains(f'_{strategy}') | 
                                   (features_df.columns == 'season_production_id')]
            ].dropna(subset=['season_production_id'])
            
            if not strategy_features.empty:
                merged_df = merged_df.merge(strategy_features, on='season_production_id', how='left')
        
        return merged_df
    else:
        return features_df

def calculate_director_fee_comps(similar_directors_df, person_data, title_data, 
                                fee_column='director_fee_usd', role='Director',
                                weighting_strategy='equal', enable_fallbacks=True):
    """
    Calculate director fee comps based on similar directors' most recent work.
    
    Args:
        similar_directors_df: Output from find_similar_directors()
        person_data: DataFrame with person data
        title_data: DataFrame with title data including fees
        fee_column: Column name containing director fees
        role: Role to filter for
        weighting_strategy: How to weight multiple directors on same title
    
    Returns:
        DataFrame with director fee comp features at title level
    """
    comp_features = []
    
    # Group by target title
    for target_title_id, title_group in similar_directors_df.groupby('target_title_id'):
        
        # Get directors for this title (handle department name mapping)
        if role == 'Director':
            department_filter = (person_data['department_name'] == 'Directors')
        else:
            department_filter = (person_data['department_name'] == role)

        title_directors = person_data[
            (person_data['season_production_id'] == target_title_id) &
            department_filter
        ]['person_id'].unique()
        
        if len(title_directors) == 0:
            comp_features.append({
                'season_production_id': target_title_id,
                f'{role.lower()}_fee_comp_mean': np.nan,
                f'{role.lower()}_fee_comp_median': np.nan,
                f'{role.lower()}_fee_comp_max': np.nan,
                f'{role.lower()}_fee_comp_count': 0,
                f'{role.lower()}_director_count': 0
            })
            continue
        
        # Calculate comps for each director on this title
        director_comps = []
        director_weights = []
        
        for director_id in title_directors:
            # Get similar directors for this specific director
            director_similarities = title_group[title_group['director_id'] == director_id]
            
            if director_similarities.empty:
                director_comps.append(np.nan)
                director_weights.append(1.0)
                continue
            
            # For each similar director, get their most recent title's fee
            similar_fees = []
            
            for _, sim_row in director_similarities.iterrows():
                similar_director_id = sim_row['similar_director_id']
                
                # Get most recent title for this similar director (before target cutoff)
                target_cutoff = title_data[title_data['season_production_id'] == target_title_id]['pre_prod_start_date'].iloc[0]
                
                # Handle department name mapping for similar directors
                if role == 'Director':
                    similar_department_filter = (person_data['department_name'] == 'Directors')
                else:
                    similar_department_filter = (person_data['department_name'] == role)
                    
                similar_director_titles = person_data[
                    (person_data['person_id'] == similar_director_id) &
                    similar_department_filter
                ]['season_production_id'].unique()
                
                # Filter to historical titles only
                historical_similar_titles = title_data[
                    (title_data['season_production_id'].isin(similar_director_titles)) &
                    (title_data['launch_date'] < target_cutoff)
                ]
                
                if not historical_similar_titles.empty:
                    # Convert launch_date to datetime for comparison
                    historical_similar_titles_copy = historical_similar_titles.copy()
                    historical_similar_titles_copy['launch_date_dt'] = pd.to_datetime(
                        historical_similar_titles_copy['launch_date'], errors='coerce'
                    )

                    # Filter out titles with invalid dates
                    valid_dates = historical_similar_titles_copy[
                        historical_similar_titles_copy['launch_date_dt'].notna()
                    ]

                    if not valid_dates.empty:
                        # Get most recent title
                        most_recent_title = valid_dates.loc[
                            valid_dates['launch_date_dt'].idxmax()
                        ]
                    else:
                        # Fallback: use first title if no valid dates
                        most_recent_title = historical_similar_titles.iloc[0]
                    
                    # Get fee for that title
                    if fee_column in most_recent_title and pd.notna(most_recent_title[fee_column]):
                        similar_fees.append(most_recent_title[fee_column])
            
            # Calculate mean fee for this director's similar directors
            if similar_fees:
                director_comp = np.mean(similar_fees)
                director_comps.append(director_comp)
                
                # Weight by experience (title count) if specified
                if weighting_strategy == 'experience':
                    director_weight = director_similarities['director_title_count'].iloc[0]
                else:
                    director_weight = 1.0
                    
                director_weights.append(director_weight)
            else:
                # Fallback strategies if no similar director fees found
                fallback_comp = None
                
                if enable_fallbacks:
                    # Fallback 1: Use this director's own historical average
                    # Handle department name mapping
                    if role == 'Director':
                        fallback_department_filter = (person_data['department_name'] == 'Directors')
                    else:
                        fallback_department_filter = (person_data['department_name'] == role)
                        
                    director_historical_titles = person_data[
                        (person_data['person_id'] == director_id) &
                        fallback_department_filter
                    ]['season_production_id'].unique()
                    
                    target_cutoff = title_data[title_data['season_production_id'] == target_title_id]['pre_prod_start_date'].iloc[0]
                    
                    historical_director_titles = title_data[
                        (title_data['season_production_id'].isin(director_historical_titles)) &
                        (title_data['launch_date'] < target_cutoff)
                    ]
                    
                    if not historical_director_titles.empty and fee_column in historical_director_titles.columns:
                        director_own_fees = historical_director_titles[fee_column].dropna()
                        if len(director_own_fees) > 0:
                            fallback_comp = director_own_fees.mean()
                            print(f"Using director's own historical average: {fallback_comp}")
                    
                    # Fallback 2: Use genre/org average for similar budget range
                    if fallback_comp is None:
                        target_title_info = title_data[title_data['season_production_id'] == target_title_id].iloc[0]
                        
                        # Match on genre and buying org
                        similar_context_titles = title_data[
                            (title_data['genre_desc'] == target_title_info.get('genre_desc')) &
                            (title_data['gravity_buying_organization_desc'] == target_title_info.get('gravity_buying_organization_desc')) &
                            (title_data['launch_date'] < target_cutoff)
                        ]
                        
                        if not similar_context_titles.empty and fee_column in similar_context_titles.columns:
                            context_fees = similar_context_titles[fee_column].dropna()
                            if len(context_fees) > 0:
                                fallback_comp = context_fees.median()
                                print(f"Using genre/org median: {fallback_comp}")
                    
                    # Fallback 3: Use overall historical median
                    if fallback_comp is None:
                        historical_all_titles = title_data[
                            (title_data['launch_date'] < target_cutoff)
                        ]
                        
                        if not historical_all_titles.empty and fee_column in historical_all_titles.columns:
                            all_fees = historical_all_titles[fee_column].dropna()
                            if len(all_fees) > 0:
                                fallback_comp = all_fees.median()
                                print(f"Using overall historical median: {fallback_comp}")
                
                director_comps.append(fallback_comp if fallback_comp is not None else np.nan)
                director_weights.append(1.0)
        
        # Aggregate across multiple directors on the title
        valid_comps = [comp for comp in director_comps if pd.notna(comp)]
        valid_weights = [director_weights[i] for i, comp in enumerate(director_comps) if pd.notna(comp)]
        
        if valid_comps:
            # Normalize weights
            if sum(valid_weights) > 0:
                normalized_weights = np.array(valid_weights) / sum(valid_weights)
            else:
                normalized_weights = np.ones(len(valid_weights)) / len(valid_weights)
            
            # Weighted aggregation
            weighted_mean = np.average(valid_comps, weights=normalized_weights)
            
            comp_features.append({
                'season_production_id': target_title_id,
                f'{role.lower()}_fee_comp_mean': weighted_mean,
                f'{role.lower()}_fee_comp_median': np.median(valid_comps),
                f'{role.lower()}_fee_comp_max': np.max(valid_comps),
                f'{role.lower()}_fee_comp_count': len(valid_comps),
                f'{role.lower()}_director_count': len(title_directors)
            })
        else:
            comp_features.append({
                'season_production_id': target_title_id,
                f'{role.lower()}_fee_comp_mean': np.nan,
                f'{role.lower()}_fee_comp_median': np.nan,
                f'{role.lower()}_fee_comp_max': np.nan,
                f'{role.lower()}_fee_comp_count': 0,
                f'{role.lower()}_director_count': len(title_directors)
            })
    
    return pd.DataFrame(comp_features)

def cache_director_similarities_batch(person_data, title_data, role='Director',
                                    top_n=10, min_titles=1, similarity_threshold=0.05,
                                    use_cache=True, cache_dir=None):
    """
    Cache director similarity calculations for all titles.

    Args:
        person_data: DataFrame with person data
        title_data: DataFrame with title and date information
        role: Role to process
        top_n: Number of similar directors per director
        min_titles: Minimum titles for director inclusion
        similarity_threshold: Minimum similarity threshold
        use_cache: Whether to use caching
        cache_dir: Directory to store cache files

    Returns:
        DataFrame with all similarity relationships
    """
    import os
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    os.makedirs(cache_dir, exist_ok=True)

    # Create cache filename for similarities only
    import hashlib
    similarity_params = f"similarities_{role}_{top_n}_{min_titles}_{similarity_threshold}"
    similarity_hash = hashlib.md5(similarity_params.encode()).hexdigest()[:8]
    similarity_cache_file = os.path.join(cache_dir, f'director_similarities_cache_{similarity_hash}.csv')

    # Try to load similarities from cache
    if use_cache and os.path.exists(similarity_cache_file):
        print(f"Loading cached similarities from: {similarity_cache_file}")
        try:
            cached_similarities = pd.read_csv(similarity_cache_file)
            print(f"Successfully loaded {len(cached_similarities)} cached similarity relationships")
            return cached_similarities
        except Exception as e:
            print(f"Warning: Failed to load similarities cache ({e}), regenerating...")

    print("Computing director similarities...")

    # Check pre_prod_start_date availability
    if 'pre_prod_start_date' not in title_data.columns:
        print("ERROR - 'pre_prod_start_date' column not found in title_data")
        return pd.DataFrame()

    # Sort titles by preproduction date
    valid_titles = title_data.dropna(subset=['pre_prod_start_date'])
    if valid_titles.empty:
        print("ERROR - No titles with valid pre_prod_start_date")
        return pd.DataFrame()

    sorted_titles = valid_titles.sort_values('pre_prod_start_date')['season_production_id'].unique()

    all_similarities = []

    for i, title_id in enumerate(sorted_titles):
        if i % 50 == 0:
            print(f"Computing similarities for title {i+1}/{len(sorted_titles)}: {title_id}")

        try:
            # Find similar directors for this title
            similar_directors = find_similar_directors(
                person_data, title_data, target_title_id=title_id,
                role=role, top_n=top_n, min_titles=min_titles,
                similarity_threshold=similarity_threshold
            )

            if not similar_directors.empty:
                all_similarities.append(similar_directors)

        except Exception as e:
            print(f"Error computing similarities for title {title_id}: {e}")

    # Combine all similarities
    if all_similarities:
        similarities_df = pd.concat(all_similarities, ignore_index=True)
    else:
        similarities_df = pd.DataFrame()

    # Save similarities to cache
    if use_cache and not similarities_df.empty:
        try:
            similarities_df.to_csv(similarity_cache_file, index=False)
            print(f"Similarities cached to: {similarity_cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save similarities cache ({e})")

    return similarities_df

def generate_director_fee_comps_batch(person_data, title_data, role='Director',
                                     fee_column='director_fee_usd', top_n=10,
                                     min_titles=1, similarity_threshold=0.05,
                                     weighting_strategies=['equal', 'experience'],
                                     enable_fallbacks=True, use_cache=True, cache_dir=None):
    """
    Generate director fee comp features for all titles in batch.

    Args:
        person_data: DataFrame with person data
        title_data: DataFrame with title and fee data
        role: Role to process
        fee_column: Column containing director fees
        top_n: Number of similar directors per director
        min_titles: Minimum titles for director inclusion
        similarity_threshold: Minimum similarity threshold
        weighting_strategies: List of weighting methods
        use_cache: Whether to use caching (default True)
        cache_dir: Directory to store cache files (default: analysis/features/data/)

    Returns:
        DataFrame with director fee comp features at title level
    """
    print(f"DEBUG - Starting batch processing:")
    print(f"  person_data shape: {person_data.shape}")
    print(f"  title_data shape: {title_data.shape}")
    print(f"  role: {role}")
    print(f"  fee_column: {fee_column}")

    # Set up caching
    import os
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    os.makedirs(cache_dir, exist_ok=True)

    # Create cache filename based on parameters
    import hashlib
    cache_params = f"{role}_{fee_column}_{top_n}_{min_titles}_{similarity_threshold}_{'-'.join(weighting_strategies)}_{enable_fallbacks}"
    cache_hash = hashlib.md5(cache_params.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f'director_fee_comps_cache_{cache_hash}.csv')

    # Check if fee column exists
    if fee_column not in title_data.columns:
        print(f"ERROR - Column '{fee_column}' not found in title_data. Available columns:")
        print(f"  {list(title_data.columns)}")
        return pd.DataFrame()

    # Check director data availability
    if role == 'Director':
        director_count = len(person_data[person_data['department_name'] == 'Directors'])
        print(f"DEBUG - Found {director_count} director records in person_data")
        if director_count == 0:
            print("ERROR - No directors found in person_data (department_name='Directors')")
            return pd.DataFrame()

    # First, get or compute all similarity relationships (cached separately)
    print("Step 1: Getting director similarities...")
    all_similarities = cache_director_similarities_batch(
        person_data, title_data, role=role, top_n=top_n,
        min_titles=min_titles, similarity_threshold=similarity_threshold,
        use_cache=use_cache, cache_dir=cache_dir
    )

    if all_similarities.empty:
        print("No similarity relationships found")
        return pd.DataFrame()

    print(f"Using {len(all_similarities)} similarity relationships")

    # Check if final results are cached (only fee calculations, not similarities)
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached fee comp results from: {cache_file}")
        try:
            cached_df = pd.read_csv(cache_file)
            print(f"Successfully loaded {len(cached_df)} cached fee comp results")
            return cached_df
        except Exception as e:
            print(f"Warning: Failed to load fee comps cache ({e}), regenerating...")

    # Step 2: Calculate fee comps using cached similarities
    print("Step 2: Computing fee comps from cached similarities...")

    all_comp_features = []
    unique_titles = all_similarities['target_title_id'].unique()

    print(f"Generating director fee comps for {len(unique_titles)} titles...")

    for i, title_id in enumerate(unique_titles):
        if i % 50 == 0:
            print(f"Processing fee comps for title {i+1}/{len(unique_titles)}: {title_id}")

        try:
            # Get similarities for this title from cached data
            similar_directors = all_similarities[all_similarities['target_title_id'] == title_id]
            
            if similar_directors.empty:
                if i < 5:  # Debug first few titles
                    print(f"  DEBUG - No similar directors found for title {title_id}")
                # Add empty features
                for strategy in weighting_strategies:
                    suffix = f"_{strategy}" if strategy != 'equal' else ""
                    empty_features = {
                        'season_production_id': title_id,
                        f'{role.lower()}_fee_comp_mean{suffix}': np.nan,
                        f'{role.lower()}_fee_comp_median{suffix}': np.nan,
                        f'{role.lower()}_fee_comp_max{suffix}': np.nan,
                        f'{role.lower()}_fee_comp_count{suffix}': 0,
                        f'{role.lower()}_director_count{suffix}': 0
                    }
                    all_comp_features.append(empty_features)
                continue
            
            # Calculate fee comps for each weighting strategy
            for strategy in weighting_strategies:
                comp_features = calculate_director_fee_comps(
                    similar_directors, person_data, title_data,
                    fee_column=fee_column, role=role, 
                    weighting_strategy=strategy, enable_fallbacks=enable_fallbacks
                )
                
                # Add strategy suffix if not equal
                if strategy != 'equal':
                    comp_features_renamed = {}
                    for col in comp_features.columns:
                        if col == 'season_production_id':
                            comp_features_renamed[col] = comp_features[col].iloc[0]
                        else:
                            comp_features_renamed[f"{col}_{strategy}"] = comp_features[col].iloc[0]
                    all_comp_features.append(comp_features_renamed)
                else:
                    all_comp_features.append(comp_features.iloc[0].to_dict())
        
        except Exception as e:
            print(f"Error processing title {title_id}: {e}")
            # Add empty features for all strategies
            for strategy in weighting_strategies:
                suffix = f"_{strategy}" if strategy != 'equal' else ""
                empty_features = {
                    'season_production_id': title_id,
                    f'{role.lower()}_fee_comp_mean{suffix}': np.nan,
                    f'{role.lower()}_fee_comp_median{suffix}': np.nan,
                    f'{role.lower()}_fee_comp_max{suffix}': np.nan,
                    f'{role.lower()}_fee_comp_count{suffix}': 0,
                    f'{role.lower()}_director_count{suffix}': 0
                }
                all_comp_features.append(empty_features)
    
    # Convert to DataFrame and merge strategies
    if not all_comp_features:
        return pd.DataFrame()

    features_df = pd.DataFrame(all_comp_features)

    # Ensure exactly one row per title by removing duplicates and handling multiple strategies
    features_df_clean = features_df.drop_duplicates(subset=['season_production_id'])

    # Merge multiple strategies if needed
    if len(weighting_strategies) > 1:
        # Group by season_production_id and combine strategies
        result_rows = []

        for title_id in features_df['season_production_id'].unique():
            title_rows = features_df[features_df['season_production_id'] == title_id]

            # Start with base (equal strategy) features
            base_row = {'season_production_id': title_id}

            # Find base strategy row (without suffix)
            base_cols = [col for col in title_rows.columns if not any(suffix in col for suffix in ['_experience', '_success'])]
            base_data = title_rows[base_cols].dropna().iloc[0] if len(title_rows[base_cols].dropna()) > 0 else None

            if base_data is not None:
                base_row.update(base_data.to_dict())

            # Add strategy-specific columns
            for strategy in weighting_strategies[1:]:
                strategy_cols = [col for col in title_rows.columns if f'_{strategy}' in col]
                if strategy_cols:
                    strategy_data = title_rows[['season_production_id'] + strategy_cols].dropna()
                    if not strategy_data.empty:
                        strategy_row = strategy_data.iloc[0]
                        for col in strategy_cols:
                            base_row[col] = strategy_row[col]

            result_rows.append(base_row)

        result_df = pd.DataFrame(result_rows)
    else:
        result_df = features_df_clean

    # Save to cache if enabled
    if use_cache and not result_df.empty:
        try:
            result_df.to_csv(cache_file, index=False)
            print(f"Results cached to: {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache file ({e})")

    return result_df


def clear_director_similarities_cache(cache_dir=None):
    """
    Clear all cached director similarity files.

    Args:
        cache_dir: Directory containing cache files (default: analysis/features/data/)
    """
    if cache_dir is None:
        import os
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    import glob
    cache_files = glob.glob(os.path.join(cache_dir, 'director_similarities_cache_*.csv'))

    if not cache_files:
        print("No similarity cache files found")
        return

    removed_count = 0
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            removed_count += 1
            print(f"Removed: {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"Failed to remove {cache_file}: {e}")

    print(f"Cleared {removed_count} similarity cache files")

def clear_director_fee_comps_cache(cache_dir=None):
    """
    Clear all cached director fee comp files.

    Args:
        cache_dir: Directory containing cache files (default: analysis/features/data/)
    """
    if cache_dir is None:
        import os
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    import glob
    cache_files = glob.glob(os.path.join(cache_dir, 'director_fee_comps_cache_*.csv'))

    if not cache_files:
        print("No fee comp cache files found")
        return

    removed_count = 0
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            removed_count += 1
            print(f"Removed: {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"Failed to remove {cache_file}: {e}")

    print(f"Cleared {removed_count} fee comp cache files")

def clear_all_director_caches(cache_dir=None):
    """
    Clear both similarity and fee comp cache files.

    Args:
        cache_dir: Directory containing cache files (default: analysis/features/data/)
    """
    print("Clearing all director cache files...")
    clear_director_similarities_cache(cache_dir)
    clear_director_fee_comps_cache(cache_dir)
