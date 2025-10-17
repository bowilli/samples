# features/feature_selection.py

# Define feature sets for each target variable
FEATURE_SETS = {
    'atl_efc_usd': {
        # 'set_1': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_cast_count','os_production_type','relevant_title_age','median_atl_efc_usd'],
        #     'description': 'Comps approach - nearest neighbors'
        # },
        'set_2': {
            'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_cast_count','os_production_type','relevant_title_age','mean_cast_embedding_scalar','median_cast_embedding_scalar','max_cast_embedding_scalar'],
            'description': 'Embeddings approach'
        },
        # 'set_3': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','relevant_title_age'],
        #     'description': 'Base model without comps or embeddings'
        # },
        # 'set_4': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_cast_count','os_production_type','relevant_title_age','base_median_atl_efc_usd'],
        #     'description': 'Comps approach - algo comps'
        # },
        # 'set_5': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_cast_count','os_production_type','relevant_title_age','median_atl_efc_usd',
        #                  'is_oad','is_musical','est_shooting_days','num_shooting_locations','episode_count'],
        #     'description': 'Exploratory results base model'
        # }
    },
    'director_efc_usd': {
        # 'set_1': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_director_count','os_production_type','more_than_one_director','relevant_title_age','median_director_efc_usd'],
        #     'description': 'Comps approach - nearest neighbors'
        # },
        # 'set_2': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_director_count','os_production_type','more_than_one_director','relevant_title_age','mean_directors_embedding_scalar','median_directors_embedding_scalar','max_directors_embedding_scalar'],
        #     'description': 'Embeddings approach'
        # },
        # 'set_3': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age', 'is_oad','is_musical','est_shooting_days','num_shooting_locations','episode_count'],
        #     'description': 'Base model without comps or embeddings'
        # },
        


        # 'set_4': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_director_count','os_production_type','more_than_one_director','relevant_title_age','median_director_efc_usd'],
        #     'description': 'Comps approach - embeddings nearest neighbors'
        # },
        # 'set_5': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
        #                 'is_musical','est_shooting_days','num_shooting_locations','episode_count','median_director_efc_usd'],
        #     'description': 'Exploratory results comps - nearest neighbors'
        # },
        # 'set_6': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
        #                 'is_musical','est_shooting_days','num_shooting_locations','episode_count','mean_directors_embedding_scalar','median_directors_embedding_scalar','max_directors_embedding_scalar'],
        #     'description': 'Exploratory results embeddings - algo comps'
        # },
        # 'set_7': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
        #                 'is_musical','est_shooting_days','num_shooting_locations','episode_count'],
        #     'description': 'Exploratory results base model'
        # },
        # 'set_8': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_director_count','os_production_type','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
        #                 'is_musical','est_shooting_days','num_shooting_locations','episode_count','base_median_director_efc_usd'],
        #     'description': 'Exploratory results comps - algo comps'
        # },
        # 'set_9': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
        #                 'is_musical','est_shooting_days','num_shooting_locations','episode_count','genre_desc','primary_language','primary_country_shooting_location','gravity_country_of_origin',
        #                 'gravity_buying_organization_desc','ownership_structure', 'management_structure', 'nonfiction_series_buying_org','genre_documentary_standup','vfx_tier','vfx_touch_tier','median_director_efc_usd'],
        #     'description': 'Exploratory + categorical features comps - nearest neighbors'
        # },
        'set_10': {
            'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
                         'is_musical','est_shooting_days','num_shooting_locations','episode_count','genre_desc','primary_language','primary_country_shooting_location','gravity_country_of_origin',
                         'gravity_buying_organization_desc','ownership_structure', 'management_structure', 'nonfiction_series_buying_org','genre_documentary_standup','vfx_tier','vfx_touch_tier','mean_directors_embedding_scalar','median_directors_embedding_scalar','max_directors_embedding_scalar'],
            'description': 'Exploratory + categorical features embeddings - algo comps'
        },
        # 'set_11': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
        #                  'is_musical','est_shooting_days','num_shooting_locations','episode_count','genre_desc','primary_language','primary_country_shooting_location','gravity_country_of_origin',
        #                  'gravity_buying_organization_desc','ownership_structure', 'management_structure', 'nonfiction_series_buying_org','genre_documentary_standup','vfx_tier','vfx_touch_tier'],
        #     'description': 'Exploratory + categorical features base model'
        # },
        # 'set_12': {
        #     'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','total_director_count','os_production_type','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
        #                  'is_musical','est_shooting_days','num_shooting_locations','episode_count','genre_desc','primary_language','primary_country_shooting_location','gravity_country_of_origin',
        #                  'gravity_buying_organization_desc','ownership_structure', 'management_structure', 'nonfiction_series_buying_org','genre_documentary_standup','vfx_tier','vfx_touch_tier','base_median_director_efc_usd'],
        #     'description': 'Exploratory + categorical features comps - algo comps'
        # },

        'set_13':
         {
            'features': ['gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director','director_fee_comp_median','director_fee_comp_max', 'is_oad','is_musical','est_shooting_days','num_shooting_locations','episode_count'],
            'description': 'Updated Embeddings'
        },
        # 'set_14': {
        #     'base_features': [
        #         'gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director',
        #         'mean_director_nos_filmography_strength_rescaled_max','max_director_nos_filmography_strength_rescaled_max',
        #         'mean_director_max_box_office','max_director_max_box_office',
        #         'mean_director_nos_filmography_strength_rescaled_country_origin','max_director_nos_filmography_strength_rescaled_country_origin',
        #         'mean_director_director_experience','max_director_director_experience',
        #         'director_count', 'is_oad','is_musical','est_shooting_days','num_shooting_locations','episode_count'
        #     ],
        #     'fee_features': [
        #         'mean_director_previous_director_efc_usd','max_director_previous_director_efc_usd',
         
        #     ],
            #   'creates_variants': True,
        #     'description': 'Base model + director profile features (filmography, box office, experience, previous fees)'
        # },

        'set_15': {
            'base_features': [
                'gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director',
                'mean_director_nos_filmography_strength_rescaled_max','max_director_nos_filmography_strength_rescaled_max',
                'mean_director_max_box_office','max_director_max_box_office',
                'mean_director_nos_filmography_strength_rescaled_country_origin','max_director_nos_filmography_strength_rescaled_country_origin',
                'mean_director_director_experience','max_director_director_experience',
                'director_count', 'director_fee_comp_median','director_fee_comp_max', 'is_oad','is_musical','est_shooting_days','num_shooting_locations','episode_count'
            ],
            'fee_features': [
                'mean_director_previous_director_efc_usd','max_director_previous_director_efc_usd',

            ],
            'creates_variants': True,
            'description': 'Base model + director profile features (filmography, box office, experience, previous fees) + embeddings'
        },

            'set_16': {
            'base_features': [
                'gravity_buying_team', 'returning_season','unscripted','runtime_min','os_production_type','total_director_count','more_than_one_director',
                'mean_director_nos_filmography_strength_rescaled_max','max_director_nos_filmography_strength_rescaled_max','max_director_director_experience',
                'director_count', 'director_fee_comp_median','director_fee_comp_max', 'is_oad','is_musical','est_shooting_days','num_shooting_locations','episode_count', 'director_fee_comp_max',
            ],
            'fee_features': [
                'mean_director_previous_director_efc_usd','max_director_previous_director_efc_usd',
     
            ],
            'creates_variants': True,
            'description': 'Base model + director profile features simplified + embeddings'
        },

        'set_17': {
            'features': [
                'gravity_buying_team','gravity_sub_buying_team', 'returning_season','unscripted','is_animated','runtime_min','os_production_type','total_director_count','more_than_one_director','relevant_title_age','total_cast_count','is_oad',
                         'is_musical','est_shooting_days','num_shooting_locations','episode_count','genre_desc','primary_language','primary_country_shooting_location','gravity_country_of_origin',
                         'gravity_buying_organization_desc','ownership_structure', 'management_structure', 'nonfiction_series_buying_org','genre_documentary_standup','vfx_tier','vfx_touch_tier'
            ],
            'description': 'Exploratory + categorical features base model with sub-buying team and animated'
        },
    }
}

def select_features(data, feature_set_name, target='atl_efc_usd', include_fees=None, verbose=False):
    """
    Select features based on the specified feature set name and target variable.

    Args:
        data: DataFrame containing the features
        feature_set_name: Name of the feature set (e.g., 'set_5', 'set_5_no_fees', 'set_5_with_fees')
        target: Target variable name
        include_fees: Boolean to include/exclude fee features. If None, auto-detected from feature_set_name
        verbose: Whether to print warnings about missing features

    Missing features are silently omitted unless verbose=True.
    """
    if target not in FEATURE_SETS:
        raise ValueError(f"Target '{target}' is not defined. Available targets: {list(FEATURE_SETS.keys())}")

    # Handle intelligent routing - auto-detect fee inclusion from set name
    base_set_name = feature_set_name
    if include_fees is None:
        if '_no_fees' in feature_set_name:
            include_fees = False
            base_set_name = feature_set_name.replace('_no_fees', '')
        elif '_with_fees' in feature_set_name:
            include_fees = True
            base_set_name = feature_set_name.replace('_with_fees', '')
        else:
            # For base sets like 'set_5', include fees by default (backward compatibility)
            include_fees = True

    # Check if base set exists
    if base_set_name not in FEATURE_SETS[target]:
        # If the original name doesn't exist after transformation, try the original
        if feature_set_name in FEATURE_SETS[target]:
            base_set_name = feature_set_name
        else:
            raise ValueError(f"Feature set '{feature_set_name}' (base: '{base_set_name}') is not defined for target '{target}'.")

    # Get the feature set configuration
    feature_set = FEATURE_SETS[target][base_set_name]

    # Handle different feature set structures
    if isinstance(feature_set, dict) and 'base_features' in feature_set:
        # New structure with base_features and fee_features
        requested_features = feature_set['base_features'].copy()
        if include_fees and 'fee_features' in feature_set:
            requested_features.extend(feature_set['fee_features'])
    elif isinstance(feature_set, dict) and 'features' in feature_set:
        # Old structure with explicit features list
        requested_features = feature_set['features']
    else:
        # Fallback for simple list structure
        requested_features = feature_set

    # Only select features that exist in the data
    selected_features = {}
    missing_features = []

    for feature in requested_features:
        if feature in data.columns:
            selected_features[feature] = data[feature]
        else:
            missing_features.append(feature)

    # Optional verbose output for debugging
    if verbose and missing_features:
        fee_status = "with fees" if include_fees else "without fees"
        print(f"Warning: {len(missing_features)} features not found in data and were omitted ({fee_status}): {missing_features}")

    return selected_features

def get_feature_set_description(target, feature_set_name):
    """Get the description for a specific feature set."""
    if target not in FEATURE_SETS:
        return "Target not found"

    # Handle intelligent naming - get base set name
    base_set_name = feature_set_name
    suffix = ""
    if '_no_fees' in feature_set_name:
        base_set_name = feature_set_name.replace('_no_fees', '')
        suffix = " (without fee history)"
    elif '_with_fees' in feature_set_name:
        base_set_name = feature_set_name.replace('_with_fees', '')
        suffix = " (with fee history)"

    # Check base set exists
    if base_set_name not in FEATURE_SETS[target]:
        # Fallback to original name if base doesn't exist
        if feature_set_name in FEATURE_SETS[target]:
            base_set_name = feature_set_name
            suffix = ""
        else:
            return "Feature set not found"

    feature_set = FEATURE_SETS[target][base_set_name]
    if isinstance(feature_set, dict) and 'description' in feature_set:
        return feature_set['description'] + suffix
    return "No description available"
