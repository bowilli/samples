# Feature Engineering Usage

The feature engineering pipeline can now be run with command line arguments to control data regeneration.

**Note**: The script can now be run from **any directory** - it will automatically find the correct paths.

## Usage Options

### 1. Normal Run (uses existing data)

From the analysis directory:
```bash
python features/feature_engineering.py
```

From the repository root:  
```bash
python analysis/features/feature_engineering.py
```

From anywhere:
```bash
python /path/to/repo/analysis/features/feature_engineering.py
```

- Loads existing CSV files if they exist
- Skips feature engineering if processed data already exists
- Only runs SQL queries if raw CSV files are missing

### 2. Regenerate Everything
```bash
python features/feature_engineering.py --force-all
```
- Forces regeneration of both SQL data AND processed data
- Overwrites all CSV files in `data/raw_data/` and `data/processed_data/`
- Runs the complete feature engineering pipeline
- Use this for a complete fresh rebuild with new SQL data

### 3. Regenerate Processed Data Only
```bash
python features/feature_engineering.py --process-only
```
- Forces regeneration of only processed data
- Keeps existing raw SQL data in `data/raw_data/`
- Overwrites files in `data/processed_data/`
- Use this when you've changed feature engineering logic but don't need fresh SQL data

### 4. Clear Director Model Caches
```bash
python features/feature_engineering.py --clear-cache
```
- Clears all director similarity and fee comparison caches
- Useful when director model logic has changed
- Exits after clearing caches (doesn't run feature engineering)
- Use this before `--process-only` if director features have been modified

## Typical Workflow

1. **Feature engineering changes**: Use `--process-only` when you modify feature engineering logic but don't need fresh SQL data
2. **Director model changes**: Use `--clear-cache` first, then `--process-only` when director features or logic are modified
3. **Fresh data needed**: Use `--force-all` when you want updated data from SQL and complete regeneration
4. **Regular runs**: Use no arguments to quickly load existing processed data
5. **Complete rebuild**: Use `--force-all` for full pipeline regeneration

## Missing Data Visualizations

**Important**: The feature engineering pipeline only generates **text reports** for missing data. To create the **PNG visualization charts**, you need a separate step:

### After Running Feature Engineering
```bash
# Step 1: Run feature engineering (generates missing_features_report.txt)
python features/feature_engineering.py --force-all

# Step 2: Generate missing data visualizations (generates PNG charts)  
python utils/create_missing_data_visuals.py
```

### Generated Visualizations
The visualization script creates:
- `missing_data_overall_high.png` - Features with >25% missing data
- `embedding_completeness_distributions.png` - Cast and director embedding completeness  
- `organization_missing_features.png` - Missing data heatmap by buying organization
- `team_missing_features.png` - Missing data heatmap by buying team

**Note**: The "high missing data" chart only shows features with >25% missing values. If your SQL fixes reduce a feature below this threshold, it will correctly disappear from the chart.

## Key Parameters (Latest Updates)

- **EMBEDDING_THRESHOLD_ATL**: 40% threshold for ATL models (optimized for performance)
- **EMBEDDING_THRESHOLD_DIRECTOR**: 40% threshold for Director models (optimized for performance)
- **IQR_THRESHOLD**: Increased from 3 to 4 to be less aggressive with outlier removal
- **ISOLATION_FOREST_CONTAMINATION**: Set to 0.02 from testing
- **MISSING_DATA_THRESHOLD**: Updated to 0.40 (40% missing data tolerance, increased for better feature retention)

## Differential Organization Filtering

- **ATL models**: Include Animation Series and Licensing data for broader training set
- **Director models**: Exclude Animation Series and Licensing data for focused training

## Missing Value Handling

The pipeline uses enhanced hierarchical categorical imputation with three organizational levels:
1. **Sub Team mode**: Fill missing values with most common value within the sub buying team (minimum 3 titles)
2. **Team mode**: If sub team mode fails, use most common value within the buying team (minimum 5 titles)
3. **Organization mode**: If team mode fails, use most common value within organization (minimum 10 titles)
4. **"Missing" fallback**: If all levels fail, use "Missing" as the category

This provides more granular imputation at the sub-team level while maintaining fallback strategies for smaller groups.

## Modular Architecture

The feature engineering pipeline now uses a modular architecture for better maintainability:

### Core Modules
- **`feature_engineering.py`**: Main pipeline with data cleaning, feature creation, and orchestration
- **`embeddings_features.py`**: Embedding-based similarity features and director fee comparisons
- **`comps_features.py`**: Comparison features from external similarity models
- **`production_features.py`**: Director profile and production-specific features

### Utility Modules (in `utils/`)
- **`feature_reporter.py`**: Dedicated reporting functions for missing data analysis and dataset saving
- **`sample_size_reporter.py`**: Sample size tracking and logging utilities
- **`title_funnel_tracker.py`**: Title filtering and funnel analysis

### Key Reporting Functions (in utils/feature_reporter.py)
- **`generate_missing_features_report()`**: Comprehensive missing data analysis with statistical significance testing
- **`generate_embedding_completeness_report()`**: Dedicated embedding completeness analysis by role (cast/director)
- **`save_processed_datasets()`**: Saves datasets to both analysis and app directories

### Smart Import System
The pipeline uses a `safe_import()` function that handles both relative and absolute imports automatically, making the code more robust when run from different directories.

## File Structure

The pipeline generates:
- **Raw data**: `data/raw_data/*.csv` (from SQL queries)
- **Processed data**: `data/processed_data/*.csv` (from feature engineering)
- **Model outputs**: `data/model_outputs/*.csv` (from main.py analysis)
- **Reports**: `data/processed_data/missing_features_report.txt` (from utils/feature_reporter.py)

## Troubleshooting Missing Values

If you encounter missing value errors:
1. Use `--process-only` to regenerate processed data with enhanced hierarchical imputation
2. Check `MISSING_DATA_THRESHOLD` (currently 40%) if too many features are being dropped
3. Review embedding completeness thresholds if models fail due to insufficient data
4. Use `--clear-cache` before `--process-only` if director-related features are problematic
5. Check the missing features report in `data/processed_data/missing_features_report.txt` for detailed analysis

## Performance Notes

- The enhanced hierarchical imputation (sub-team → team → org → "Missing") provides better data quality
- The modular architecture improves maintainability without impacting performance
- Director model caching significantly speeds up repeated runs - only clear when logic changes
