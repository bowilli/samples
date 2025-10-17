# Model Comparison with 1301 Validation Set

## Overview
This feature allows you to compare current director models against the original 1301 model using the same validation set.

## Workflow

```
1. Check what's missing          →  python check_missing_validation_titles.py
   ↓
2. Understand why it's missing   →  python trace_missing_titles.py
   ↓
3. Train models with 1301 split  →  python ../main.py --use-1301-validation
   ↓
4. Compare performance           →  python compare_models.py
```

### Quick Start: Run Full Analysis

To run the entire analysis pipeline in one go:

```bash
cd model_comparison
./run_full_analysis.sh
```

This will automatically:
1. Check for missing titles
2. Trace dropout reasons
3. Ask if you want to train models (optional)
4. Compare model performance (if models were trained)

Or run each step individually (see below).

## Usage

To run the comparison:

```bash
python main.py --use-1301-validation
```

This will:
1. **Only run director models** (ATL models are skipped)
2. **Use the validation set from 1301_predictions_results.csv** for the test split
3. **Generate comparison outputs** in the `model_comparison/` directory

## Generated Outputs

### 1. Dataset Comparison Report
**File:** `reports/dataset_comparison_report.txt`

Contains:
- Training vs validation set sizes
- Target variable statistics for each set
- Organization distribution across sets

### 2. Individual Model Predictions
**Directory:** `current_model_results/`

Files: `{model}_{feature_set}.csv` (e.g., `catboost_set_5.csv`)

Format (matches 1301 format):
- `season_production_id`: Title ID
- `buying_org_name`: Organization
- `dataset_type`: "training" or "validation"
- `target`: Actual director fee
- `prediction`: Model prediction
- `absolute_error`: |actual - predicted|
- `percentage_error`: (absolute_error / actual) * 100

### 3. Comprehensive Comparison Summary
**File:** `reports/model_comparison_summary.csv`

Contains for each model and dataset type:
- Model name and feature set
- Number of samples
- **MAE** (Median Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Adjusted R-squared)
- **MAPE** (Median Absolute Percentage Error)
- **% titles with <20% error**
- **% titles with <6% error**

## Comparison with Original 1301 Model

The original 1301 predictions are located at:
`model_comparison/original_model_results/1301_predictions_results.csv`

To compare:
1. Look at validation set metrics in `reports/model_comparison_summary.csv`
2. Compare individual predictions by matching `season_production_id` between current and 1301 files
3. Review the dataset comparison report to ensure data consistency

## Checking for Missing Validation Titles

### Quick Check (What's Missing)

First, check if any titles are missing:

```bash
cd model_comparison
python check_missing_validation_titles.py
```

This script will:
1. Load the 1301 validation titles
2. Check which ones are present in the current dataset
3. Generate reports on any missing titles
4. Analyze the potential impact on metrics

**Output Files:**
- **`reports/missing_validation_titles_report.txt`**: Detailed text report with full analysis
- **`reports/missing_validation_titles.csv`**: List of missing titles with all details
- **`reports/missing_titles_summary.csv`**: Summary statistics

### Deep Dive (Why They're Missing)

#### Step 2a: Diagnose Feature File Structure

If you see unusually high percentages of missing titles (like 80%+), first check if it's a data structure issue:

```bash
cd model_comparison
python diagnose_feature_files.py
```

This script examines the actual structure of feature files to determine if:
- Files are person-level vs title-level (requiring aggregation)
- Titles are truly missing or just structured differently
- The merge strategy is filtering out titles incorrectly

#### Step 2b: Trace Through Pipeline

Then trace titles through the data pipeline:

```bash
cd model_comparison
python trace_missing_titles.py
```

This script traces each missing validation title through the data processing pipeline to identify exactly where and why it was filtered out. It checks:

1. **Overall Coverage Analysis**: What % of titles with directors have median_director_efc features (shows if comparable director matching is too strict)
2. **Director Data Existence**: Whether titles have ANY director data in the source (fundamental check)
3. **Embedding Completeness Filter**: Titles with <40% director embedding completeness
4. **Target Value Cleaning**: Titles with missing or invalid director fees
5. **Feature Merges**: Titles missing required features like comparable director data

**Output Files:**
- **`reports/missing_titles_trace_report.txt`**: Comprehensive trace showing where each title dropped out
- **`reports/missing_titles_with_reasons.csv`**: CSV with dropout reasons for each title

**Common Dropout Reasons:**
- **No director data in source**: Title has no directors at all (fundamental data issue)
- **Only 1 director on title**: May not meet minimum director requirements for some features
- **Embedding completeness <40%**: Not enough directors with embeddings on the title
- **Missing comparable director data**: No similar directors found for comparison features
- **Missing/invalid target values**: Director fee not recorded or <= 0
- **Missing director profile features**: Required director history features unavailable

This helps you understand:
- Which pipeline stage is removing the most titles
- Whether filtering logic is too strict
- If data quality issues exist upstream
- Whether titles should be restored to the dataset

## Comparing Models

After generating predictions with `--use-1301-validation`, run the comparison script:

```bash
cd model_comparison
python compare_models.py
```

This script will:
1. Load the 1301 model predictions
2. Load all current model predictions
3. Calculate comprehensive metrics for all models
4. Identify the best performing new model
5. Generate a detailed comparison report

### Output

The script produces:

1. **Console Output**: Side-by-side comparison table showing:
   - Validation and training metrics
   - Performance improvements/regressions
   - Rankings of all models by MAPE

2. **`reports/detailed_model_comparison.csv`**: Complete comparison data for all models including:
   - All performance metrics (MAPE, MAE, RMSE, R², etc.)
   - Both training and validation results
   - Sorted by validation MAPE

### Example Output

```
PERFORMANCE COMPARISON: 1301 MODEL vs BEST NEW MODEL
================================================================================
Metric                         1301 Model           Best New Model       Improvement

VALIDATION SET METRICS:
--------------------------------------------------------------------------------
MAPE (%)                       45.23%               32.18%               ✓ 28.9% better
Median Absolute Error ($)      $12,500              $8,200               ✓ 34.4% better
...
```

## Notes

- The validation set IDs are extracted from the 1301 file (dataset_type == "validation")
- Training set can include additional titles not in the 1301 file
- Only director models are trained when using this flag
- Intelligent director models are skipped in this mode for simplicity
