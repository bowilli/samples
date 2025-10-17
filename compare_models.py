"""
Compare performance metrics between the original 1301 model and the best new models.

This script analyzes predictions from the 1301 model and all current models,
identifies the best performing new model, and generates a detailed comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import os

def calculate_metrics(df, actual_col='target', pred_col='prediction'):
    """Calculate comprehensive performance metrics."""
    actual = df[actual_col]
    predicted = df[pred_col]

    # Basic metrics
    mae = np.median(np.abs(actual - predicted))
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    # R²
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Adjusted R² (simplified - assumes 10 features as baseline)
    n = len(actual)
    p = 10
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    # MAPE
    non_zero_mask = actual != 0
    if non_zero_mask.sum() > 0:
        mape = np.median(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]) * 100)
    else:
        mape = np.nan

    # Error rate metrics
    pct_error = np.abs((actual - predicted) / actual * 100)
    pct_under_20 = (pct_error < 20).sum() / len(pct_error) * 100
    pct_under_6 = (pct_error < 6).sum() / len(pct_error) * 100

    return {
        'n_samples': len(df),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'adj_r2': adj_r2,
        'mape': mape,
        'pct_titles_under_20_error': pct_under_20,
        'pct_titles_under_6_error': pct_under_6,
        'mean_actual': actual.mean(),
        'mean_predicted': predicted.mean()
    }


def load_1301_metrics():
    """Load and calculate metrics for the 1301 model."""
    print("Loading 1301 model predictions...")
    df = pd.read_csv('original_model_results/1301_predictions_results.csv')

    # Rename columns to match expected format
    df = df.rename(columns={
        'target_1301': 'target',
        'prediction_1301': 'prediction'
    })

    # Calculate metrics for training and validation separately
    train_df = df[df['dataset_type'] == 'training']
    val_df = df[df['dataset_type'] == 'validation']

    metrics = {
        'model_name': '1301_original',
        'training': calculate_metrics(train_df),
        'validation': calculate_metrics(val_df),
        'combined': calculate_metrics(df)
    }

    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")

    return metrics, df


def load_current_models_metrics():
    """Load and calculate metrics for all current models."""
    print("\nLoading current model predictions...")
    current_results_dir = Path('current_model_results')

    if not current_results_dir.exists():
        print("  ERROR: current_model_results directory not found!")
        print("  Please run: python main.py --use-1301-validation")
        return []

    model_files = list(current_results_dir.glob('*.csv'))

    if not model_files:
        print("  ERROR: No model prediction files found!")
        print("  Please run: python main.py --use-1301-validation")
        return []

    all_metrics = []

    for filepath in model_files:
        model_name = filepath.stem  # e.g., 'catboost_set_5'
        print(f"  Processing {model_name}...")

        df = pd.read_csv(filepath)

        train_df = df[df['dataset_type'] == 'training']
        val_df = df[df['dataset_type'] == 'validation']

        metrics = {
            'model_name': model_name,
            'training': calculate_metrics(train_df),
            'validation': calculate_metrics(val_df),
            'combined': calculate_metrics(df)
        }

        all_metrics.append(metrics)

    return all_metrics


def find_best_model(all_metrics):
    """Find the best performing model based on validation MAPE."""
    if not all_metrics:
        return None

    # Sort by validation MAPE (lower is better)
    sorted_metrics = sorted(all_metrics, key=lambda x: x['validation']['mape'])
    return sorted_metrics[0]


def create_comparison_table(metric_1301, best_new_model):
    """Create a detailed comparison table."""
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON: 1301 MODEL vs BEST NEW MODEL")
    print("="*100)

    # Header
    print(f"\n{'Metric':<30} {'1301 Model':<20} {'Best New Model':<20} {'Improvement':<20}")
    print("-"*100)
    print(f"{'Model Name':<30} {'1301_original':<20} {best_new_model['model_name']:<20}")
    print("-"*100)

    # Validation metrics (most important)
    print("\nVALIDATION SET METRICS:")
    print("-"*100)

    metrics_to_compare = [
        ('n_samples', 'Samples', '{:.0f}', False),
        ('mape', 'MAPE (%)', '{:.2f}%', True),
        ('mae', 'Median Absolute Error ($)', '${:,.0f}', True),
        ('rmse', 'RMSE ($)', '${:,.0f}', True),
        ('adj_r2', 'Adjusted R²', '{:.4f}', False),
        ('pct_titles_under_20_error', '% Titles <20% Error', '{:.1f}%', False),
        ('pct_titles_under_6_error', '% Titles <6% Error', '{:.1f}%', False),
    ]

    for metric_key, metric_name, fmt, lower_is_better in metrics_to_compare:
        old_val = metric_1301['validation'][metric_key]
        new_val = best_new_model['validation'][metric_key]

        # Format values
        if '$' in fmt:
            old_str = fmt.format(old_val)
            new_str = fmt.format(new_val)
        elif '%' in fmt:
            old_str = fmt.format(old_val)
            new_str = fmt.format(new_val)
        else:
            old_str = fmt.format(old_val)
            new_str = fmt.format(new_val)

        # Calculate improvement
        if metric_key == 'n_samples':
            improvement = "N/A"
        elif np.isnan(old_val) or np.isnan(new_val):
            improvement = "N/A"
        else:
            if lower_is_better:
                pct_change = ((old_val - new_val) / old_val * 100)
                if pct_change > 0:
                    improvement = f"✓ {pct_change:.1f}% better"
                else:
                    improvement = f"✗ {-pct_change:.1f}% worse"
            else:
                pct_change = ((new_val - old_val) / old_val * 100)
                if pct_change > 0:
                    improvement = f"✓ {pct_change:.1f}% better"
                else:
                    improvement = f"✗ {-pct_change:.1f}% worse"

        print(f"{metric_name:<30} {old_str:<20} {new_str:<20} {improvement:<20}")

    # Training metrics
    print("\n\nTRAINING SET METRICS:")
    print("-"*100)

    for metric_key, metric_name, fmt, lower_is_better in metrics_to_compare:
        old_val = metric_1301['training'][metric_key]
        new_val = best_new_model['training'][metric_key]

        # Format values
        if '$' in fmt:
            old_str = fmt.format(old_val)
            new_str = fmt.format(new_val)
        elif '%' in fmt:
            old_str = fmt.format(old_val)
            new_str = fmt.format(new_val)
        else:
            old_str = fmt.format(old_val)
            new_str = fmt.format(new_val)

        # Calculate improvement (for training)
        if metric_key == 'n_samples':
            improvement = "N/A"
        elif np.isnan(old_val) or np.isnan(new_val):
            improvement = "N/A"
        else:
            if lower_is_better:
                pct_change = ((old_val - new_val) / old_val * 100)
            else:
                pct_change = ((new_val - old_val) / old_val * 100)
            improvement = f"{pct_change:+.1f}%"

        print(f"{metric_name:<30} {old_str:<20} {new_str:<20} {improvement:<20}")

    print("\n" + "="*100)


def save_comparison_csv(metric_1301, all_current_metrics):
    """Save detailed comparison to CSV."""
    rows = []

    # Add 1301 model
    for dataset_type in ['training', 'validation']:
        row = {'model_name': '1301_original', 'dataset_type': dataset_type}
        row.update(metric_1301[dataset_type])
        rows.append(row)

    # Add all current models
    for metrics in all_current_metrics:
        for dataset_type in ['training', 'validation']:
            row = {'model_name': metrics['model_name'], 'dataset_type': dataset_type}
            row.update(metrics[dataset_type])
            rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['model_name', 'dataset_type', 'n_samples', 'mape', 'mae', 'rmse', 'adj_r2', 'r2',
            'pct_titles_under_20_error', 'pct_titles_under_6_error', 'mean_actual', 'mean_predicted']
    df = df[cols]

    # Sort by validation MAPE
    df['sort_key'] = df.apply(lambda x: x['mape'] if x['dataset_type'] == 'validation' else 999, axis=1)
    df = df.sort_values(['sort_key', 'dataset_type'], ascending=[True, False])
    df = df.drop('sort_key', axis=1)

    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)

    output_file = 'reports/detailed_model_comparison.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed comparison saved to: {output_file}")

    return df


def main():
    """Main execution function."""
    print("="*100)
    print("MODEL COMPARISON ANALYSIS")
    print("="*100)

    # Load 1301 metrics
    metric_1301, df_1301 = load_1301_metrics()

    # Load current models metrics
    all_current_metrics = load_current_models_metrics()

    if not all_current_metrics:
        print("\n❌ No current model predictions found. Exiting.")
        return

    # Find best model
    print(f"\n✓ Found {len(all_current_metrics)} current models")
    best_new_model = find_best_model(all_current_metrics)
    print(f"\n🏆 Best new model: {best_new_model['model_name']}")
    print(f"   Validation MAPE: {best_new_model['validation']['mape']:.2f}%")
    print(f"   Validation R²: {best_new_model['validation']['adj_r2']:.4f}")

    # Create comparison table
    create_comparison_table(metric_1301, best_new_model)

    # Save detailed comparison
    df_comparison = save_comparison_csv(metric_1301, all_current_metrics)

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    val_mape_1301 = metric_1301['validation']['mape']
    val_mape_new = best_new_model['validation']['mape']
    improvement = ((val_mape_1301 - val_mape_new) / val_mape_1301 * 100)

    if improvement > 0:
        print(f"\n✓ The best new model ({best_new_model['model_name']}) shows {improvement:.1f}% improvement in validation MAPE")
        print(f"  - 1301 model: {val_mape_1301:.2f}% MAPE")
        print(f"  - New model: {val_mape_new:.2f}% MAPE")
    else:
        print(f"\n✗ The best new model ({best_new_model['model_name']}) shows {-improvement:.1f}% worse validation MAPE")
        print(f"  - 1301 model: {val_mape_1301:.2f}% MAPE")
        print(f"  - New model: {val_mape_new:.2f}% MAPE")

    print("\nAll models ranked by validation MAPE:")
    all_models = [metric_1301] + all_current_metrics
    sorted_models = sorted(all_models, key=lambda x: x['validation']['mape'])
    for i, model in enumerate(sorted_models[:10], 1):
        mape = model['validation']['mape']
        r2 = model['validation']['adj_r2']
        print(f"  {i}. {model['model_name']:<30} MAPE: {mape:>6.2f}%  R²: {r2:>6.4f}")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
