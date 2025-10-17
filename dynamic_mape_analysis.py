#!/usr/bin/env python3
"""
Dynamic Model MAPE Analysis Script
=================================

This script automatically identifies the top 2 performing models from any model run
and generates comprehensive MAPE (Mean Absolute Percentage Error) analysis.

Features:
- Auto-detects top performing models by R² score
- Works with any model output directory structure
- Generates comprehensive MAPE breakdowns by organization and buying team
- Calculates percentage of titles with MAPE below 6% and 20% thresholds
- Outputs plain text report with detailed title counts
- Can be integrated into model training pipelines
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse
import sys
try:
    from utils.adjusted_r2 import calculate_adjusted_r2_from_r2, estimate_feature_count_from_context
except ImportError:
    from adjusted_r2 import calculate_adjusted_r2_from_r2, estimate_feature_count_from_context

def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error
    MAPE = mean(|actual - predicted| / |actual|) * 100
    
    Handles zero actual values by excluding them from calculation
    """
    # Convert to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove rows where actual is zero to avoid division by zero
    non_zero_mask = actual != 0
    actual_filtered = actual[non_zero_mask]
    predicted_filtered = predicted[non_zero_mask]
    
    if len(actual_filtered) == 0:
        return np.nan
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
    return mape

def calculate_mape_thresholds(actual, predicted, thresholds=[6, 20]):
    """
    Calculate percentage of predictions within MAPE thresholds
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove rows where actual is zero
    non_zero_mask = actual != 0
    actual_filtered = actual[non_zero_mask]
    predicted_filtered = predicted[non_zero_mask]
    
    if len(actual_filtered) == 0:
        return {f"below_{t}%": 0.0 for t in thresholds}
    
    # Calculate individual MAPE for each prediction
    individual_mape = np.abs((actual_filtered - predicted_filtered) / actual_filtered) * 100
    
    results = {}
    for threshold in thresholds:
        pct_below = (individual_mape < threshold).mean() * 100
        results[f"below_{threshold}%"] = pct_below
    
    return results

def calculate_adjusted_r2_for_model(model_info):
    """Calculate adjusted R² for a model based on its info."""
    r2 = model_info['r2_score']
    n_samples = model_info['n_train'] + model_info['n_test']
    model_key = model_info['model_key']
    n_features = estimate_feature_count_from_context(model_key, n_samples)
    
    return calculate_adjusted_r2_from_r2(r2, n_samples, n_features)

def identify_top_models(model_metrics_file, top_n=2):
    """
    Automatically identify top N performing models by adjusted R² score
    Returns list of model configurations for analysis
    """
    print(f"📊 Identifying top {top_n} models from {model_metrics_file}")
    
    # Read model metrics
    df = pd.read_csv(model_metrics_file)
    
    # Calculate adjusted R² for all models
    df['adjusted_r2'] = df.apply(lambda row: calculate_adjusted_r2_for_model(row), axis=1)
    
    # Sort by adjusted R² score (descending) to get top performers
    df_sorted = df.sort_values('adjusted_r2', ascending=False)
    
    # Get top models for each target type
    atl_models = df_sorted[df_sorted['target'] == 'atl_efc_usd'].head(1)
    director_models = df_sorted[df_sorted['target'] == 'director_efc_usd'].head(1)
    
    top_models = []
    
    # Add top ATL model
    if not atl_models.empty:
        atl_model = atl_models.iloc[0]
        top_models.append({
            'file': f"predictions_{atl_model['model_key']}.csv",
            'name': f"ATL {atl_model['model'].title()} {atl_model['feature_set'].title()} ({atl_model['feature_set_description']})",
            'type': 'ATL',
            'r2': atl_model['adjusted_r2'],
            'model_key': atl_model['model_key']
        })
        print(f"✅ Top ATL Model: {atl_model['model_key']} (Adjusted R² = {atl_model['adjusted_r2']:.3f})")
    
    # Add top Director model
    if not director_models.empty:
        director_model = director_models.iloc[0]
        top_models.append({
            'file': f"predictions_{director_model['model_key']}.csv",
            'name': f"Director {director_model['model'].title()} {director_model['feature_set'].title()} ({director_model['feature_set_description']})",
            'type': 'Director',
            'r2': director_model['adjusted_r2'],
            'model_key': director_model['model_key']
        })
        print(f"✅ Top Director Model: {director_model['model_key']} (Adjusted R² = {director_model['adjusted_r2']:.3f})")
    
    print(f"🎯 Selected {len(top_models)} models for MAPE analysis\n")
    return top_models

def analyze_model_mape(predictions_file, model_name, model_type, r2_score):
    """
    Analyze MAPE for a single model
    """
    print(f"{'='*60}")
    print(f"ANALYZING: {model_name}")
    print(f"File: {predictions_file}")
    print(f"{'='*60}")
    
    # Load predictions
    if not os.path.exists(predictions_file):
        print(f"❌ WARNING: File not found: {predictions_file}")
        return None, None, None
    
    df = pd.read_csv(predictions_file)
    print(f"Loaded {len(df)} predictions")
    
    # Overall MAPE
    overall_mape = calculate_mape(df['actual'], df['prediction'])
    threshold_results = calculate_mape_thresholds(df['actual'], df['prediction'])
    
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'total_predictions': len(df),
        'overall_mape': overall_mape,
        'r2_score': r2_score,
        **threshold_results
    }
    
    print(f"Overall MAPE: {overall_mape:.2f}%")
    print(f"Titles with MAPE < 6%: {threshold_results['below_6%']:.1f}%")
    print(f"Titles with MAPE < 20%: {threshold_results['below_20%']:.1f}%")
    
    # Organization breakdown
    print(f"\n--- MAPE BY ORGANIZATION ---")
    org_results = []
    for org in df['gravity_buying_organization_desc'].unique():
        org_data = df[df['gravity_buying_organization_desc'] == org]
        org_mape = calculate_mape(org_data['actual'], org_data['prediction'])
        org_thresholds = calculate_mape_thresholds(org_data['actual'], org_data['prediction'])
        
        org_result = {
            'organization': org,
            'count': len(org_data),
            'mape': org_mape,
            **org_thresholds
        }
        org_results.append(org_result)
        
        print(f"{org} (n={len(org_data)}): {org_mape:.2f}% MAPE, "
              f"{org_thresholds['below_6%']:.1f}% <6%, {org_thresholds['below_20%']:.1f}% <20%")
    
    # Team breakdown  
    print(f"\n--- MAPE BY BUYING TEAM ---")
    team_results = []
    for team in df['gravity_buying_team'].unique():
        team_data = df[df['gravity_buying_team'] == team]
        team_mape = calculate_mape(team_data['actual'], team_data['prediction'])
        team_thresholds = calculate_mape_thresholds(team_data['actual'], team_data['prediction'])
        
        team_result = {
            'buying_team': team,
            'count': len(team_data),
            'mape': team_mape,
            **team_thresholds
        }
        team_results.append(team_result)
        
        if len(team_data) >= 3:  # Only print teams with sufficient sample size
            print(f"{team} (n={len(team_data)}): {team_mape:.2f}% MAPE, "
                  f"{team_thresholds['below_6%']:.1f}% <6%, {team_thresholds['below_20%']:.1f}% <20%")
    
    return results, org_results, team_results

def generate_report(results, org_results, team_results, output_path, run_timestamp=None):
    """
    Generate comprehensive MAPE analysis report in plain text format
    """
    with open(output_path, 'w') as f:
        f.write("DYNAMIC MODEL MAPE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        if run_timestamp:
            f.write(f"Model Run: {run_timestamp}\n")
        f.write(f"Analysis Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report provides comprehensive MAPE (Mean Absolute Percentage Error) analysis ")
        f.write("for the top performing models from the current model run, automatically identified by R² score.\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 17 + "\n\n")
        
        total_titles = 0
        for result in results:
            if result is None:
                continue
            total_titles += result['total_predictions']
            f.write(f"{result['model_type'].upper()} MODEL:\n")
            f.write(f"  Model Name: {result['model_name']}\n")
            f.write(f"  Adjusted R² Score: {result.get('r2_score', 'N/A')}\n")
            f.write(f"  Total Titles: {result['total_predictions']}\n")
            f.write(f"  Overall MAPE: {result['overall_mape']:.2f}%\n")
            f.write(f"  Excellent Predictions (MAPE < 6%): {result['below_6%']:.1f}% ({int(result['below_6%'] * result['total_predictions'] / 100)} titles)\n")
            f.write(f"  Good Predictions (MAPE < 20%): {result['below_20%']:.1f}% ({int(result['below_20%'] * result['total_predictions'] / 100)} titles)\n")
            f.write(f"  Challenging Predictions (MAPE >= 20%): {100 - result['below_20%']:.1f}% ({int((100 - result['below_20%']) * result['total_predictions'] / 100)} titles)\n\n")
        
        f.write(f"TOTAL TITLES ANALYZED: {total_titles}\n\n")
        
        # Detailed Analysis for Each Model
        for i, result in enumerate(results):
            if result is None:
                continue
            
            org_data = org_results[i] if i < len(org_results) else []
            team_data = team_results[i] if i < len(team_results) else []
            
            f.write(f"{result['model_type'].upper()} MODEL DETAILED ANALYSIS\n")
            f.write("=" * (len(result['model_type']) + 25) + "\n\n")
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"Adjusted R² Score: {result.get('r2_score', 'N/A')}\n") 
            f.write(f"Total Predictions: {result['total_predictions']}\n")
            f.write(f"Overall MAPE: {result['overall_mape']:.2f}%\n\n")
            
            f.write("Performance Thresholds\n")
            f.write("-" * 21 + "\n")
            f.write(f"- Excellent predictions (MAPE < 6%): {result['below_6%']:.1f}% of titles ({int(result['below_6%'] * result['total_predictions'] / 100)} titles)\n")
            f.write(f"- Good predictions (MAPE < 20%): {result['below_20%']:.1f}% of titles ({int(result['below_20%'] * result['total_predictions'] / 100)} titles)\n")
            f.write(f"- Challenging predictions (MAPE >= 20%): {100 - result['below_20%']:.1f}% of titles ({int((100 - result['below_20%']) * result['total_predictions'] / 100)} titles)\n\n")
            
            # Organization breakdown
            f.write("MAPE by Organization\n")
            f.write("-" * 19 + "\n\n")
            f.write(f"{'Organization':<25} {'Count':<6} {'MAPE':<8} {'<6% MAPE':<9} {'<20% MAPE':<10}\n")
            f.write("-" * 65 + "\n")
            
            # Sort by MAPE (best first)
            org_data_sorted = sorted(org_data, key=lambda x: x['mape'] if not pd.isna(x['mape']) else float('inf'))
            
            for org in org_data_sorted:
                f.write(f"{org['organization']:<25} {org['count']:<6} {org['mape']:>7.2f}% {org['below_6%']:>8.1f}% {org['below_20%']:>9.1f}%\n")
            
            f.write("\n")
            
            # Team breakdown
            f.write("MAPE by Buying Team (Teams with 3+ titles)\n")
            f.write("-" * 42 + "\n\n")
            f.write(f"{'Buying Team':<35} {'Count':<6} {'MAPE':<8} {'<6% MAPE':<9} {'<20% MAPE':<10}\n")
            f.write("-" * 75 + "\n")
            
            # Sort by MAPE (best first), but only show teams with reasonable sample sizes
            team_data_filtered = [t for t in team_data if t['count'] >= 3]  # Min 3 samples
            team_data_sorted = sorted(team_data_filtered, key=lambda x: x['mape'] if not pd.isna(x['mape']) else float('inf'))
            
            for team in team_data_sorted:
                f.write(f"{team['buying_team']:<35} {team['count']:<6} {team['mape']:>7.2f}% {team['below_6%']:>8.1f}% {team['below_20%']:>9.1f}%\n")
            
            f.write("\n")
        
        # Business Insights
        f.write("BUSINESS INSIGHTS & RECOMMENDATIONS\n")
        f.write("=" * 35 + "\n\n")
        f.write("Model Performance Assessment\n")
        f.write("-" * 27 + "\n\n")
        
        # Compare models if we have both
        valid_results = [r for r in results if r is not None]
        if len(valid_results) == 2:
            atl_result = next((r for r in valid_results if r['model_type'] == 'ATL'), None)
            director_result = next((r for r in valid_results if r['model_type'] == 'Director'), None)
            
            if atl_result and director_result:
                f.write("Comparative Performance:\n")
                f.write(f"- ATL Model MAPE: {atl_result['overall_mape']:.2f}% ({atl_result['total_predictions']} titles)\n")
                f.write(f"- Director Model MAPE: {director_result['overall_mape']:.2f}% ({director_result['total_predictions']} titles)\n")
                
                better_model = "ATL" if atl_result['overall_mape'] < director_result['overall_mape'] else "Director"
                f.write(f"- {better_model} model shows superior MAPE performance\n\n")
        
        f.write("Deployment Recommendations\n")
        f.write("-" * 25 + "\n\n")
        for result in valid_results:
            mape = result['overall_mape']
            below_20 = result['below_20%']
            
            if mape < 30 and below_20 > 70:
                recommendation = "RECOMMENDED for production deployment"
            elif mape < 50 and below_20 > 50:
                recommendation = "CAUTION - Deploy with error margins and monitoring"
            else:
                recommendation = "NOT RECOMMENDED - Requires model improvement"
            
            f.write(f"{result['model_type'].upper()} Model: {recommendation}\n")
            f.write(f"- Overall MAPE: {mape:.2f}%\n")
            f.write(f"- Total Titles: {result['total_predictions']}\n")
            f.write(f"- Reliable predictions (MAPE < 20%): {below_20:.1f}% ({int(below_20 * result['total_predictions'] / 100)} titles)\n\n")
        
        f.write("Key Findings\n")
        f.write("-" * 12 + "\n\n")
        f.write("1. Accuracy Distribution: Review the percentage of titles achieving excellent (<6% MAPE) ")
        f.write("vs. good (<20% MAPE) vs. challenging (>=20% MAPE) performance levels.\n\n")
        f.write("2. Organizational Patterns: Some organizations may show consistently better or worse ")
        f.write("prediction accuracy, indicating potential data quality or market complexity differences.\n\n")
        f.write("3. Team-Level Insights: Buying teams with sufficient sample sizes (>=3 predictions) ")
        f.write("show varying levels of predictability, useful for targeted model improvements.\n\n")
        
        f.write("-" * 50 + "\n")
        f.write(f"Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}\n")

def main(model_output_path="data/model_outputs", run_timestamp=None):
    """
    Main analysis function that can be called programmatically or as standalone script
    """
    print("🎯 DYNAMIC MODEL MAPE ANALYSIS")
    print("=" * 80)
    
    # Define paths
    model_metrics_file = os.path.join(model_output_path, "model_metrics", "model_accuracy_metrics.csv")
    predictions_path = os.path.join(model_output_path, "predictions")
    output_path = os.path.join(model_output_path, "model_metrics")
    
    # Check if required files exist
    if not os.path.exists(model_metrics_file):
        print(f"❌ ERROR: Model metrics file not found: {model_metrics_file}")
        return False
    
    if not os.path.exists(predictions_path):
        print(f"❌ ERROR: Predictions directory not found: {predictions_path}")
        return False
    
    # Identify top performing models
    top_models = identify_top_models(model_metrics_file, top_n=2)
    
    if not top_models:
        print("❌ ERROR: No models found for analysis")
        return False
    
    # Analyze each model
    all_results = []
    all_org_results = []
    all_team_results = []
    
    for model_config in top_models:
        file_path = os.path.join(predictions_path, model_config['file'])
        
        results, org_results, team_results = analyze_model_mape(
            file_path, 
            model_config['name'],
            model_config['type'],
            model_config['r2']
        )
        
        all_results.append(results)
        all_org_results.append(org_results if org_results else [])
        all_team_results.append(team_results if team_results else [])
    
    # Generate comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_path, f"MAPE_analysis_top_models_{timestamp}.txt")
    
    generate_report(all_results, all_org_results, all_team_results, report_path, run_timestamp)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📊 Report saved to: {report_path}")
    
    return True

if __name__ == "__main__":
    # Command line interface
    parser = argparse.ArgumentParser(description='Dynamic MAPE Analysis for Top Performing Models')
    parser.add_argument('--model_output_path', 
                       default='data/model_outputs',
                       help='Path to model outputs directory (default: data/model_outputs)')
    parser.add_argument('--run_timestamp',
                       help='Optional timestamp identifier for the model run')
    
    args = parser.parse_args()
    
    success = main(args.model_output_path, args.run_timestamp)
    sys.exit(0 if success else 1)
