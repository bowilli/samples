#!/usr/bin/env python3
"""
Generate updated performance charts using actual data from metrics summary files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

def load_actual_data():
    """Load the actual organization and team metrics data."""
    org_file = 'data/model_outputs/model_metrics/organization_metrics_summary.csv'
    team_file = 'data/model_outputs/model_metrics/team_metrics_summary.csv'
    
    if not os.path.exists(org_file) or not os.path.exists(team_file):
        print(f"Error: Data files not found")
        print(f"Organization file exists: {os.path.exists(org_file)}")
        print(f"Team file exists: {os.path.exists(team_file)}")
        return None, None
    
    org_df = pd.read_csv(org_file)
    team_df = pd.read_csv(team_file)
    
    print(f"Loaded organization data: {org_df.shape}")
    print(f"Loaded team data: {team_df.shape}")
    print(f"Organization targets: {org_df['target'].unique()}")
    print(f"Team targets: {team_df['target'].unique()}")
    
    return org_df, team_df

def main():
    """Generate the updated charts using actual data."""
    import sys
    sys.path.append('.')
    from utils.visualization import _create_team_charts, _create_organization_charts
    
    print("Loading actual data...")
    org_df, team_df = load_actual_data()
    
    if org_df is None or team_df is None:
        print("Failed to load data files")
        return
    
    print("Generating organization performance charts...")
    _create_organization_charts(org_df)
    
    print("Generating team performance charts...")
    _create_team_charts(team_df)
    
    print("All charts generated successfully!")
    print("\nGenerated files:")
    files = [
        'data/model_outputs/visuals/atl_team_performance_charts.png',
        'data/model_outputs/visuals/director_team_performance_charts.png',
        'data/model_outputs/visuals/atl_organization_performance_charts.png',
        'data/model_outputs/visuals/director_organization_performance_charts.png'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")

if __name__ == "__main__":
    main()
