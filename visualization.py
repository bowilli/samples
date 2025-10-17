"""
Visualization and charting utilities.
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

# Suppress numpy warnings for NaN calculations in team visualizations
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')


def create_summary_charts(metrics_df, org_metrics_df=None, team_metrics_df=None):
    """Create summary visualizations for model performance.
    
    org_metrics_df and team_metrics_df should be filtered to only include best models.
    """
    print("\nCreating summary visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create subplots - 2x2 layout for performance + feature importance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. MdAPE comparison by model and target with feature set info (Primary Metric)
    _create_mape_comparison(axes[0, 0], metrics_df)
    
    # 2. MdAE comparison with feature set info (Secondary Metric)
    _create_mdae_comparison(axes[0, 1], metrics_df)
    

    # Get model information dynamically from the data - select best models using MAPE-based selection
    try:
        from model_evaluation import get_best_models_by_target
    except ImportError:
        from utils.model_evaluation import get_best_models_by_target
    
    # Use shared MAPE-based model selection logic
    best_models_dict = get_best_models_by_target(verbose=False)
    
    # 3. SHAP Feature Importance for ATL Model (with fallback to regular feature importance)
    best_atl = best_models_dict.get('atl_efc_usd')
    if best_atl:
        atl_model_key = best_atl['model_key']
        atl_model_info = f"ATL Model: {best_atl['model'].title()} ({best_atl['feature_set']} - {best_atl.get('feature_set_description', 'N/A')})"
        print(f"Selected ATL model: {atl_model_key} with MAPE = {best_atl.get('mdape', 'N/A'):.1f}%")
        _create_feature_importance_plot(axes[1, 0], atl_model_key, atl_model_info, 'steelblue')
    
    # 4. SHAP Feature Importance for Director Model (with fallback to regular feature importance)
    best_director = best_models_dict.get('director_efc_usd')
    if best_director:
        director_model_key = best_director['model_key']
        director_model_info = f"Director Model: {best_director['model'].title()} ({best_director['feature_set']} - {best_director.get('feature_set_description', 'N/A')})"
        print(f"Selected Director model: {director_model_key} with MAPE = {best_director.get('mdape', 'N/A'):.1f}%")
        _create_feature_importance_plot(axes[1, 1], director_model_key, director_model_info, 'orange')
    
    plt.tight_layout()
    
    # Save the figure (renamed to avoid conflict with main 4-metric summary)
    #chart_path = os.path.join('data', 'model_outputs', 'visuals','model_performance_with_feature_importance.png')
    #plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    #print(f"Saved performance chart with feature importance to {chart_path}")
    #plt.close()
    
    # Create additional heatmap for all combinations - DISABLED per user request
    # _create_performance_heatmap(metrics_df)
    
    # Create organization-level visualization if data available
    if org_metrics_df is not None and not org_metrics_df.empty:
        _create_organization_charts(org_metrics_df)
    
    # Create team-level visualization if data available
    if team_metrics_df is not None and not team_metrics_df.empty:
        _create_team_charts(team_metrics_df)

        # Note: Detailed team visualizations and model comparison chart removed


def _create_mape_comparison(ax, metrics_df):
    """Create MdAPE (Median Absolute Percentage Error) comparison chart."""
    # Create custom labels that include feature set information
    custom_labels = []
    mape_values = []
    colors = []
    
    for _, row in metrics_df.iterrows():
        if row['target'] == 'atl_efc_usd':
            label = f"ATL (set_{row['feature_set'][-1]})"
            color = 'steelblue'
        else:
            label = f"Director (set_{row['feature_set'][-1]})"
            color = 'orange'
        custom_labels.append(label)
        # Use MAPE if available, fallback to calculated value or NaN
        mape_value = row.get('mdape', float('nan'))
        mape_values.append(mape_value)
        colors.append(color)
    
    bars = ax.bar(custom_labels, mape_values, color=colors)
    ax.set_title('Best Model MdAPE Performance by Target\n(Lower is Better - Primary Metric)')
    ax.set_ylabel('MdAPE (%)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mape_values):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_values) * 0.02,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    'N/A', ha='center', va='bottom', fontweight='bold')


def _create_mdae_comparison(ax, metrics_df):
    """Create MdAE (Median Absolute Error) comparison chart."""
    custom_labels = []
    mdae_values = []
    colors = []
    
    for _, row in metrics_df.iterrows():
        if row['target'] == 'atl_efc_usd':
            label = f"ATL (set_{row['feature_set'][-1]})"
            color = 'steelblue'
        else:
            label = f"Director (set_{row['feature_set'][-1]})"
            color = 'orange'
        custom_labels.append(label)
        # Use MAE as approximation for MdAE if not available
        mdae_value = row.get('mdae', float('nan'))
        mdae_values.append(mdae_value)
        colors.append(color)
    
    bars = ax.bar(custom_labels, mdae_values, color=colors)
    ax.set_title('Best Model MdAE Performance by Target\n(Lower is Better - Secondary Metric)')
    ax.set_ylabel('MdAE (USD)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars (formatted for readability)
    for bar, value in zip(bars, mdae_values):
        if not np.isnan(value):
            label = f'${value:,.0f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max([v for v in mdae_values if not np.isnan(v)]) * 0.02,
                    label, ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
                    'N/A', ha='center', va='bottom', fontweight='bold')


def _create_feature_importance_plot(ax, model_key, model_info, color):
    """Create feature importance plot with SHAP fallback."""
    shap_file = f'data/model_outputs/visuals/SHAP_visuals/SHAP_importance_{model_key}.png'
    shap_loaded = False
    
    if os.path.exists(shap_file):
        try:
            # Load and display the SHAP image
            shap_img = mpimg.imread(shap_file)
            ax.imshow(shap_img)
            ax.axis('off')  # Remove axes for cleaner look
            ax.set_title(f'{model_info}\nSHAP Feature Importance')
            shap_loaded = True
            print(f"Loaded SHAP visualization: {shap_file}")
        except Exception as e:
            print(f"Could not load SHAP image: {e}")
            pass  # Will fall back to feature importance
    
    # Fallback to regular feature importance if SHAP not available
    if not shap_loaded:
        importance_file = f'data/model_outputs/feature_importance_setsfeature_importances_{model_key}.csv'
        if os.path.exists(importance_file):
            fi_df = pd.read_csv(importance_file)
            top_features = fi_df.head(6)  # Top 6 features
            bars = ax.barh(top_features['feature'], top_features['importance'], color=color, alpha=0.7)
            ax.set_title(f'{model_info}\nTop Feature Importance')
            ax.set_xlabel('Importance (%)')
            ax.invert_yaxis()  # Show highest importance at top
            
            # Add value labels
            for bar, value in zip(bars, top_features['importance']):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='left', va='center', fontweight='bold')
            print(f"Using fallback feature importance: {importance_file}")
        else:
            ax.text(0.5, 0.5, 'Feature importance data\nnot available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{model_info}\nFeature Importance')
            print(f"No feature importance data available for {model_key}")


def _create_performance_heatmap(metrics_df):
    """Create performance heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = metrics_df.pivot_table(
        values='adjusted_r2_score',
        index=['target', 'feature_set'],
        columns='model'
    )
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'R² Score'}, ax=ax)
    ax.set_title('Model Performance Heatmap: R² Scores')
    
    # Save heatmap
    heatmap_path = os.path.join('data', 'model_outputs', 'visuals', 'model_performance_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_path}")
    plt.close()


def _create_organization_charts(org_metrics_df):
    """Create separate organization-level performance charts for each model showing top 5 and bottom 5 with R², RMSE, and MAE."""
    _create_atl_organization_charts(org_metrics_df)
    _create_director_organization_charts(org_metrics_df)


def _create_atl_organization_charts(org_metrics_df):
    """Create ATL organization performance charts with top 5 and bottom 5, including R², RMSE, and MAE."""
    # Separate data by target
    atl_org_data = org_metrics_df[org_metrics_df['target'] == 'atl_efc_usd']
    
    if atl_org_data.empty:
        return
    
    # Extract model information dynamically from the data
    first_row = atl_org_data.iloc[0]
    model_name = first_row['model'].title()
    feature_set = first_row['feature_set']
    feature_description = first_row.get('feature_set_description', '')
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    title = f'ATL Model: {model_name} - {feature_description} - Organization Performance Analysis'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # Get organization stats with all metrics
    atl_org_stats = atl_org_data.groupby('gravity_buying_organization_desc').agg({
        'mdape': 'first',
        'mdae': 'first',
        'rmse': 'first', 
        'adjusted_r2_score': 'first',
        'count': 'first'
    })
    
    # Filter out organizations with NaN MAPE scores for proper ranking, fallback to R² if needed
    valid_mape_orgs = atl_org_stats.dropna(subset=['mdape'])
    
    if valid_mape_orgs.empty:
        print("Warning: No ATL organizations have valid MAPE scores - falling back to R² ranking")
        valid_orgs = atl_org_stats.dropna(subset=['adjusted_r2_score'])
        if valid_orgs.empty:
            print("Warning: No ATL organizations have valid performance scores - charts may be empty")
            valid_orgs = atl_org_stats
        # Use R² ranking (higher is better)
        top_5_orgs = valid_orgs.sort_values('adjusted_r2_score', ascending=False).head(5)
        bottom_5_orgs = valid_orgs.sort_values('adjusted_r2_score', ascending=True).head(5)
    else:
        # Use MAPE ranking (lower is better)
        top_5_orgs = valid_mape_orgs.sort_values('mdape', ascending=True).head(5)  # Lowest MAPE = best
        bottom_5_orgs = valid_mape_orgs.sort_values('mdape', ascending=False).head(5)  # Highest MAPE = worst
    
    # Remove any overlap between top 5 and bottom 5
    bottom_5_orgs = bottom_5_orgs[~bottom_5_orgs.index.isin(top_5_orgs.index)]
    
    # Create charts for top performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'skyblue'), 
                                               ('mdae', 'MdAE ($)', 'lightgreen'),
                                               ('rmse', 'RMSE ($)', 'lightcoral')]):
        # For "lower is better" metrics, reverse order so best performers appear at top
        display_orgs = top_5_orgs.iloc[::-1] if metric in ['mdape', 'mdae', 'rmse'] else top_5_orgs
        
        bars = axes[0,i].barh(range(len(display_orgs)), display_orgs[metric], color=color)
        axes[0,i].set_yticks(range(len(display_orgs)))
        
        org_labels = [f"{org}\n(n={int(display_orgs.loc[org, 'count'])})" 
                     for org in display_orgs.index]
        axes[0,i].set_yticklabels(org_labels, fontsize=14)
        
        axes[0,i].set_title(f'Top 5 Organizations - {title}', fontweight='bold', fontsize=16)
        axes[0,i].set_xlabel(title)
        axes[0,i].set_ylabel('Organization')
        axes[0,i].grid(True, alpha=0.3, axis='x')
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, display_orgs[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            axes[0,i].text(bar.get_width() + (max(display_orgs[metric]) * 0.01), 
                          bar.get_y() + bar.get_height()/2,
                          label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    
    # Create charts for bottom performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'red'), 
                                               ('mdae', 'MdAE ($)', 'orange'),
                                               ('rmse', 'RMSE ($)', 'darkred')]):
        bars = axes[1,i].barh(range(len(bottom_5_orgs)), bottom_5_orgs[metric], color=color)
        axes[1,i].set_yticks(range(len(bottom_5_orgs)))
        
        org_labels = [f"{org}\n(n={int(bottom_5_orgs.loc[org, 'count'])})" 
                     for org in bottom_5_orgs.index]
        axes[1,i].set_yticklabels(org_labels, fontsize=14)
        
        axes[1,i].set_title(f'Bottom 5 Organizations - {title}', fontweight='bold', fontsize=16)
        axes[1,i].set_xlabel(title)
        axes[1,i].set_ylabel('Organization')
        axes[1,i].grid(True, alpha=0.3, axis='x')
        
        if metric == 'adjusted_r2_score':
            axes[1,i].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, bottom_5_orgs[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            
            if metric == 'adjusted_r2_score' and value < 0:
                axes[1,i].text(bar.get_width() - (abs(min(bottom_5_orgs[metric])) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='right', va='center', fontweight='bold', fontsize=16)
            else:
                axes[1,i].text(bar.get_width() + (max(bottom_5_orgs[metric]) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    atl_org_chart_path = os.path.join('data', 'model_outputs', 'visuals', 'atl_organization_performance_charts.png')
    plt.savefig(atl_org_chart_path, dpi=300, bbox_inches='tight')
    print(f"Saved ATL organization charts to {atl_org_chart_path}")
    plt.close()


def _create_director_organization_charts(org_metrics_df):
    """Create Director organization performance charts with top 5 and bottom 5, including R², RMSE, and MAE."""
    # Separate data by target
    director_org_data = org_metrics_df[org_metrics_df['target'] == 'director_efc_usd']
    
    if director_org_data.empty:
        return
    
    # Extract model information dynamically from the data
    first_row = director_org_data.iloc[0]
    model_name = first_row['model'].title()
    feature_set = first_row['feature_set']
    feature_description = first_row.get('feature_set_description', '')
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    title = f'Director Model: {model_name} - {feature_description} - Organization Performance Analysis'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # Get organization stats with all metrics
    director_org_stats = director_org_data.groupby('gravity_buying_organization_desc').agg({
        'mdape': 'first',
        'mdae': 'first', 
        'rmse': 'first',
        'adjusted_r2_score': 'first',
        'count': 'first'
    })
    
    # Filter out organizations with NaN MAPE scores for proper ranking, fallback to R² if needed
    valid_mape_orgs = director_org_stats.dropna(subset=['mdape'])
    
    if valid_mape_orgs.empty:
        print("Warning: No director organizations have valid MAPE scores - falling back to R² ranking")
        valid_orgs = director_org_stats.dropna(subset=['adjusted_r2_score'])
        if valid_orgs.empty:
            print("Warning: No director organizations have valid performance scores - charts may be empty")
            valid_orgs = director_org_stats
        # Use R² ranking (higher is better)
        top_5_orgs = valid_orgs.sort_values('adjusted_r2_score', ascending=False).head(5)
        bottom_5_orgs = valid_orgs.sort_values('adjusted_r2_score', ascending=True).head(5)
    else:
        # Use MAPE ranking (lower is better)
        top_5_orgs = valid_mape_orgs.sort_values('mdape', ascending=True).head(5)  # Lowest MAPE = best
        bottom_5_orgs = valid_mape_orgs.sort_values('mdape', ascending=False).head(5)  # Highest MAPE = worst
    
    # Remove any overlap between top 5 and bottom 5
    bottom_5_orgs = bottom_5_orgs[~bottom_5_orgs.index.isin(top_5_orgs.index)]
    
    # Create charts for top performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'skyblue'), 
                                               ('mdae', 'MdAE ($)', 'lightgreen'),
                                               ('rmse', 'RMSE ($)', 'lightcoral')]):
        # For "lower is better" metrics, reverse order so best performers appear at top
        display_orgs = top_5_orgs.iloc[::-1] if metric in ['mdape', 'mdae', 'rmse'] else top_5_orgs
        
        bars = axes[0,i].barh(range(len(display_orgs)), display_orgs[metric], color=color)
        axes[0,i].set_yticks(range(len(display_orgs)))
        
        org_labels = [f"{org}\n(n={int(display_orgs.loc[org, 'count'])})" 
                     for org in display_orgs.index]
        axes[0,i].set_yticklabels(org_labels, fontsize=14)
        
        axes[0,i].set_title(f'Top 5 Organizations - {title}', fontweight='bold', fontsize=16)
        axes[0,i].set_xlabel(title)
        axes[0,i].set_ylabel('Organization')
        axes[0,i].grid(True, alpha=0.3, axis='x')
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, display_orgs[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            axes[0,i].text(bar.get_width() + (max(display_orgs[metric]) * 0.01), 
                          bar.get_y() + bar.get_height()/2,
                          label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    
    # Create charts for bottom performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'red'), 
                                               ('mdae', 'MdAE ($)', 'orange'),
                                               ('rmse', 'RMSE ($)', 'darkred')]):
        bars = axes[1,i].barh(range(len(bottom_5_orgs)), bottom_5_orgs[metric], color=color)
        axes[1,i].set_yticks(range(len(bottom_5_orgs)))
        
        org_labels = [f"{org}\n(n={int(bottom_5_orgs.loc[org, 'count'])})" 
                     for org in bottom_5_orgs.index]
        axes[1,i].set_yticklabels(org_labels, fontsize=14)
        
        axes[1,i].set_title(f'Bottom 5 Organizations - {title}', fontweight='bold', fontsize=16)
        axes[1,i].set_xlabel(title)
        axes[1,i].set_ylabel('Organization')
        axes[1,i].grid(True, alpha=0.3, axis='x')
        

        if metric == 'adjusted_r2_score':
            axes[1,i].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, bottom_5_orgs[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            
            if metric == 'adjusted_r2_score' and value < 0:
                axes[1,i].text(bar.get_width() - (abs(min(bottom_5_orgs[metric])) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='right', va='center', fontweight='bold', fontsize=16)
            else:
                axes[1,i].text(bar.get_width() + (max(bottom_5_orgs[metric]) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    plt.tight_layout()
    director_org_chart_path = os.path.join('data', 'model_outputs', 'visuals', 'director_organization_performance_charts.png')
    plt.savefig(director_org_chart_path, dpi=300, bbox_inches='tight')
    print(f"Saved Director organization charts to {director_org_chart_path}")
    plt.close()


def _create_team_charts(team_metrics_df):
    """Create separate team-level performance charts for each model showing top 5 and bottom 5 with R², RMSE, and MAE."""
    _create_atl_team_charts(team_metrics_df)
    _create_director_team_charts(team_metrics_df)


def _create_atl_team_charts(team_metrics_df):
    """Create ATL team performance charts with top 5 and bottom 5, including R², RMSE, and MAE."""
    # Separate data by target
    atl_team_data = team_metrics_df[team_metrics_df['target'] == 'atl_efc_usd']
    
    if atl_team_data.empty:
        return
    
    # Extract model information dynamically from the data
    first_row = atl_team_data.iloc[0]
    model_name = first_row['model'].title()
    feature_set = first_row['feature_set']
    feature_description = first_row.get('feature_set_description', '')
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    title = f'ATL Model: {model_name} - {feature_description} - Team Performance Analysis'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # Get team stats with all metrics
    atl_team_stats = atl_team_data.groupby('gravity_buying_team').agg({
        'mdape': 'first',
        'mdae': 'first',
        'rmse': 'first', 
        'adjusted_r2_score': 'first',
        'count': 'first'
    })
    
    # Filter out teams with insufficient sample size (n < 3) for statistical reliability
    min_sample_size = 3
    atl_team_stats = atl_team_stats[atl_team_stats['count'] >= min_sample_size]
    
    if atl_team_stats.empty:
        print(f"Warning: No ATL teams have sufficient sample size (n >= {min_sample_size}) - charts will be empty")
        return
    
    # Filter teams with NaN MAPE scores for proper ranking, fallback to R² if needed
    valid_mape_teams = atl_team_stats.dropna(subset=['mdape'])
    
    if valid_mape_teams.empty:
        print("Warning: No ATL teams have valid MAPE scores - falling back to R² ranking")
        valid_teams = atl_team_stats.dropna(subset=['adjusted_r2_score'])
        if valid_teams.empty:
            print("Warning: No ATL teams have valid performance scores - charts may be empty")
            valid_teams = atl_team_stats
        # Use R² ranking (higher is better)
        top_5_teams = valid_teams.sort_values('adjusted_r2_score', ascending=False).head(5)
        bottom_5_teams = valid_teams.sort_values('adjusted_r2_score', ascending=True).head(5)
    else:
        # Use MAPE ranking (lower is better)
        top_5_teams = valid_mape_teams.sort_values('mdape', ascending=True).head(5)  # Lowest MAPE = best
        bottom_5_teams = valid_mape_teams.sort_values('mdape', ascending=False).head(5)  # Highest MAPE = worst
    
    # Remove any overlap between top 5 and bottom 5
    bottom_5_teams = bottom_5_teams[~bottom_5_teams.index.isin(top_5_teams.index)]
    
    # Create charts for top performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'skyblue'), 
                                               ('mdae', 'MdAE ($)', 'lightgreen'),
                                               ('rmse', 'RMSE ($)', 'lightcoral')]):
        # For "lower is better" metrics, reverse order so best performers appear at top
        display_teams = top_5_teams.iloc[::-1] if metric in ['mdape', 'mdae', 'rmse'] else top_5_teams
        
        bars = axes[0,i].barh(range(len(display_teams)), display_teams[metric], color=color)
        axes[0,i].set_yticks(range(len(display_teams)))
        
        team_labels = [f"{team}\n(n={int(display_teams.loc[team, 'count'])})" 
                      for team in display_teams.index]
        axes[0,i].set_yticklabels(team_labels, fontsize=14)
        
        axes[0,i].set_title(f'Top 5 Teams - {title}', fontweight='bold', fontsize=16)
        axes[0,i].set_xlabel(title)
        axes[0,i].set_ylabel('Team')
        axes[0,i].grid(True, alpha=0.3, axis='x')
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, display_teams[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            axes[0,i].text(bar.get_width() + (max(display_teams[metric]) * 0.01), 
                          bar.get_y() + bar.get_height()/2,
                          label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    
    # Create charts for bottom performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'red'), 
                                               ('mdae', 'MdAE ($)', 'orange'),
                                               ('rmse', 'RMSE ($)', 'darkred')]):
        bars = axes[1,i].barh(range(len(bottom_5_teams)), bottom_5_teams[metric], color=color)
        axes[1,i].set_yticks(range(len(bottom_5_teams)))
        
        team_labels = [f"{team}\n(n={int(bottom_5_teams.loc[team, 'count'])})" 
                      for team in bottom_5_teams.index]
        axes[1,i].set_yticklabels(team_labels, fontsize=14)
        
        axes[1,i].set_title(f'Bottom 5 Teams - {title}', fontweight='bold', fontsize=16)
        axes[1,i].set_xlabel(title)
        axes[1,i].set_ylabel('Team')
        axes[1,i].grid(True, alpha=0.3, axis='x')
        
        if metric == 'adjusted_r2_score':
            axes[1,i].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, bottom_5_teams[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            
            if metric == 'adjusted_r2_score' and value < 0:
                axes[1,i].text(bar.get_width() - (abs(min(bottom_5_teams[metric])) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='right', va='center', fontweight='bold', fontsize=16)
            else:
                axes[1,i].text(bar.get_width() + (max(bottom_5_teams[metric]) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    atl_chart_path = os.path.join('data', 'model_outputs', 'visuals', 'atl_team_performance_charts.png')
    plt.savefig(atl_chart_path, dpi=300, bbox_inches='tight')
    print(f"Saved ATL team charts to {atl_chart_path}")
    plt.close()


def _create_director_team_charts(team_metrics_df):
    """Create Director team performance charts with top 5 and bottom 5, including R², RMSE, and MAE."""
    # Separate data by target
    director_team_data = team_metrics_df[team_metrics_df['target'] == 'director_efc_usd']

    if director_team_data.empty:
        print("Warning: No director team data available - skipping director team charts")
        return
    
    # Extract model information dynamically from the data
    first_row = director_team_data.iloc[0]
    model_name = first_row['model'].title()
    feature_set = first_row['feature_set']
    feature_description = first_row.get('feature_set_description', '')
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    title = f'Director Model: {model_name} - {feature_description} - Team Performance Analysis'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # Get team stats with all metrics
    director_team_stats = director_team_data.groupby('gravity_buying_team').agg({
        'mdape': 'first',
        'mdae': 'first',
        'rmse': 'first',
        'adjusted_r2_score': 'first', 
        'count': 'first'
    })
    
    # Filter out teams with insufficient sample size (n < 3) for statistical reliability
    min_sample_size = 3
    director_team_stats = director_team_stats[director_team_stats['count'] >= min_sample_size]
    
    if director_team_stats.empty:
        print(f"Warning: No Director teams have sufficient sample size (n >= {min_sample_size}) - charts will be empty")
        return
    
    # Filter out teams with NaN MAPE scores for proper ranking, fallback to R² if needed
    valid_mape_teams = director_team_stats.dropna(subset=['mdape'])
    
    if valid_mape_teams.empty:
        print("Warning: No director teams have valid MAPE scores - falling back to R² ranking")
        valid_teams = director_team_stats.dropna(subset=['adjusted_r2_score'])
        if valid_teams.empty:
            print("Warning: No director teams have valid performance scores - charts may be empty")
            valid_teams = director_team_stats
        # Use R² ranking (higher is better)
        top_5_teams = valid_teams.sort_values('adjusted_r2_score', ascending=False).head(5)
        bottom_5_teams = valid_teams.sort_values('adjusted_r2_score', ascending=True).head(5)
    else:
        # Use MAPE ranking (lower is better)
        top_5_teams = valid_mape_teams.sort_values('mdape', ascending=True).head(5)  # Lowest MAPE = best
        bottom_5_teams = valid_mape_teams.sort_values('mdape', ascending=False).head(5)  # Highest MAPE = worst
    
    # Remove any overlap between top 5 and bottom 5
    bottom_5_teams = bottom_5_teams[~bottom_5_teams.index.isin(top_5_teams.index)]
    
    # Create charts for top performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'skyblue'), 
                                               ('mdae', 'MdAE ($)', 'lightgreen'),
                                               ('rmse', 'RMSE ($)', 'lightcoral')]):
        # For "lower is better" metrics, reverse order so best performers appear at top
        display_teams = top_5_teams.iloc[::-1] if metric in ['mdape', 'mdae', 'rmse'] else top_5_teams
        
        bars = axes[0,i].barh(range(len(display_teams)), display_teams[metric], color=color)
        axes[0,i].set_yticks(range(len(display_teams)))
        
        team_labels = [f"{team}\n(n={int(display_teams.loc[team, 'count'])})" 
                      for team in display_teams.index]
        axes[0,i].set_yticklabels(team_labels, fontsize=14)
        
        axes[0,i].set_title(f'Top 5 Teams - {title}', fontweight='bold', fontsize=16)
        axes[0,i].set_xlabel(title)
        axes[0,i].set_ylabel('Team')
        axes[0,i].grid(True, alpha=0.3, axis='x')
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, display_teams[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            axes[0,i].text(bar.get_width() + (max(display_teams[metric]) * 0.01), 
                          bar.get_y() + bar.get_height()/2,
                          label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    
    # Create charts for bottom performers
    for i, (metric, title, color) in enumerate([('mdape', 'MdAPE (%)', 'red'), 
                                               ('mdae', 'MdAE ($)', 'orange'),
                                               ('rmse', 'RMSE ($)', 'darkred')]):
        bars = axes[1,i].barh(range(len(bottom_5_teams)), bottom_5_teams[metric], color=color)
        axes[1,i].set_yticks(range(len(bottom_5_teams)))
        
        team_labels = [f"{team}\n(n={int(bottom_5_teams.loc[team, 'count'])})" 
                      for team in bottom_5_teams.index]
        axes[1,i].set_yticklabels(team_labels, fontsize=14)
        
        axes[1,i].set_title(f'Bottom 5 Teams - {title}', fontweight='bold', fontsize=16)
        axes[1,i].set_xlabel(title)
        axes[1,i].set_ylabel('Team')
        axes[1,i].grid(True, alpha=0.3, axis='x')
        
        if metric == 'adjusted_r2_score':
            axes[1,i].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels with appropriate formatting
        for j, (bar, value) in enumerate(zip(bars, bottom_5_teams[metric])):
            if metric == 'mdape':
                label_text = f'{value:.1f}%'
            elif metric in ['mdae', 'rmse']:
                label_text = f'${value:,.0f}'
            else:
                label_text = f'{value:.3f}'
            
            if metric == 'adjusted_r2_score' and value < 0:
                axes[1,i].text(bar.get_width() - (abs(min(bottom_5_teams[metric])) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='right', va='center', fontweight='bold', fontsize=16)
            else:
                axes[1,i].text(bar.get_width() + (max(bottom_5_teams[metric]) * 0.01), 
                              bar.get_y() + bar.get_height()/2,
                              label_text, ha='left', va='center', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    director_chart_path = os.path.join('data', 'model_outputs', 'visuals', 'director_team_performance_charts.png')
    plt.savefig(director_chart_path, dpi=300, bbox_inches='tight')
    print(f"Saved Director team charts to {director_chart_path}")
    plt.close()
