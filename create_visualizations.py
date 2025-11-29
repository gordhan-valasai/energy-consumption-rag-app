"""
Comprehensive Visualization Module for Energy Consumption Data

Creates publication-ready figures and charts including:
- Time series plots
- Bar charts (horizontal and vertical)
- Scatter plots
- Heatmaps
- Growth rate visualizations
- Data quality visualizations
- Regional comparisons
- Per-capita analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def load_processed_data(file_path: str = 'output/cleaned_energy_data.csv') -> pd.DataFrame:
    """Load processed energy data."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    elif os.path.exists('output_robust/cleaned_energy_data.csv'):
        return pd.read_csv('output_robust/cleaned_energy_data.csv')
    else:
        raise FileNotFoundError(f"Processed data file not found. Please run the processing pipeline first.")


def plot_time_series_comprehensive(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create comprehensive time series visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Top 10 countries time series
    top_countries = df.groupby('country')['total_energy_twh'].max().nlargest(10).index
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    
    for i, country in enumerate(top_countries):
        country_data = df[df['country'] == country].sort_values('year')
        ax.plot(country_data['year'], country_data['total_energy_twh'], 
                marker='o', label=country, linewidth=2.5, markersize=4, color=colors[i])
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Energy Consumption (TWh)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Consumption Time Series - Top 10 Countries', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', ncol=2, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_time_series_top10.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Growth trends (2000-2024) for selected countries
    if 2000 in df['year'].values:
        growth_countries = df.groupby('country')['total_energy_twh'].max().nlargest(8).index
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, country in enumerate(growth_countries):
            country_data = df[df['country'] == country].sort_values('year')
            country_data = country_data[country_data['year'] >= 2000]
            
            if len(country_data) > 0:
                ax = axes[idx]
                ax.plot(country_data['year'], country_data['total_energy_twh'], 
                       marker='o', linewidth=2, markersize=3, color='steelblue')
                ax.set_title(country, fontsize=10, fontweight='bold')
                ax.set_xlabel('Year', fontsize=9)
                ax.set_ylabel('Energy (TWh)', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        plt.suptitle('Energy Consumption Trends (2000-2024) - Top 8 Countries', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_growth_trends_panel.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Time series plots created")


def plot_bar_charts_comprehensive(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create comprehensive bar chart visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()
    
    # 1. Top 15 countries - horizontal bar
    top_15 = latest_data.nlargest(15, 'total_energy_twh')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
    bars = ax.barh(top_15['country'], top_15['total_energy_twh'], color=colors)
    ax.set_xlabel('Total Energy Consumption (TWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Countries by Energy Consumption ({latest_year})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_15['total_energy_twh'])):
        ax.text(val, i, f' {val:.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_top15_countries_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-capita energy consumption (if available)
    if 'energy_per_capita_twh' in df.columns:
        latest_data_pc = latest_data[latest_data['energy_per_capita_twh'].notna()]
        top_20_pc = latest_data_pc.nlargest(20, 'energy_per_capita_twh')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        colors_pc = plt.cm.plasma(np.linspace(0, 1, len(top_20_pc)))
        bars = ax.barh(top_20_pc['country'], top_20_pc['energy_per_capita_twh'], 
                      color=colors_pc)
        ax.set_xlabel('Energy per Capita (TWh)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top 20 Countries by Energy Consumption per Capita ({latest_year})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_20_pc['energy_per_capita_twh'])):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_per_capita_top20.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Comparison: Total vs Per-Capita (if both available)
    if 'energy_per_capita_twh' in df.columns:
        top_10_total = latest_data.nlargest(10, 'total_energy_twh')
        top_10_pc = latest_data[latest_data['energy_per_capita_twh'].notna()].nlargest(10, 'energy_per_capita_twh')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Total consumption
        ax1.barh(top_10_total['country'], top_10_total['total_energy_twh'], color='steelblue')
        ax1.set_xlabel('Total Energy (TWh)', fontsize=11, fontweight='bold')
        ax1.set_title('Top 10 by Total Consumption', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Per-capita
        ax2.barh(top_10_pc['country'], top_10_pc['energy_per_capita_twh'], color='coral')
        ax2.set_xlabel('Energy per Capita (TWh)', fontsize=11, fontweight='bold')
        ax2.set_title('Top 10 by Per-Capita Consumption', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Energy Consumption Rankings Comparison ({latest_year})', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_total_vs_percapita_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Bar charts created")


def plot_growth_analysis(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create growth rate analysis visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    latest_year = df['year'].max()
    
    if 2000 not in df['year'].values:
        print("⚠ Skipping growth analysis (no 2000 data)")
        return
    
    # Calculate growth rates
    growth_data = []
    for country in df['country'].unique():
        country_data = df[df['country'] == country].sort_values('year')
        year_2000_data = country_data[country_data['year'] == 2000]
        year_latest_data = country_data[country_data['year'] == latest_year]
        
        if len(year_2000_data) > 0 and len(year_latest_data) > 0:
            val_2000 = year_2000_data['total_energy_twh'].iloc[0]
            val_latest = year_latest_data['total_energy_twh'].iloc[0]
            
            if pd.notna(val_2000) and pd.notna(val_latest) and val_2000 > 0:
                years = latest_year - 2000
                growth_rate = ((val_latest / val_2000) ** (1 / years) - 1) * 100
                growth_data.append({
                    'country': country,
                    'growth_rate_pct': growth_rate,
                    'value_2000': val_2000,
                    'value_latest': val_latest,
                    'absolute_change': val_latest - val_2000
                })
    
    if not growth_data:
        print("⚠ No growth data available")
        return
    
    growth_df = pd.DataFrame(growth_data)
    
    # 1. Top and bottom growth rates
    top_growth = growth_df.nlargest(15, 'growth_rate_pct')
    bottom_growth = growth_df.nsmallest(10, 'growth_rate_pct')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top growers
    colors1 = plt.cm.Greens(np.linspace(0.4, 1, len(top_growth)))
    ax1.barh(top_growth['country'], top_growth['growth_rate_pct'], color=colors1)
    ax1.set_xlabel('Annual Growth Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Top 15 Fastest Growing Countries', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    
    # Bottom growers (or decliners)
    colors2 = plt.cm.Reds(np.linspace(0.4, 1, len(bottom_growth)))
    ax2.barh(bottom_growth['country'], bottom_growth['growth_rate_pct'], color=colors2)
    ax2.set_xlabel('Annual Growth Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Slowest Growing / Declining Countries', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Energy Consumption Growth Rates (2000-{latest_year})', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_growth_rates_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Growth vs Absolute Change scatter
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(growth_df['growth_rate_pct'], growth_df['absolute_change'],
                        s=100, alpha=0.6, c=growth_df['value_latest'], 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Label top countries
    top_5_growth = growth_df.nlargest(5, 'growth_rate_pct')
    for _, row in top_5_growth.iterrows():
        ax.annotate(row['country'], 
                   (row['growth_rate_pct'], row['absolute_change']),
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Annual Growth Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Change (TWh)', fontsize=12, fontweight='bold')
    ax.set_title('Growth Rate vs Absolute Change in Energy Consumption', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Latest Value (TWh)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_growth_vs_absolute_change.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Growth analysis plots created")


def plot_data_quality_visualization(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create data quality visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'data_quality_flag' not in df.columns:
        print("⚠ No data quality flags found")
        return
    
    # 1. Data quality flag distribution
    quality_labels = {
        0: 'Original',
        1: 'Interpolated',
        2: 'Missing Block',
        3: 'Removed Outlier'
    }
    
    quality_counts = df['data_quality_flag'].value_counts().sort_index()
    quality_names = [quality_labels.get(i, f'Flag {i}') for i in quality_counts.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_quality = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars = ax.bar(quality_names, quality_counts.values, color=colors_quality[:len(quality_names)])
    ax.set_ylabel('Number of Data Points', fontsize=12, fontweight='bold')
    ax.set_title('Data Quality Flag Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, quality_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val}\n({val/len(df)*100:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_data_quality_flags.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Missing data by country
    missing_by_country = df.groupby('country').agg({
        'total_energy_twh': lambda x: x.isna().sum()
    }).rename(columns={'total_energy_twh': 'missing_count'})
    missing_by_country = missing_by_country[missing_by_country['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if len(missing_by_country) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_missing = missing_by_country.head(15)
        ax.barh(top_missing.index, top_missing['missing_count'], color='orange')
        ax.set_xlabel('Number of Missing Years', fontsize=12, fontweight='bold')
        ax.set_title('Countries with Missing Data', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_missing_data_by_country.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Data quality visualizations created")


def plot_heatmap_visualizations(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create heatmap visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pivot table: countries x years
    pivot_data = df.pivot_table(
        values='total_energy_twh',
        index='country',
        columns='year',
        aggfunc='mean'
    )
    
    # Select top 20 countries by latest year
    latest_year = df['year'].max()
    if latest_year in pivot_data.columns:
        top_countries = pivot_data.nlargest(20, latest_year).index
        pivot_subset = pivot_data.loc[top_countries]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot_subset, annot=False, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Energy Consumption (TWh)'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        ax.set_title('Energy Consumption Heatmap: Top 20 Countries Over Time', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_consumption_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Heatmap visualizations created")


def plot_correlation_analysis(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create correlation analysis visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()
    
    # Prepare numeric columns for correlation
    numeric_cols = ['total_energy_twh']
    if 'energy_per_capita_twh' in latest_data.columns:
        numeric_cols.append('energy_per_capita_twh')
    if 'population' in latest_data.columns:
        numeric_cols.append('population')
    if 'gdp' in latest_data.columns:
        numeric_cols.append('gdp')
    
    if len(numeric_cols) > 1:
        corr_data = latest_data[numeric_cols].dropna()
        if len(corr_data) > 0:
            correlation_matrix = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       vmin=-1, vmax=1, ax=ax)
            ax.set_title(f'Correlation Matrix - Energy Consumption Variables ({latest_year})', 
                        fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/11_correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("✓ Correlation analysis created")


def plot_distribution_analysis(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create distribution and statistical analysis visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()
    
    # 1. Distribution of energy consumption
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(latest_data['total_energy_twh'].dropna(), bins=30, 
             color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Energy Consumption (TWh)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Energy Consumption', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(latest_data['total_energy_twh'].median(), color='red', 
               linestyle='--', linewidth=2, label=f'Median: {latest_data["total_energy_twh"].median():.0f}')
    ax1.legend()
    
    # Box plot
    ax2 = axes[1]
    box_data = [latest_data['total_energy_twh'].dropna()]
    bp = ax2.boxplot(box_data, vert=True, patch_artist=True,
                     labels=['All Countries'], showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Energy Consumption (TWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot of Energy Consumption', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Statistical Distribution Analysis ({latest_year})', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/12_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Distribution analysis created")


def plot_regional_comparison(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create regional comparison visualizations (if region data available)."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'region' not in df.columns:
        # Create a simple regional grouping based on country names
        # This is a fallback if no region column exists
        print("⚠ No region column found, skipping regional comparison")
        return
    
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()
    
    # Regional totals
    regional_totals = latest_data.groupby('region')['total_energy_twh'].sum().sort_values(ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(regional_totals)))
    ax1.pie(regional_totals.values, labels=regional_totals.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Regional Energy Consumption Share', fontsize=12, fontweight='bold')
    
    # Bar chart
    ax2.bar(regional_totals.index, regional_totals.values, color=colors_pie)
    ax2.set_ylabel('Total Energy Consumption (TWh)', fontsize=11, fontweight='bold')
    ax2.set_title('Regional Energy Consumption Totals', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (region, val) in enumerate(regional_totals.items()):
        ax2.text(i, val, f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Regional Energy Consumption Comparison ({latest_year})', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/13_regional_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Regional comparison created")


def plot_temporal_evolution(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create temporal evolution visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Global total over time
    global_totals = df.groupby('year')['total_energy_twh'].sum()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Global total
    ax1 = axes[0]
    ax1.plot(global_totals.index, global_totals.values, 
            marker='o', linewidth=3, markersize=6, color='darkgreen')
    ax1.fill_between(global_totals.index, global_totals.values, alpha=0.3, color='green')
    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Global Total Energy Consumption (TWh)', fontsize=11, fontweight='bold')
    ax1.set_title('Global Energy Consumption Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Year-over-year growth rate
    ax2 = axes[1]
    yoy_growth = global_totals.pct_change() * 100
    colors_growth = ['green' if x > 0 else 'red' for x in yoy_growth.values]
    ax2.bar(yoy_growth.index, yoy_growth.values, color=colors_growth, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Year-over-Year Growth Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Global Energy Consumption Growth Rate', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Global Energy Consumption Evolution', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/14_temporal_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Temporal evolution plots created")


def create_all_visualizations(df: Optional[pd.DataFrame] = None, 
                              input_file: Optional[str] = None,
                              output_dir: str = 'figures'):
    """
    Create all visualizations.
    
    ⚠ WARNING: These visualizations do NOT show data quality flags.
    For research/academic use, use create_visualizations_robust.py instead.
    
    Args:
        df: Optional DataFrame (if None, will try to load from file)
        input_file: Optional path to processed data CSV
        output_dir: Output directory for figures
    """
    print("=" * 80)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("⚠ WARNING: These visualizations do NOT show data quality issues.")
    print("   For transparent data quality visualization, use create_visualizations_robust.py")
    print("=" * 80)
    
    # Load data if not provided
    if df is None:
        if input_file:
            df = pd.read_csv(input_file)
        else:
            df = load_processed_data()
    
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries: {df['country'].nunique()}")
    print()
    
    # Create all visualizations
    plot_time_series_comprehensive(df, output_dir)
    plot_bar_charts_comprehensive(df, output_dir)
    plot_growth_analysis(df, output_dir)
    plot_data_quality_visualization(df, output_dir)
    plot_heatmap_visualizations(df, output_dir)
    plot_correlation_analysis(df, output_dir)
    plot_distribution_analysis(df, output_dir)
    plot_regional_comparison(df, output_dir)
    plot_temporal_evolution(df, output_dir)
    
    print()
    print("=" * 80)
    print(f"✓ All visualizations saved to {output_dir}/")
    print("=" * 80)
    print("\nGenerated files:")
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}. {f}")


if __name__ == '__main__':
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    create_all_visualizations(input_file=input_file)

