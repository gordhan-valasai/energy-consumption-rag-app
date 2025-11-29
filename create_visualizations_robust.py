"""
ROBUST Visualization Module - Data Quality Aware

Creates visualizations that TRANSPARENTLY show:
- Interpolated vs original data
- Synthetic population indicators
- Unit conversion warnings
- Missing data patterns
- Outlier removal markers
- Uncertainty bands
- Data quality flags

This version does NOT hide data quality issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import os
from typing import Optional, List, Tuple
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

# Data quality flag constants
FLAG_ORIGINAL = 0
FLAG_INTERPOLATED = 1
FLAG_MISSING_BLOCK = 2
FLAG_REMOVED_OUTLIER = 3


def add_data_quality_warning(fig, synthetic_population: bool = False, 
                            unit_guessing: bool = False, 
                            interpolation_count: int = 0):
    """Add warning banner to figure about data quality issues."""
    warning_text = []
    if synthetic_population:
        warning_text.append("⚠ PER-CAPITA VALUES USE SYNTHETIC POPULATION DATA")
    if unit_guessing:
        warning_text.append("⚠ SOME VALUES DERIVED FROM UNIT MAGNITUDE GUESSING")
    if interpolation_count > 0:
        warning_text.append(f"⚠ {interpolation_count} INTERPOLATED DATA POINTS")
    
    if warning_text:
        fig.text(0.5, 0.02, " | ".join(warning_text), 
                ha='center', fontsize=8, color='red', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                weight='bold')


def plot_time_series_with_quality_flags(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create time series plots that show data quality flags."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for synthetic population
    synthetic_pop = 'population' in df.columns and df['population'].notna().any()
    
    # Count interpolated points
    if 'data_quality_flag' in df.columns:
        interpolated_count = (df['data_quality_flag'] == FLAG_INTERPOLATED).sum()
    else:
        interpolated_count = 0
    
    top_countries = df.groupby('country')['total_energy_twh'].max().nlargest(10).index
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    
    for i, country in enumerate(top_countries):
        country_data = df[df['country'] == country].sort_values('year').copy()
        
        # Separate original and interpolated data
        if 'data_quality_flag' in country_data.columns:
            original_data = country_data[country_data['data_quality_flag'] == FLAG_ORIGINAL]
            interpolated_data = country_data[country_data['data_quality_flag'] == FLAG_INTERPOLATED]
        else:
            original_data = country_data
            interpolated_data = pd.DataFrame()
        
        # Plot original data (solid line, filled markers)
        if len(original_data) > 0:
            ax.plot(original_data['year'], original_data['total_energy_twh'], 
                   marker='o', label=country, linewidth=2.5, markersize=5, 
                   color=colors[i], linestyle='-', zorder=2)
        
        # Plot interpolated data (dashed line, hollow markers)
        if len(interpolated_data) > 0:
            ax.plot(interpolated_data['year'], interpolated_data['total_energy_twh'], 
                   marker='o', linewidth=2, markersize=4, 
                   color=colors[i], linestyle='--', markerfacecolor='white',
                   markeredgewidth=1.5, alpha=0.7, zorder=1)
    
    # Add legend for data quality
    original_patch = mpatches.Patch(color='black', label='Original Data')
    interpolated_patch = mpatches.Patch(color='gray', linestyle='--', label='Interpolated Data')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Energy Consumption (TWh)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Consumption Time Series - Top 10 Countries\n(Solid=Original, Dashed=Interpolated)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Create combined legend
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([original_patch, interpolated_patch])
    labels.extend(['Original Data', 'Interpolated Data'])
    ax.legend(handles, labels, loc='upper left', ncol=2, frameon=True, 
             fancybox=True, shadow=True, fontsize=8)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add warning banner
    add_data_quality_warning(fig, synthetic_population=synthetic_pop, 
                            interpolation_count=interpolated_count)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f'{output_dir}/01_time_series_with_quality_flags.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Time series with quality flags created")


def plot_missing_data_patterns(df: pd.DataFrame, output_dir: str = 'figures'):
    """Visualize missing data patterns by country."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create missing data matrix
    countries = sorted(df['country'].unique())
    years = sorted(df['year'].unique())
    
    missing_matrix = np.zeros((len(countries), len(years)))
    
    for i, country in enumerate(countries):
        country_data = df[df['country'] == country]
        for j, year in enumerate(years):
            year_data = country_data[country_data['year'] == year]
            if len(year_data) == 0 or year_data['total_energy_twh'].isna().all():
                missing_matrix[i, j] = 1  # Missing
            elif 'data_quality_flag' in year_data.columns:
                flag = year_data['data_quality_flag'].iloc[0]
                if flag == FLAG_INTERPOLATED:
                    missing_matrix[i, j] = 0.5  # Interpolated
                elif flag == FLAG_MISSING_BLOCK:
                    missing_matrix[i, j] = 0.75  # Missing block
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(countries) * 0.3)))
    
    # Create custom colormap: white (data), yellow (interpolated), orange (missing), red (missing block)
    from matplotlib.colors import ListedColormap
    colors_missing = ['white', 'lightyellow', 'orange', 'red']
    cmap_missing = ListedColormap(colors_missing)
    
    im = ax.imshow(missing_matrix, aspect='auto', cmap=cmap_missing, 
                   vmin=0, vmax=1, interpolation='nearest')
    
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=8)
    ax.set_xticks(range(0, len(years), max(1, len(years)//10)))
    ax.set_xticklabels([years[i] for i in range(0, len(years), max(1, len(years)//10))], 
                       rotation=45, fontsize=8)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
    ax.set_title('Data Completeness Matrix\n(White=Original, Yellow=Interpolated, Orange=Missing, Red=Missing Block)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Original Data'),
        Rectangle((0, 0), 1, 1, facecolor='lightyellow', edgecolor='black', label='Interpolated'),
        Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='black', label='Missing'),
        Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', label='Missing Block')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_missing_data_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Missing data patterns visualization created")


def plot_data_quality_breakdown(df: pd.DataFrame, output_dir: str = 'figures'):
    """Comprehensive data quality breakdown visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'data_quality_flag' not in df.columns:
        print("⚠ No data quality flags found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Flag distribution
    ax1 = axes[0, 0]
    quality_labels = {
        FLAG_ORIGINAL: 'Original',
        FLAG_INTERPOLATED: 'Interpolated',
        FLAG_MISSING_BLOCK: 'Missing Block',
        FLAG_REMOVED_OUTLIER: 'Removed Outlier'
    }
    quality_counts = df['data_quality_flag'].value_counts().sort_index()
    quality_names = [quality_labels.get(i, f'Flag {i}') for i in quality_counts.index]
    colors_quality = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars = ax1.bar(quality_names, quality_counts.values, 
                   color=[colors_quality[i] for i in range(len(quality_names))])
    ax1.set_ylabel('Number of Data Points', fontsize=11, fontweight='bold')
    ax1.set_title('Data Quality Flag Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, quality_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}\n({val/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Interpolation gap lengths
    ax2 = axes[0, 1]
    if FLAG_INTERPOLATED in df['data_quality_flag'].values:
        interpolated_countries = df[df['data_quality_flag'] == FLAG_INTERPOLATED]['country'].unique()
        gap_lengths = []
        for country in interpolated_countries:
            country_data = df[df['country'] == country].sort_values('year')
            interpolated_years = country_data[country_data['data_quality_flag'] == FLAG_INTERPOLATED]['year'].values
            original_years = country_data[country_data['data_quality_flag'] == FLAG_ORIGINAL]['year'].values
            for interp_year in interpolated_years:
                # Find gap length
                before = original_years[original_years < interp_year]
                after = original_years[original_years > interp_year]
                if len(before) > 0 and len(after) > 0:
                    gap = min(after) - max(before) - 1
                    gap_lengths.append(gap)
        
        if gap_lengths:
            ax2.hist(gap_lengths, bins=range(1, max(gap_lengths)+2), 
                    color='orange', edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Interpolation Gap Length (years)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax2.set_title('Distribution of Interpolation Gap Lengths', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No interpolation gaps detected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Interpolation Gap Lengths', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No interpolated data', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Interpolation Gap Lengths', fontsize=12, fontweight='bold')
    
    # 3. Missing data by country
    ax3 = axes[1, 0]
    missing_by_country = df.groupby('country').agg({
        'total_energy_twh': lambda x: x.isna().sum(),
        'data_quality_flag': lambda x: (x == FLAG_INTERPOLATED).sum()
    }).rename(columns={'total_energy_twh': 'missing', 'data_quality_flag': 'interpolated'})
    missing_by_country = missing_by_country[missing_by_country['missing'] > 0].sort_values('missing', ascending=False)
    
    if len(missing_by_country) > 0:
        x = np.arange(len(missing_by_country.head(15)))
        width = 0.35
        ax3.barh(x - width/2, missing_by_country.head(15)['missing'], width, 
                label='Missing', color='red', alpha=0.7)
        ax3.barh(x + width/2, missing_by_country.head(15)['interpolated'], width,
                label='Interpolated', color='orange', alpha=0.7)
        ax3.set_yticks(x)
        ax3.set_yticklabels(missing_by_country.head(15).index, fontsize=8)
        ax3.set_xlabel('Number of Years', fontsize=11, fontweight='bold')
        ax3.set_title('Missing vs Interpolated Data by Country', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No missing data detected', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Missing Data by Country', fontsize=12, fontweight='bold')
    
    # 4. Data completeness over time
    ax4 = axes[1, 1]
    completeness_by_year = df.groupby('year').agg({
        'total_energy_twh': lambda x: x.notna().sum(),
        'data_quality_flag': lambda x: (x == FLAG_ORIGINAL).sum()
    }).rename(columns={'total_energy_twh': 'total', 'data_quality_flag': 'original'})
    completeness_by_year['pct_original'] = completeness_by_year['original'] / completeness_by_year['total'] * 100
    
    ax4_twin = ax4.twinx()
    ax4.plot(completeness_by_year.index, completeness_by_year['total'], 
            marker='o', label='Total Data Points', color='blue', linewidth=2)
    ax4_twin.plot(completeness_by_year.index, completeness_by_year['pct_original'], 
                 marker='s', label='% Original', color='green', linewidth=2)
    
    ax4.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total Data Points', fontsize=11, fontweight='bold', color='blue')
    ax4_twin.set_ylabel('% Original Data', fontsize=11, fontweight='bold', color='green')
    ax4.set_title('Data Completeness Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.suptitle('Comprehensive Data Quality Analysis', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_data_quality_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Data quality breakdown created")


def plot_bar_charts_with_uncertainty(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create bar charts with uncertainty indicators."""
    os.makedirs(output_dir, exist_ok=True)
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()
    
    # Check for synthetic population
    synthetic_pop = 'population' in df.columns and df['population'].notna().any()
    
    top_15 = latest_data.nlargest(15, 'total_energy_twh')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate uncertainty: if interpolated or missing block, mark it
    colors = []
    edge_colors = []
    for idx, row in top_15.iterrows():
        if 'data_quality_flag' in df.columns:
            flag = row.get('data_quality_flag', FLAG_ORIGINAL)
            if flag == FLAG_INTERPOLATED:
                colors.append('lightblue')
                edge_colors.append('blue')
            elif flag == FLAG_MISSING_BLOCK:
                colors.append('lightcoral')
                edge_colors.append('red')
            else:
                colors.append('steelblue')
                edge_colors.append('black')
        else:
            colors.append('steelblue')
            edge_colors.append('black')
    
    bars = ax.barh(top_15['country'], top_15['total_energy_twh'], 
                  color=colors, edgecolor=edge_colors, linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_15['total_energy_twh'])):
        ax.text(val, i, f' {val:.0f}', va='center', fontsize=9)
    
    ax.set_xlabel('Total Energy Consumption (TWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Countries by Energy Consumption ({latest_year})\n(Blue=Original, Light Blue=Interpolated, Red=Missing Block)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add warning if synthetic population
    if synthetic_pop and 'energy_per_capita_twh' in latest_data.columns:
        fig.text(0.5, 0.02, "⚠ PER-CAPITA VALUES USE SYNTHETIC POPULATION DATA - NOT RELIABLE", 
                ha='center', fontsize=9, color='red', weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f'{output_dir}/04_top15_with_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Bar charts with uncertainty created")


def plot_growth_analysis_with_warnings(df: pd.DataFrame, output_dir: str = 'figures'):
    """Create growth analysis with warnings about data quality."""
    os.makedirs(output_dir, exist_ok=True)
    latest_year = df['year'].max()
    
    if 2000 not in df['year'].values:
        print("⚠ Skipping growth analysis (no 2000 data)")
        return
    
    # Calculate growth rates with quality checks
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
                
                # Check data quality
                flag_2000 = year_2000_data['data_quality_flag'].iloc[0] if 'data_quality_flag' in year_2000_data.columns else FLAG_ORIGINAL
                flag_latest = year_latest_data['data_quality_flag'].iloc[0] if 'data_quality_flag' in year_latest_data.columns else FLAG_ORIGINAL
                
                # Mark if baseline or latest is interpolated
                unreliable = (flag_2000 == FLAG_INTERPOLATED) or (flag_latest == FLAG_INTERPOLATED)
                
                growth_data.append({
                    'country': country,
                    'growth_rate_pct': growth_rate,
                    'value_2000': val_2000,
                    'value_latest': val_latest,
                    'absolute_change': val_latest - val_2000,
                    'unreliable': unreliable
                })
    
    if not growth_data:
        print("⚠ No growth data available")
        return
    
    growth_df = pd.DataFrame(growth_data)
    
    # Separate reliable and unreliable
    reliable = growth_df[~growth_df['unreliable']]
    unreliable = growth_df[growth_df['unreliable']]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot reliable data
    if len(reliable) > 0:
        top_reliable = reliable.nlargest(15, 'growth_rate_pct')
        ax.barh(top_reliable['country'], top_reliable['growth_rate_pct'], 
               color='green', alpha=0.7, label='Reliable (Original Data)')
    
    # Plot unreliable data (with warning)
    if len(unreliable) > 0:
        top_unreliable = unreliable.nlargest(15, 'growth_rate_pct')
        ax.barh(top_unreliable['country'], top_unreliable['growth_rate_pct'], 
               color='orange', alpha=0.7, label='Unreliable (Interpolated Baseline/Latest)',
               hatch='///', edgecolor='red', linewidth=1)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Annual Growth Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Growth Rates (2000-{latest_year})\n⚠ Orange bars use interpolated baseline or latest values', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_growth_rates_with_warnings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Growth analysis with warnings created")


def create_all_robust_visualizations(df: Optional[pd.DataFrame] = None, 
                                    input_file: Optional[str] = None,
                                    output_dir: str = 'figures_robust'):
    """
    Create all robust visualizations with data quality transparency.
    
    Args:
        df: Optional DataFrame (if None, will try to load from file)
        input_file: Optional path to processed data CSV
        output_dir: Output directory for figures
    """
    print("=" * 80)
    print("CREATING ROBUST, DATA-QUALITY-AWARE VISUALIZATIONS")
    print("=" * 80)
    
    # Load data if not provided
    if df is None:
        if input_file:
            df = pd.read_csv(input_file)
        elif os.path.exists('output/cleaned_energy_data.csv'):
            df = pd.read_csv('output/cleaned_energy_data.csv')
        elif os.path.exists('output_robust/cleaned_energy_data.csv'):
            df = pd.read_csv('output_robust/cleaned_energy_data.csv')
        else:
            raise FileNotFoundError("Processed data file not found. Please run the processing pipeline first.")
    
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries: {df['country'].nunique()}")
    
    # Check for data quality issues
    if 'data_quality_flag' in df.columns:
        interpolated = (df['data_quality_flag'] == FLAG_INTERPOLATED).sum()
        missing_block = (df['data_quality_flag'] == FLAG_MISSING_BLOCK).sum()
        print(f"Data Quality: {interpolated} interpolated, {missing_block} missing blocks")
    
    synthetic_pop = 'population' in df.columns and df['population'].notna().any()
    if synthetic_pop:
        print("⚠ WARNING: Population data detected - per-capita values may be unreliable")
    
    print()
    
    # Create all visualizations
    plot_time_series_with_quality_flags(df, output_dir)
    plot_missing_data_patterns(df, output_dir)
    plot_data_quality_breakdown(df, output_dir)
    plot_bar_charts_with_uncertainty(df, output_dir)
    plot_growth_analysis_with_warnings(df, output_dir)
    
    print()
    print("=" * 80)
    print(f"✓ All robust visualizations saved to {output_dir}/")
    print("=" * 80)
    print("\n⚠ IMPORTANT: These visualizations show data quality issues transparently.")
    print("   Use these for analysis, NOT the standard visualizations.")
    print("\nGenerated files:")
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}. {f}")


if __name__ == '__main__':
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    create_all_robust_visualizations(input_file=input_file)

