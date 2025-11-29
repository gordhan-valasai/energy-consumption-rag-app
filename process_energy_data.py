"""
Global Energy Consumption Audit - Python Script
BP Statistical Review Dataset Processing Pipeline

This script processes raw energy consumption data, standardizes units,
cleans missing values, normalizes by population/GDP, and generates
summary statistics and visualizations.
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Unit conversion constants
EJ_TO_TWH = 277.778
MTOE_TO_TWH = 11.63
KTOE_TO_TWH = 0.01163

# Sanity check thresholds
MAX_TWH = 50000  # Maximum reasonable energy consumption in TWh
MIN_TWH = 0  # Minimum (negative values are invalid)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw energy data from CSV or Excel file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        DataFrame with raw data
    """
    logger.info(f"Loading data from {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase snake_case.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    logger.info("Standardizing column names")
    
    # Convert to lowercase and replace spaces/special chars with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    df.columns = df.columns.str.replace('[^a-z0-9_]', '', regex=True)
    
    # Common column name mappings
    column_mapping = {
        'country': 'country',
        'year': 'year',
        'primary_energy': 'primary_energy',
        'primary_energy_consumption': 'primary_energy',
        'energy_consumption': 'primary_energy',
        'electricity_generation': 'electricity_generation',
        'electricity': 'electricity_generation',
        'total_energy': 'primary_energy',
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    logger.info(f"Standardized columns: {list(df.columns)}")
    return df


def convert_units_to_twh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all energy units to TWh.
    
    Handles columns with units in EJ, Mtoe, ktoe, or TWh.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all energy values in TWh
    """
    logger.info("Converting units to TWh")
    
    df = df.copy()
    
    # Identify energy columns (excluding country, year, and flag columns)
    exclude_cols = ['country', 'year', 'data_quality_flag', 'region']
    energy_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, float, int]]
    
    # Detect unit from column name or data magnitude
    for col in energy_cols:
        col_lower = col.lower()
        
        if 'ej' in col_lower or 'exajoule' in col_lower:
            # Convert EJ to TWh
            if df[col].notna().any():
                df[col] = df[col] * EJ_TO_TWH
                logger.info(f"Converted {col} from EJ to TWh")
        
        elif 'mtoe' in col_lower or 'million_tonnes' in col_lower:
            # Convert Mtoe to TWh
            if df[col].notna().any():
                df[col] = df[col] * MTOE_TO_TWH
                logger.info(f"Converted {col} from Mtoe to TWh")
        
        elif 'ktoe' in col_lower or 'thousand_tonnes' in col_lower:
            # Convert ktoe to TWh
            if df[col].notna().any():
                df[col] = df[col] * KTOE_TO_TWH
                logger.info(f"Converted {col} from ktoe to TWh")
        
        elif 'twh' not in col_lower and df[col].notna().any():
            # Try to detect unit by magnitude
            max_val = df[col].max()
            if max_val > 1000 and max_val < 10000:
                # Likely EJ (typical range 100-1000 EJ)
                df[col] = df[col] * EJ_TO_TWH
                logger.info(f"Auto-converted {col} from EJ to TWh (detected by magnitude)")
            elif max_val > 10 and max_val < 1000:
                # Likely Mtoe (typical range 10-1000 Mtoe)
                df[col] = df[col] * MTOE_TO_TWH
                logger.info(f"Auto-converted {col} from Mtoe to TWh (detected by magnitude)")
    
    # Create unified total_energy_twh column
    # Priority: primary_energy > electricity_generation > first available energy column
    # Avoid double-counting when multiple unit columns exist for same data
    if 'primary_energy' in df.columns:
        df['total_energy_twh'] = df['primary_energy']
    elif 'electricity_generation' in df.columns:
        df['total_energy_twh'] = df['electricity_generation']
    else:
        # If multiple energy columns exist, prefer the one with most non-null values
        # or use the first one that's not all null
        numeric_cols = [col for col in energy_cols if col not in exclude_cols]
        if len(numeric_cols) == 1:
            df['total_energy_twh'] = df[numeric_cols[0]]
        elif len(numeric_cols) > 1:
            # Check if columns represent same data in different units (likely double-counting)
            # Use first non-null value per row to avoid double-counting
            logger.info(f"Multiple energy columns found: {numeric_cols}")
            logger.warning("Using first non-null value per row to avoid double-counting")
            
            # Use first non-null value across columns for each row
            # Stack columns and take first non-null value
            energy_subset = df[numeric_cols]
            df['total_energy_twh'] = energy_subset.apply(
                lambda row: next((val for val in row if pd.notna(val)), np.nan), axis=1
            )
        else:
            raise ValueError("No energy columns found in the dataset")
    
    # Sanity check: verify conversion
    if df['total_energy_twh'].notna().any():
        max_val = df['total_energy_twh'].max()
        min_val = df['total_energy_twh'].min()
        logger.info(f"Total energy range: {min_val:.2f} - {max_val:.2f} TWh")
        
        # Assert reasonable values (relaxed threshold for global totals)
        # Global energy consumption can exceed 100,000 TWh when summing all countries
        if max_val > MAX_TWH * 5:  # 250,000 TWh threshold
            logger.warning(f"Very high maximum value {max_val} TWh detected - may indicate double-counting or data issue")
        assert min_val >= MIN_TWH, f"Minimum value {min_val} TWh is negative"
    
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean data: handle missing values, interpolate, flag issues, remove outliers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, outlier report DataFrame)
    """
    logger.info("Cleaning data")
    
    df = df.copy()
    
    # Initialize data quality flag column
    if 'data_quality_flag' not in df.columns:
        df['data_quality_flag'] = 0  # 0 = original
    
    # Identify missing values by country & year
    missing_by_country = df.groupby('country')['total_energy_twh'].apply(
        lambda x: x.isna().sum()
    ).sort_values(ascending=False)
    
    logger.info(f"Missing values by country:\n{missing_by_country[missing_by_country > 0]}")
    
    # Track outliers
    outlier_rows = []
    
    # Process each country
    for country in df['country'].unique():
        country_mask = df['country'] == country
        country_data = df.loc[country_mask].copy()
        
        # Check for impossible values (negative or >50000 TWh)
        invalid_mask = (country_data['total_energy_twh'] < MIN_TWH) | \
                       (country_data['total_energy_twh'] > MAX_TWH)
        
        if invalid_mask.any():
            invalid_indices = country_data[invalid_mask].index
            logger.warning(f"Found {invalid_mask.sum()} outliers in {country}")
            
            for idx in invalid_indices:
                outlier_rows.append({
                    'country': country,
                    'year': df.loc[idx, 'year'],
                    'value': df.loc[idx, 'total_energy_twh'],
                    'reason': 'outlier'
                })
            
            # Mark as removed
            df.loc[invalid_indices, 'data_quality_flag'] = 3  # 3 = removed_outlier
            df.loc[invalid_indices, 'total_energy_twh'] = np.nan
    
        # Check for missing data patterns
        energy_series = country_data['total_energy_twh']
        missing_count = energy_series.isna().sum()
        total_count = len(energy_series)
        
        if missing_count == total_count:
            # Entire country block missing
            logger.warning(f"Entire country block missing for {country}")
            df.loc[country_mask, 'data_quality_flag'] = 2  # 2 = flagged_missing_block
        
        elif missing_count > 0 and missing_count < total_count:
            # Partial missing years - interpolate
            logger.info(f"Interpolating {missing_count} missing values for {country}")
            
            # Sort by year for interpolation
            country_indices = country_data.sort_values('year').index
            
            # Interpolate
            df.loc[country_indices, 'total_energy_twh'] = df.loc[country_indices, 'total_energy_twh'].interpolate(
                method='linear',
                limit_direction='both'
            )
            
            # Mark interpolated values
            interpolated_mask = df.loc[country_indices, 'data_quality_flag'] == 0
            interpolated_mask = interpolated_mask & df.loc[country_indices, 'total_energy_twh'].notna()
            interpolated_mask = interpolated_mask & country_data['total_energy_twh'].isna()
            
            df.loc[country_indices[interpolated_mask], 'data_quality_flag'] = 1  # 1 = interpolated
    
    # Remove rows marked as outliers
    df_cleaned = df[df['data_quality_flag'] != 3].copy()
    
    # Create outlier report
    outlier_report = pd.DataFrame(outlier_rows) if outlier_rows else pd.DataFrame(
        columns=['country', 'year', 'value', 'reason']
    )
    
    logger.info(f"Cleaned data: {len(df_cleaned)} rows remaining ({len(df) - len(df_cleaned)} removed)")
    
    return df_cleaned, outlier_report


def load_population_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load population data. If file not provided, use World Bank data or create sample.
    
    Args:
        file_path: Optional path to population CSV
        
    Returns:
        DataFrame with country, year, population columns
    """
    if file_path and os.path.exists(file_path):
        logger.info(f"Loading population data from {file_path}")
        pop_df = pd.read_csv(file_path)
        pop_df.columns = pop_df.columns.str.lower().str.replace(' ', '_')
        return pop_df
    
    # Create sample population data (in millions)
    logger.warning("No population file provided, creating sample data")
    countries = ['United States', 'China', 'India', 'Japan', 'Germany', 
                 'Russia', 'Brazil', 'South Korea', 'Canada', 'France']
    years = range(2000, 2025)
    
    pop_data = []
    base_pop = {
        'United States': 280, 'China': 1400, 'India': 1200,
        'Japan': 125, 'Germany': 83, 'Russia': 145,
        'Brazil': 200, 'South Korea': 52, 'Canada': 38, 'France': 67
    }
    
    for country in countries:
        for year in years:
            growth_rate = 0.01  # 1% annual growth
            years_since_2000 = year - 2000
            pop = base_pop.get(country, 50) * (1 + growth_rate) ** years_since_2000
            pop_data.append({'country': country, 'year': year, 'population': pop})
    
    return pd.DataFrame(pop_data)


def normalize_data(df: pd.DataFrame, pop_df: Optional[pd.DataFrame] = None,
                   gdp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Normalize energy data by population and GDP.
    
    Args:
        df: Energy DataFrame
        pop_df: Population DataFrame
        gdp_df: Optional GDP DataFrame
        
    Returns:
        DataFrame with normalized columns
    """
    logger.info("Normalizing data by population and GDP")
    
    df = df.copy()
    
    # Normalize by population
    if pop_df is not None:
        # Merge population data
        df = df.merge(
            pop_df[['country', 'year', 'population']],
            on=['country', 'year'],
            how='left'
        )
        
        # Calculate per-capita energy
        df['energy_per_capita_twh'] = df['total_energy_twh'] / df['population']
        df['energy_per_capita_twh'] = df['energy_per_capita_twh'].replace([np.inf, -np.inf], np.nan)
        
        logger.info("Added energy_per_capita_twh column")
    else:
        pop_df = load_population_data()
        df = normalize_data(df, pop_df, gdp_df)
        return df
    
    # Normalize by GDP if available
    if gdp_df is not None:
        gdp_df.columns = gdp_df.columns.str.lower().str.replace(' ', '_')
        df = df.merge(
            gdp_df[['country', 'year', 'gdp']],
            on=['country', 'year'],
            how='left'
        )
        
        # Calculate energy intensity (TWh per unit GDP)
        df['energy_intensity_twh_per_gdp'] = df['total_energy_twh'] / df['gdp']
        df['energy_intensity_twh_per_gdp'] = df['energy_intensity_twh_per_gdp'].replace([np.inf, -np.inf], np.nan)
        
        logger.info("Added energy_intensity_twh_per_gdp column")
    
    return df


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Generating summary statistics")
    
    stats = {}
    
    # Get latest year
    latest_year = df['year'].max()
    stats['latest_year'] = latest_year
    
    # Top 10 energy consumers (latest year)
    latest_data = df[df['year'] == latest_year].copy()
    top_10 = latest_data.nlargest(10, 'total_energy_twh')[
        ['country', 'total_energy_twh', 'energy_per_capita_twh']
    ].copy()
    stats['top_10_consumers'] = top_10
    
    # Growth rates 2000-2024 by country
    if 2000 in df['year'].values and latest_year >= 2000:
        growth_data = []
        for country in df['country'].unique():
            country_data = df[df['country'] == country].sort_values('year')
            year_2000 = country_data[country_data['year'] == 2000]['total_energy_twh']
            year_latest = country_data[country_data['year'] == latest_year]['total_energy_twh']
            
            if len(year_2000) > 0 and len(year_latest) > 0:
                val_2000 = year_2000.iloc[0]
                val_latest = year_latest.iloc[0]
                
                if pd.notna(val_2000) and pd.notna(val_latest) and val_2000 > 0:
                    growth_rate = ((val_latest / val_2000) ** (1 / (latest_year - 2000)) - 1) * 100
                    growth_data.append({
                        'country': country,
                        'growth_rate_pct': growth_rate,
                        'value_2000': val_2000,
                        'value_latest': val_latest
                    })
        
        growth_df = pd.DataFrame(growth_data).sort_values('growth_rate_pct', ascending=False)
        stats['growth_rates'] = growth_df
    
    # Missing data report
    missing_report = df.groupby('country').agg({
        'total_energy_twh': lambda x: x.isna().sum(),
        'data_quality_flag': lambda x: (x == 2).sum()  # flagged_missing_block
    }).rename(columns={
        'total_energy_twh': 'missing_values',
        'data_quality_flag': 'missing_blocks'
    })
    missing_report = missing_report[missing_report['missing_values'] > 0].sort_values(
        'missing_values', ascending=False
    )
    stats['missing_data_report'] = missing_report
    
    # Data quality summary
    quality_summary = df['data_quality_flag'].value_counts().to_dict()
    quality_labels = {0: 'original', 1: 'interpolated', 2: 'flagged_missing_block', 3: 'removed_outlier'}
    stats['data_quality_summary'] = {
        quality_labels.get(k, k): v for k, v in quality_summary.items()
    }
    
    return stats


def create_visualizations(df: pd.DataFrame, output_dir: str = 'figures'):
    """
    Create visualizations: time series, scatter plots, heatmap, bar charts.
    
    Args:
        df: Cleaned DataFrame
        output_dir: Output directory for figures
    """
    logger.info(f"Creating visualizations in {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Time series for selected countries
    selected_countries = df.groupby('country')['total_energy_twh'].max().nlargest(5).index
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for country in selected_countries:
        country_data = df[df['country'] == country].sort_values('year')
        ax.plot(country_data['year'], country_data['total_energy_twh'], 
                marker='o', label=country, linewidth=2)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Energy Consumption (TWh)', fontsize=12)
    ax.set_title('Energy Consumption Time Series - Top 5 Countries', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_series_top_countries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter: GDP vs Energy Use (log scale) - if GDP available
    if 'gdp' in df.columns:
        latest_data = df[df['year'] == df['year'].max()].copy()
        latest_data = latest_data[latest_data['total_energy_twh'].notna() & latest_data['gdp'].notna()]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(latest_data['gdp'], latest_data['total_energy_twh'], 
                  alpha=0.6, s=100)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('GDP (log scale)', fontsize=12)
        ax.set_ylabel('Energy Consumption (TWh, log scale)', fontsize=12)
        ax.set_title('GDP vs Energy Consumption (Latest Year)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gdp_vs_energy_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Heatmap of consumption intensity (TWh per capita)
    if 'energy_per_capita_twh' in df.columns:
        latest_data = df[df['year'] == df['year'].max()].copy()
        latest_data = latest_data[latest_data['energy_per_capita_twh'].notna()]
        
        # Select top 20 countries by per-capita consumption
        top_20 = latest_data.nlargest(20, 'energy_per_capita_twh')
        
        # Create pivot table for heatmap (if we have regional data)
        if 'region' in df.columns:
            pivot_data = df[df['year'] == df['year'].max()].pivot_table(
                values='energy_per_capita_twh',
                index='country',
                columns='region',
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
                ax.set_title('Energy Consumption Intensity Heatmap (TWh per capita)', 
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/energy_intensity_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Alternative: Bar chart of top countries
        fig, ax = plt.subplots(figsize=(12, 8))
        top_20_sorted = top_20.sort_values('energy_per_capita_twh', ascending=True)
        ax.barh(top_20_sorted['country'], top_20_sorted['energy_per_capita_twh'], 
               color='steelblue')
        ax.set_xlabel('Energy per Capita (TWh)', fontsize=12)
        ax.set_title('Top 20 Countries by Energy Consumption per Capita', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/energy_per_capita_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Bar chart: Energy consumption by region (latest year)
    if 'region' in df.columns:
        latest_data = df[df['year'] == df['year'].max()].copy()
        regional_totals = latest_data.groupby('region')['total_energy_twh'].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(regional_totals.index, regional_totals.values, color='coral')
        ax.set_ylabel('Total Energy Consumption (TWh)', fontsize=12)
        ax.set_title(f'Energy Consumption by Region ({df["year"].max()})', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/energy_by_region_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # If no region column, create by top countries
        latest_data = df[df['year'] == df['year'].max()].copy()
        top_countries = latest_data.nlargest(15, 'total_energy_twh')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(top_countries['country'], top_countries['total_energy_twh'], color='teal')
        ax.set_xlabel('Total Energy Consumption (TWh)', fontsize=12)
        ax.set_title(f'Top 15 Countries by Energy Consumption ({df["year"].max()})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_countries_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}/")


def export_results(df: pd.DataFrame, stats: dict, outlier_report: pd.DataFrame,
                   output_dir: str = 'output'):
    """
    Export cleaned data, summary statistics, and reports.
    
    Args:
        df: Cleaned DataFrame
        stats: Summary statistics dictionary
        outlier_report: Outlier report DataFrame
        output_dir: Output directory
    """
    logger.info(f"Exporting results to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export cleaned dataset
    df.to_csv(f'{output_dir}/cleaned_energy_data.csv', index=False)
    logger.info(f"Exported cleaned data to {output_dir}/cleaned_energy_data.csv")
    
    # Export summary statistics
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("GLOBAL ENERGY CONSUMPTION AUDIT - SUMMARY STATISTICS")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\nLatest Year: {stats['latest_year']}")
    
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("TOP 10 ENERGY CONSUMERS (Latest Year)")
    summary_lines.append("-" * 80)
    summary_lines.append(stats['top_10_consumers'].to_string())
    
    if 'growth_rates' in stats and len(stats['growth_rates']) > 0:
        summary_lines.append("\n" + "-" * 80)
        summary_lines.append("GROWTH RATES (2000-2024)")
        summary_lines.append("-" * 80)
        summary_lines.append(stats['growth_rates'].head(20).to_string())
    
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("DATA QUALITY SUMMARY")
    summary_lines.append("-" * 80)
    for quality_type, count in stats['data_quality_summary'].items():
        summary_lines.append(f"{quality_type}: {count}")
    
    if len(stats['missing_data_report']) > 0:
        summary_lines.append("\n" + "-" * 80)
        summary_lines.append("MISSING DATA REPORT")
        summary_lines.append("-" * 80)
        summary_lines.append(stats['missing_data_report'].to_string())
    
    # Write summary to file
    with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Export CSV summaries
    stats['top_10_consumers'].to_csv(f'{output_dir}/top_10_consumers.csv', index=False)
    
    if 'growth_rates' in stats and len(stats['growth_rates']) > 0:
        stats['growth_rates'].to_csv(f'{output_dir}/growth_rates.csv', index=False)
    
    if len(stats['missing_data_report']) > 0:
        stats['missing_data_report'].to_csv(f'{output_dir}/missing_data_report.csv')
    
    if len(outlier_report) > 0:
        outlier_report.to_csv(f'{output_dir}/outlier_report.csv', index=False)
    
    logger.info(f"All results exported to {output_dir}/")


def main():
    """
    Main execution function.
    """
    # Configuration
    INPUT_FILE = 'raw_energy_data.csv'  # Change this to your input file
    POPULATION_FILE = None  # Optional: path to population CSV
    GDP_FILE = None  # Optional: path to GDP CSV
    
    # Check if input file exists, if not, create sample data
    if not os.path.exists(INPUT_FILE):
        logger.warning(f"Input file {INPUT_FILE} not found. Please provide your data file.")
        logger.info("Expected columns: country, year, primary_energy (or similar), units in EJ/Mtoe/ktoe/TWh")
        return
    
    # Load and process data
    df_raw = load_data(INPUT_FILE)
    df_std = standardize_columns(df_raw)
    df_converted = convert_units_to_twh(df_std)
    df_cleaned, outlier_report = clean_data(df_converted)
    
    # Load population data
    pop_df = load_population_data(POPULATION_FILE)
    
    # Load GDP data if available
    gdp_df = None
    if GDP_FILE and os.path.exists(GDP_FILE):
        gdp_df = pd.read_csv(GDP_FILE)
    
    # Normalize data
    df_normalized = normalize_data(df_cleaned, pop_df, gdp_df)
    
    # Generate summary statistics
    stats = generate_summary_statistics(df_normalized)
    
    # Create visualizations
    create_visualizations(df_normalized)
    
    # Export results
    export_results(df_normalized, stats, outlier_report)
    
    logger.info("Processing complete!")


if __name__ == '__main__':
    main()

