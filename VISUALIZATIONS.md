# Visualization Guide

## Overview

This document describes all the visualizations created by the `create_visualizations.py` script.

## Generated Visualizations

### 1. Time Series Plots

#### `01_time_series_top10.png`
- **Type**: Line chart
- **Content**: Energy consumption trends for top 10 countries over time
- **Features**: 
  - Multiple colored lines with markers
  - Legend with country names
  - Grid for easy reading
  - High-resolution (300 DPI)

#### `02_growth_trends_panel.png`
- **Type**: Panel of 8 subplots
- **Content**: Individual time series for top 8 countries (2000-2024)
- **Features**: 
  - One country per subplot
  - Consistent styling across panels
  - Easy comparison of trends

### 2. Bar Charts

#### `03_top15_countries_bar.png`
- **Type**: Horizontal bar chart
- **Content**: Top 15 countries by total energy consumption (latest year)
- **Features**: 
  - Color gradient (viridis colormap)
  - Value labels on bars
  - Grid lines for reference

#### `04_per_capita_top20.png`
- **Type**: Horizontal bar chart
- **Content**: Top 20 countries by energy consumption per capita
- **Features**: 
  - Plasma colormap
  - Per-capita values displayed
  - Clean, readable layout

#### `05_total_vs_percapita_comparison.png`
- **Type**: Side-by-side bar charts
- **Content**: Comparison of top 10 by total vs per-capita consumption
- **Features**: 
  - Two panels for easy comparison
  - Shows how rankings differ between metrics

### 3. Growth Analysis

#### `06_growth_rates_comparison.png`
- **Type**: Side-by-side horizontal bar charts
- **Content**: 
  - Left: Top 15 fastest growing countries
  - Right: Slowest growing/declining countries
- **Features**: 
  - Green colors for growth, red for decline
  - Zero line reference
  - Annual growth rate percentages

#### `07_growth_vs_absolute_change.png`
- **Type**: Scatter plot
- **Content**: Growth rate vs absolute change in energy consumption
- **Features**: 
  - Color-coded by latest value
  - Top 5 countries labeled
  - Zero line reference
  - Colorbar for value scale

### 4. Data Quality

#### `08_data_quality_flags.png`
- **Type**: Vertical bar chart
- **Content**: Distribution of data quality flags
- **Features**: 
  - Color-coded by flag type
  - Percentage labels
  - Shows data processing status

#### `09_missing_data_by_country.png` (if applicable)
- **Type**: Horizontal bar chart
- **Content**: Countries with missing data
- **Features**: 
  - Shows data completeness
  - Helps identify data gaps

### 5. Heatmaps

#### `10_consumption_heatmap.png`
- **Type**: Heatmap
- **Content**: Energy consumption for top 20 countries over all years
- **Features**: 
  - Color intensity represents consumption level
  - Easy to spot trends and patterns
  - YlOrRd colormap

### 6. Statistical Analysis

#### `11_correlation_matrix.png`
- **Type**: Correlation heatmap
- **Content**: Correlation between energy variables
- **Features**: 
  - Coolwarm colormap (red=positive, blue=negative)
  - Annotated with correlation values
  - Shows relationships between variables

#### `12_distribution_analysis.png`
- **Type**: Histogram and box plot
- **Content**: Statistical distribution of energy consumption
- **Features**: 
  - Histogram with median line
  - Box plot showing quartiles
  - Statistical summary visualization

### 7. Temporal Evolution

#### `14_temporal_evolution.png`
- **Type**: Two-panel time series
- **Content**: 
  - Top: Global total energy consumption over time
  - Bottom: Year-over-year growth rate
- **Features**: 
  - Filled area under curve
  - Color-coded growth bars (green=positive, red=negative)
  - Shows global trends

## Usage

### Basic Usage
```bash
python create_visualizations.py
```

### With Custom Input File
```bash
python create_visualizations.py output/cleaned_energy_data.csv
```

### From Python Script
```python
from create_visualizations import create_all_visualizations
import pandas as pd

df = pd.read_csv('output/cleaned_energy_data.csv')
create_all_visualizations(df=df, output_dir='my_figures')
```

## Customization

### Change Output Directory
```python
create_all_visualizations(output_dir='custom_figures')
```

### Create Specific Visualizations
```python
from create_visualizations import (
    plot_time_series_comprehensive,
    plot_bar_charts_comprehensive,
    plot_growth_analysis
)

df = pd.read_csv('output/cleaned_energy_data.csv')
plot_time_series_comprehensive(df, 'figures')
plot_bar_charts_comprehensive(df, 'figures')
plot_growth_analysis(df, 'figures')
```

## Styling

All visualizations use:
- **DPI**: 300 (publication quality)
- **Style**: Seaborn whitegrid (or default fallback)
- **Color Palette**: Husl (diverse, accessible colors)
- **Font Sizes**: Optimized for readability
- **Grid**: Subtle grid lines for reference

## File Naming Convention

Files are numbered sequentially:
- `01_` - Time series
- `02_` - Growth trends
- `03_` - Bar charts
- `04_` - Per-capita
- `05_` - Comparisons
- `06_` - Growth rates
- `07_` - Scatter plots
- `08_` - Data quality
- `09_` - Missing data
- `10_` - Heatmaps
- `11_` - Correlations
- `12_` - Distributions
- `13_` - Regional (if available)
- `14_` - Temporal evolution

## Requirements

- pandas
- matplotlib
- seaborn
- numpy

All included in `requirements.txt`.

## Tips

1. **For Presentations**: Use `01_time_series_top10.png` and `03_top15_countries_bar.png`
2. **For Analysis**: Use `06_growth_rates_comparison.png` and `07_growth_vs_absolute_change.png`
3. **For Reports**: Use `11_correlation_matrix.png` and `12_distribution_analysis.png`
4. **For Overview**: Use `14_temporal_evolution.png` and `10_consumption_heatmap.png`

## Troubleshooting

### No figures generated
- Check that processed data file exists (`output/cleaned_energy_data.csv`)
- Run the processing pipeline first: `python process_energy_data.py`

### Missing visualizations
- Some visualizations require specific columns (e.g., `energy_per_capita_twh`, `region`)
- Check data columns: `df.columns.tolist()`

### Low quality images
- All images are saved at 300 DPI by default
- Check `plt.rcParams['savefig.dpi']` if quality is low

## Future Enhancements

Potential additions:
- Interactive Plotly charts
- Animated time series
- 3D visualizations
- Custom country groupings
- Export to PDF report

