# Global Energy Consumption Audit - BP Statistical Review Dataset

A comprehensive data processing pipeline for analyzing global energy consumption data from the BP Statistical Review (or similar datasets). This project provides both **Python** (Pandas) and **Julia** (DataFrames.jl) implementations with full reproducibility.

## ⚠️ Important: Two Versions Available

### 🚀 **Robust Version** (Recommended for Research)
- **File**: `process_energy_data_robust.py`
- **Features**: Explicit unit handling, schema validation, robust outlier detection, gap-limited interpolation, comprehensive audit logging
- **Use for**: Research publications, policy analysis, peer-reviewed work
- **See**: `IMPROVEMENTS.md` for full details

### 📝 **Original Version** (Quick Prototyping)
- **File**: `process_energy_data.py`
- **Features**: Auto-detection, quick setup, basic validation
- **Use for**: Quick exploration, prototyping, learning

**For serious analysis, use the robust version.**

## Overview

This pipeline processes raw energy consumption data through the following stages:

1. **Data Import & Standardization** - Loads CSV/Excel files and standardizes column names
2. **Unit Conversion** - Converts all energy units (EJ, Mtoe, ktoe) to TWh
3. **Data Cleaning** - Handles missing values, interpolates gaps, flags issues, removes outliers
4. **Normalization** - Normalizes by population and GDP (if available)
5. **Summary Statistics** - Generates comprehensive reports
6. **Visualization** - Creates publication-ready plots
7. **Export** - Saves cleaned data and reports

## Features

- ✅ **Dual Language Support**: Python (Pandas) and Julia (DataFrames.jl)
- ✅ **Automatic Unit Detection**: Handles EJ, Mtoe, ktoe, and TWh
- ✅ **Intelligent Interpolation**: Linear interpolation for missing years
- ✅ **Outlier Detection**: Flags and removes impossible values
- ✅ **Data Quality Tracking**: Flags for original, interpolated, missing, and outlier data
- ✅ **Per-Capita Normalization**: Energy consumption normalized by population
- ✅ **GDP Normalization**: Energy intensity calculations (if GDP data available)
- ✅ **Comprehensive Reports**: Top consumers, growth rates, missing data analysis
- ✅ **Publication-Ready Visualizations**: Time series, scatter plots, heatmaps, bar charts

## Project Structure

```
.
├── process_energy_data.py      # Python processing script
├── process_energy_data.jl      # Julia processing script
├── generate_sample_data.py     # Sample data generator (for testing)
├── README.md                   # This file
├── raw_energy_data.csv         # Input data file (you provide this)
├── output/                     # Generated output directory
│   ├── cleaned_energy_data.csv
│   ├── summary_statistics.txt
│   ├── top_10_consumers.csv
│   ├── growth_rates.csv
│   └── missing_data_report.csv
└── figures/                    # Generated visualizations
    ├── time_series_top_countries.png
    ├── energy_per_capita_bar.png
    ├── top_countries_bar.png
    └── ...
```

## Requirements

### Python Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly openpyxl
```

Or install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Python Packages:**
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- plotly >= 5.14.0
- openpyxl >= 3.0.0 (for Excel support)

### Julia Requirements

```julia
using Pkg
Pkg.add(["DataFrames", "CSV", "Statistics", "Plots", "StatsPlots"])
```

**Julia Packages:**
- DataFrames.jl >= 1.5.0
- CSV.jl >= 0.10.0
- Statistics.jl (standard library)
- Plots.jl >= 1.38.0
- StatsPlots.jl >= 0.15.0

## Quick Start

### 1. Prepare Your Data

Your input CSV/Excel file should contain:
- `Country` (or `country`) - Country names
- `Year` (or `year`) - Year values
- Energy consumption column(s) with units in:
  - EJ (Exajoules)
  - Mtoe (Million tonnes of oil equivalent)
  - ktoe (Thousand tonnes of oil equivalent)
  - TWh (Terawatt-hours)

**Example input format:**
```csv
Country,Year,Primary Energy Consumption (EJ)
United States,2000,100.5
China,2000,80.2
...
```

### 2. Generate Sample Data (Optional)

To test the pipeline with sample data:

```bash
python generate_sample_data.py
```

This creates `raw_energy_data.csv` with realistic sample data.

### 3. Run Python Script

```bash
python process_energy_data.py
```

**Configuration:**
Edit the `main()` function in `process_energy_data.py`:
```python
INPUT_FILE = 'raw_energy_data.csv'  # Your input file
POPULATION_FILE = None  # Optional: path to population CSV
GDP_FILE = None  # Optional: path to GDP CSV
```

### 4. Run Julia Script

```julia
julia process_energy_data.jl
```

Or in Julia REPL:
```julia
include("process_energy_data.jl")
main()
```

**Configuration:**
Edit the `main()` function in `process_energy_data.jl`:
```julia
INPUT_FILE = "raw_energy_data.csv"  # Your input file
POPULATION_FILE = nothing  # Optional: path to population CSV
GDP_FILE = nothing  # Optional: path to GDP CSV
```

## Unit Conversion

The scripts automatically convert energy units using:

- **1 EJ = 277.778 TWh**
- **1 Mtoe = 11.63 TWh**
- **1 ktoe = 0.01163 TWh**

Unit detection is performed by:
1. Column name analysis (e.g., "Primary Energy (EJ)")
2. Data magnitude analysis (fallback method)

## Data Quality Flags

The pipeline tracks data quality with flags:

- **0 = original** - Original data, no modifications
- **1 = interpolated** - Missing values filled by linear interpolation
- **2 = flagged_missing_block** - Entire country block missing
- **3 = removed_outlier** - Outlier removed (negative or >50,000 TWh)

## Output Files

### Cleaned Data
- `output/cleaned_energy_data.csv` - Full cleaned dataset with all normalized columns

### Summary Statistics
- `output/summary_statistics.txt` - Human-readable summary report
- `output/top_10_consumers.csv` - Top 10 energy consumers (latest year)
- `output/growth_rates.csv` - Annual growth rates 2000-2024 by country
- `output/missing_data_report.csv` - Missing data analysis
- `output/outlier_report.csv` - Removed outliers log

### Visualizations
- `figures/time_series_top_countries.png` - Energy consumption trends
- `figures/energy_per_capita_bar.png` - Per-capita consumption ranking
- `figures/top_countries_bar.png` - Total consumption ranking
- `figures/gdp_vs_energy_scatter.png` - GDP vs Energy (if GDP data available)
- `figures/energy_by_region_bar.png` - Regional breakdown (if region column exists)

## Population and GDP Data

### Population Data Format

If providing population data, use this format:
```csv
country,year,population
United States,2000,280.0
China,2000,1400.0
...
```

**Note:** Population should be in millions.

### GDP Data Format

If providing GDP data, use this format:
```csv
country,year,gdp
United States,2000,10250.0
China,2000,1210.0
...
```

**Note:** GDP should be in consistent units (e.g., billion USD).

If population/GDP files are not provided, the scripts will generate sample data for testing purposes.

## Data Cleaning Rules

1. **Partial Missing Years**: Linear interpolation between available data points
2. **Entire Country Missing**: Flagged but not interpolated
3. **Outliers**: Values < 0 or > 50,000 TWh are removed and logged
4. **Unit Standardization**: All energy values converted to TWh

## Logging

Both scripts generate detailed logs:
- **Python**: `energy_audit.log` (file) + console output
- **Julia**: Console output only

Logs include:
- Data loading progress
- Unit conversion details
- Missing data identification
- Interpolation operations
- Outlier detection
- Export confirmations

## Sanity Checks

The scripts include assertions for:
- Unit conversion accuracy
- Reasonable value ranges
- Data completeness
- Interpolation validity

## Example Workflow

```bash
# 1. Generate sample data (or use your own)
python generate_sample_data.py

# 2. Run Python pipeline
python process_energy_data.py

# 3. Check outputs
ls output/
ls figures/

# 4. Run Julia pipeline (for comparison)
julia process_energy_data.jl
```

## Customization

### Adding New Visualizations

**Python:**
Add new plotting functions in `create_visualizations()` function.

**Julia:**
Add new plotting code in `create_visualizations()` function.

### Modifying Cleaning Rules

Edit the `clean_data()` function in either script to adjust:
- Interpolation methods
- Outlier thresholds
- Missing data handling

### Adding New Normalizations

Extend the `normalize_data()` function to add:
- Energy per unit area
- Energy per unit GDP (already included)
- Sector-specific normalizations

## Troubleshooting

### Common Issues

1. **File Not Found**
   - Ensure input file path is correct
   - Check file permissions

2. **Missing Columns**
   - Scripts will attempt to auto-detect columns
   - Check column name standardization in logs

3. **Unit Conversion Errors**
   - Verify unit detection in logs
   - Manually specify units in column names

4. **Memory Issues (Large Datasets)**
   - Use chunking for very large files
   - Consider using Polars instead of Pandas (modify script)

## Performance Notes

- **Python**: Optimized with vectorized Pandas operations
- **Julia**: Uses efficient DataFrames.jl operations
- Both scripts avoid unnecessary loops
- Large datasets (>1M rows) may require memory optimization

## Citation

If using BP Statistical Review data, please cite:
> BP Statistical Review of World Energy, [Year]. Available at: https://www.bp.com/statisticalreview

## License

This code is provided as-is for data processing purposes. Adapt as needed for your use case.

## Contributing

To extend this pipeline:
1. Keep functions modular
2. Add type hints (Python) / type annotations (Julia)
3. Include logging statements
4. Add assertions for data quality
5. Update this README

## Contact

For questions or issues, please refer to the code comments or create an issue in your repository.

---

**Last Updated**: 2024
**Version**: 1.0

