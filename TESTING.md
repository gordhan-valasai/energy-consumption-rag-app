# Testing and Validation Guide

## Overview

This document describes the testing framework and enhanced validation features for the robust energy data processing pipeline.

## Running Tests

### Python Tests

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
pytest test_process_energy_data_robust.py -v

# Run with coverage
pytest test_process_energy_data_robust.py --cov=process_energy_data_robust --cov-report=html

# Run specific test class
pytest test_process_energy_data_robust.py::TestSchemaValidator -v

# Run specific test
pytest test_process_energy_data_robust.py::TestSchemaValidator::test_required_columns_present -v
```

### Julia Tests

```bash
# Run Julia tests
julia test_process_energy_data_robust.jl

# Or in Julia REPL
include("test_process_energy_data_robust.jl")
```

## Test Coverage

### Schema Validation Tests
- ✅ Required columns present
- ✅ Missing required columns detection
- ✅ Duplicate country-year pairs
- ✅ Year range validation
- ✅ Data type validation

### Unit Conversion Tests
- ✅ EJ to TWh conversion
- ✅ Mtoe to TWh conversion
- ✅ Unified column creation from multiple sources
- ✅ Missing value handling

### Outlier Detection Tests
- ✅ MAD (Median Absolute Deviation) method
- ✅ IQR (Interquartile Range) method
- ✅ Z-score method
- ✅ Fixed threshold method
- ✅ Per-country outlier detection

### Interpolation Tests
- ✅ Interpolation within gap limit
- ✅ No interpolation beyond gap limit
- ✅ Minimum years requirement

### Column Standardization Tests
- ✅ Lowercase conversion
- ✅ Special character handling
- ✅ Column name mapping

## Enhanced Validation Features

### 1. Country ISO Code Validation

The `CountryValidator` class normalizes country names and maps them to ISO 3166-1 alpha-3 codes.

**Usage:**
```python
from validation_enhanced import CountryValidator

validator = CountryValidator()
df_validated = validator.validate_countries(df, country_col='country')
# Adds columns: 'country_normalized' and 'iso_code'
```

**Features:**
- Maps common country name variations (e.g., "USA", "United States", "US" → "USA")
- Supports 100+ countries
- Reports unmapped countries for manual review
- Adds ISO codes for data linking

### 2. GDP Validation

The `GDPValidator` class validates GDP data for currency, units, and consistency.

**Usage:**
```python
from validation_enhanced import GDPValidator

validator = GDPValidator()
issues = validator.validate_gdp_data(
    df, 
    gdp_col='gdp',
    currency_col='currency',  # Optional
    unit_col='unit_type'      # Optional
)
```

**Checks:**
- Negative GDP values (error)
- Missing GDP values (warning)
- Unknown currency codes (warning)
- Unrealistic values (>100 trillion, warning)
- Unit type validation (nominal/real/PPP)

### 3. Data Integrity Checksums

The `DataIntegrityChecker` class computes SHA256 checksums for reproducibility.

**Usage:**
```python
from validation_enhanced import DataIntegrityChecker

checker = DataIntegrityChecker()

# Compute checksum
checksum = checker.compute_checksum(df, 'my_dataset')

# Verify checksum
is_valid = checker.verify_checksum(df, 'my_dataset', expected_checksum)

# Check row count consistency
checker.check_row_count_consistency(df_before, df_after, 'operation_name')

# Check value ranges
range_check = checker.check_value_range_consistency(
    df, 'total_energy_twh',
    expected_min=0,
    expected_max=50000
)
```

**Features:**
- SHA256 checksums for all datasets
- Row count tracking
- Value range validation
- Export integrity reports to JSON

### 4. Time Series Validation

The `TimeSeriesValidator` class checks time series consistency and continuity.

**Usage:**
```python
from validation_enhanced import TimeSeriesValidator

validator = TimeSeriesValidator()

# Check continuity (gaps in time series)
continuity_issues = validator.check_continuity(
    df, 'country', 'year', 'total_energy_twh',
    max_gap=3
)

# Check monotonicity (sudden large changes)
monotonicity_violations = validator.check_monotonicity(
    df, 'country', 'year', 'total_energy_twh',
    allow_decrease=True
)

# Check structural breaks (e.g., 2008 crisis, 2020 COVID)
structural_breaks = validator.check_structural_breaks(
    df, 'country', 'year', 'total_energy_twh',
    break_years=[2008, 2020]
)
```

**Features:**
- Gap detection and reporting
- Large change detection (>50% change)
- Structural break identification
- Per-country analysis

## Integration with Robust Pipeline

The enhanced validation features are automatically integrated into `process_energy_data_robust.py`:

1. **Country Validation**: Applied after column standardization
2. **Data Integrity**: Checksums computed at each major step
3. **Time Series Validation**: Applied after interpolation
4. **GDP Validation**: Applied if GDP file is provided

All validation results are included in the audit log (`output_robust/audit_log.json`).

## Output Files

### Audit Log (`audit_log.json`)
Contains:
- Timestamp
- Configuration used
- Step-by-step processing log
- Validation results
- Checksums

### Integrity Report (`integrity_report.json`)
Contains:
- All computed checksums
- Integrity check results
- Value range validations

### Validation Report (`validation_report.json`)
Contains:
- Country validation results
- GDP validation issues
- Time series validation results

## Example Test Output

```
test_process_energy_data_robust.py::TestSchemaValidator::test_required_columns_present PASSED
test_process_energy_data_robust.py::TestSchemaValidator::test_missing_required_columns PASSED
test_process_energy_data_robust.py::TestUnitConverter::test_ej_to_twh_conversion PASSED
test_process_energy_data_robust.py::TestRobustOutlierDetector::test_mad_outlier_detection PASSED
...

======================== 15 passed in 2.34s ========================
```

## Continuous Integration

To set up CI/CD testing:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest test_process_energy_data_robust.py --cov --cov-report=xml
```

## Best Practices

1. **Run tests before committing**: `pytest test_process_energy_data_robust.py`
2. **Check coverage**: Aim for >80% code coverage
3. **Add tests for new features**: Write tests before implementing features
4. **Review validation reports**: Check `output_robust/validation_report.json` after processing
5. **Verify checksums**: Compare checksums across runs for reproducibility

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Make sure you're in the project directory
cd /path/to/project
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest test_process_energy_data_robust.py
```

### Julia Tests Fail
```julia
# Make sure all packages are installed
using Pkg
Pkg.add(["Test", "DataFrames", "CSV", "Statistics", "YAML", "JSON"])
```

### Validation Reports Empty
- Check that `validation_enhanced.py` is in the same directory
- Verify that enhanced validation is enabled in config
- Check logs for import warnings

## Future Enhancements

- [ ] Integration tests with real BP Statistical Review data
- [ ] Performance benchmarks
- [ ] Property-based testing (Hypothesis for Python, QuickCheck for Julia)
- [ ] Visual regression tests for plots
- [ ] End-to-end pipeline tests

