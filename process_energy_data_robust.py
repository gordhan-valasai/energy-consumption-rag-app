"""
Global Energy Consumption Audit - ROBUST Python Script
BP Statistical Review Dataset Processing Pipeline

RESEARCH-GRADE VERSION with:
- Explicit unit handling (no magnitude-based auto-detection)
- Schema validation
- Robust outlier detection (MAD/IQR/Z-score)
- Gap-limited interpolation
- Comprehensive audit logging
- No synthetic data generation
- Configuration file support
"""

import pandas as pd
import numpy as np
import logging
import os
import yaml
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import enhanced validation features
try:
    from validation_enhanced import (
        CountryValidator, GDPValidator, DataIntegrityChecker, TimeSeriesValidator,
        create_validation_report
    )
    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATION_AVAILABLE = False
    logger.warning("Enhanced validation module not available. Some features will be skipped.")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(module)s | %(message)s',
    handlers=[
        logging.FileHandler('energy_audit_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data quality flags
@dataclass
class DataQualityFlags:
    ORIGINAL: int = 0
    INTERPOLATED: int = 1
    FLAGGED_MISSING_BLOCK: int = 2
    REMOVED_OUTLIER: int = 3
    INVALID_UNIT: int = 4
    SCHEMA_VIOLATION: int = 5

FLAGS = DataQualityFlags()


class SchemaValidator:
    """Validates data schema and structure."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.schema_config = config.get('schema', {})
        self.errors = []
        self.warnings = []
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate DataFrame against schema."""
        self.errors = []
        self.warnings = []
        
        # Check required columns
        required = self.schema_config.get('required_columns', ['country', 'year'])
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'column_types' in self.schema_config:
            for col, expected_type in self.schema_config['column_types'].items():
                if col in df.columns:
                    if expected_type == 'integer' and not pd.api.types.is_integer_dtype(df[col]):
                        self.warnings.append(f"Column {col} expected integer type")
                    elif expected_type == 'float' and not pd.api.types.is_numeric_dtype(df[col]):
                        self.warnings.append(f"Column {col} expected float type")
                    elif expected_type == 'string' and not pd.api.types.is_string_dtype(df[col]):
                        self.warnings.append(f"Column {col} expected string type")
        
        # Check year range
        if 'year' in df.columns and 'year_range' in self.schema_config:
            year_range = self.schema_config['year_range']
            min_year = year_range.get('min', 1900)
            max_year = year_range.get('max', 2100)
            invalid_years = df[(df['year'] < min_year) | (df['year'] > max_year)]
            if len(invalid_years) > 0:
                self.warnings.append(f"Found {len(invalid_years)} rows with years outside [{min_year}, {max_year}]")
        
        # Check for duplicate country-year pairs
        if 'country' in df.columns and 'year' in df.columns:
            duplicates = df.duplicated(subset=['country', 'year'], keep=False)
            if duplicates.any():
                self.errors.append(f"Found {duplicates.sum()} duplicate country-year pairs")
        
        # Check for missing values in required columns
        for col in required:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    self.errors.append(f"Column {col} has {missing_count} missing values")
        
        return len(self.errors) == 0, self.errors, self.warnings


class UnitConverter:
    """Handles unit conversion with explicit unit specification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.conversions = config.get('unit_conversions', {})
        self.unit_map = {
            'ej': 'EJ_to_TWh',
            'exajoule': 'EJ_to_TWh',
            'mtoe': 'Mtoe_to_TWh',
            'million_tonnes': 'Mtoe_to_TWh',
            'ktoe': 'ktoe_to_TWh',
            'thousand_tonnes': 'ktoe_to_TWh',
            'twh': 'TWh_to_TWh',
            'terawatt_hour': 'TWh_to_TWh',
        }
    
    def detect_unit_from_column_name(self, col_name: str) -> Optional[str]:
        """Detect unit from column name (explicit only, no magnitude guessing)."""
        col_lower = col_name.lower()
        for unit_key, conversion_key in self.unit_map.items():
            if unit_key in col_lower:
                return conversion_key
        return None
    
    def convert_column(self, df: pd.DataFrame, col: str, unit: Optional[str] = None) -> pd.Series:
        """
        Convert a column to TWh.
        
        Args:
            df: DataFrame
            col: Column name
            unit: Explicit unit (if None, tries to detect from column name)
        
        Returns:
            Series in TWh
        """
        if unit is None:
            unit = self.detect_unit_from_column_name(col)
        
        if unit is None:
            logger.warning(f"Could not determine unit for column {col}. Assuming TWh.")
            return df[col]
        
        if unit not in self.conversions:
            logger.error(f"Unknown unit conversion: {unit}")
            raise ValueError(f"Unknown unit conversion: {unit}")
        
        conversion_factor = self.conversions[unit]
        converted = df[col] * conversion_factor
        
        logger.info(f"Converted {col} from {unit} to TWh (factor: {conversion_factor})")
        return converted
    
    def create_unified_column(self, df: pd.DataFrame, energy_columns: List[str]) -> pd.Series:
        """
        Create unified energy column from multiple columns.
        Uses first non-null value per row (assumes same data in different units).
        """
        if len(energy_columns) == 0:
            raise ValueError("No energy columns found")
        
        if len(energy_columns) == 1:
            return df[energy_columns[0]]
        
        # Convert all columns to TWh first
        converted_cols = {}
        for col in energy_columns:
            converted_cols[col] = self.convert_column(df, col)
        
        # Use first non-null value per row
        result = pd.Series(index=df.index, dtype=float)
        for idx in df.index:
            for col in energy_columns:
                val = converted_cols[col].loc[idx]
                if pd.notna(val):
                    result.loc[idx] = val
                    break
        
        logger.info(f"Created unified column from {len(energy_columns)} energy columns")
        return result


class RobustOutlierDetector:
    """Robust outlier detection using MAD, IQR, or Z-score."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.method = config.get('data_quality', {}).get('outlier_method', 'mad')
        self.mad_multiplier = config.get('data_quality', {}).get('mad_multiplier', 3.0)
        self.iqr_multiplier = config.get('data_quality', {}).get('iqr_multiplier', 1.5)
        self.zscore_threshold = config.get('data_quality', {}).get('zscore_threshold', 3.0)
        self.fixed_threshold = config.get('data_quality', {}).get('max_energy_twh', 50000)
    
    def detect_mad(self, series: pd.Series) -> pd.Series:
        """Median Absolute Deviation method."""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return pd.Series(False, index=series.index)
        threshold = median + self.mad_multiplier * mad
        return series > threshold
    
    def detect_iqr(self, series: pd.Series) -> pd.Series:
        """Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return pd.Series(False, index=series.index)
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_zscore(self, series: pd.Series) -> pd.Series:
        """Z-score method."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        z_scores = np.abs((series - mean) / std)
        return z_scores > self.zscore_threshold
    
    def detect_fixed(self, series: pd.Series) -> pd.Series:
        """Fixed threshold method."""
        return (series < 0) | (series > self.fixed_threshold)
    
    def detect(self, series: pd.Series, group_by: Optional[pd.Series] = None) -> pd.Series:
        """
        Detect outliers in series.
        
        Args:
            series: Energy values
            group_by: Optional grouping (e.g., by country) for per-group detection
        
        Returns:
            Boolean series indicating outliers
        """
        if group_by is not None:
            # Detect outliers per group
            outlier_mask = pd.Series(False, index=series.index)
            for group_val in group_by.unique():
                group_mask = group_by == group_val
                group_series = series[group_mask]
                if len(group_series) > 0 and group_series.notna().sum() > 0:
                    if self.method == 'mad':
                        group_outliers = self.detect_mad(group_series)
                    elif self.method == 'iqr':
                        group_outliers = self.detect_iqr(group_series)
                    elif self.method == 'zscore':
                        group_outliers = self.detect_zscore(group_series)
                    else:
                        group_outliers = self.detect_fixed(group_series)
                    # Ensure boolean dtype and proper indexing
                    group_outliers_series = pd.Series(group_outliers, index=group_series.index, dtype=bool)
                    outlier_mask.loc[group_mask] = group_outliers_series
            return outlier_mask.astype(bool)
        else:
            # Detect outliers globally
            if self.method == 'mad':
                return self.detect_mad(series)
            elif self.method == 'iqr':
                return self.detect_iqr(series)
            elif self.method == 'zscore':
                return self.detect_zscore(series)
            else:
                return self.detect_fixed(series)


class SafeInterpolator:
    """Safe interpolation with gap limits."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_gap = config.get('data_quality', {}).get('max_interpolation_gap', 3)
        self.min_years = config.get('data_quality', {}).get('min_years_for_interpolation', 2)
    
    def interpolate(self, df: pd.DataFrame, value_col: str, 
                   group_col: str = 'country', year_col: str = 'year') -> pd.DataFrame:
        """
        Interpolate missing values with gap limits.
        
        Only interpolates gaps <= max_gap years.
        """
        df = df.copy()
        df = df.sort_values([group_col, year_col])
        
        if value_col not in df.columns:
            return df
        
        # Initialize flag column if needed
        if 'data_quality_flag' not in df.columns:
            df['data_quality_flag'] = FLAGS.ORIGINAL
        
        # Process each group
        for group_val in df[group_col].unique():
            group_mask = df[group_col] == group_val
            group_data = df[group_mask].copy()
            group_indices = df[group_mask].index
            
            if len(group_data) < self.min_years:
                logger.warning(f"Group {group_val} has < {self.min_years} years, skipping interpolation")
                continue
            
            # Find gaps
            values = group_data[value_col].values
            years = group_data[year_col].values
            
            # Identify gaps
            for i in range(len(values) - 1):
                if pd.isna(values[i]) or pd.isna(values[i+1]):
                    continue
                
                gap_size = years[i+1] - years[i] - 1
                
                if gap_size > 0 and gap_size <= self.max_gap:
                    # Interpolate this gap
                    start_idx = group_indices[i]
                    end_idx = group_indices[i+1]
                    
                    start_val = values[i]
                    end_val = values[i+1]
                    
                    # Linear interpolation
                    for j, gap_year in enumerate(range(years[i] + 1, years[i+1]), 1):
                        weight = j / (gap_size + 1)
                        interpolated_val = start_val + weight * (end_val - start_val)
                        
                        # Find row index for this year
                        year_mask = (df[group_col] == group_val) & (df[year_col] == gap_year)
                        if year_mask.any():
                            idx = df[year_mask].index[0]
                            df.loc[idx, value_col] = interpolated_val
                            df.loc[idx, 'data_quality_flag'] = FLAGS.INTERPOLATED
                            logger.debug(f"Interpolated {group_val} year {gap_year}: {interpolated_val:.2f} TWh")
        
        return df


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}


def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """Load data with encoding handling."""
    logger.info(f"Loading data from {file_path}")
    
    try:
        if file_path.endswith('.csv'):
            # Try multiple encodings
            for enc in [encoding, 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    logger.info(f"Successfully loaded with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Could not decode file with any encoding")
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check for duplicates
    duplicates = df.duplicated()
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate rows")
        df = df.drop_duplicates()
    
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names preserving meaning."""
    logger.info("Standardizing column names")
    
    df = df.copy()
    
    # Convert to lowercase first, then clean
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('-', '_')
    # Remove special chars but preserve underscores
    df.columns = df.columns.str.replace(r'[^a-z0-9_]', '', regex=True)
    
    # Explicit mappings (preserve unit information)
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


def normalize_country_names(df: pd.DataFrame, mapping_file: Optional[str] = None) -> pd.DataFrame:
    """Normalize country names using mapping file if provided."""
    if mapping_file and os.path.exists(mapping_file):
        mapping_df = pd.read_csv(mapping_file)
        if 'original' in mapping_df.columns and 'normalized' in mapping_df.columns:
            mapping_dict = dict(zip(mapping_df['original'], mapping_df['normalized']))
            df['country'] = df['country'].map(mapping_dict).fillna(df['country'])
            logger.info(f"Applied country name mapping from {mapping_file}")
    return df


def load_population_data(file_path: Optional[str], config: Dict) -> Optional[pd.DataFrame]:
    """Load population data with validation."""
    if file_path and os.path.exists(file_path):
        logger.info(f"Loading population data from {file_path}")
        pop_df = pd.read_csv(file_path)
        pop_df.columns = pop_df.columns.str.lower().str.replace(' ', '_')
        
        # Validate required columns
        required = config.get('population', {}).get('required_columns', ['country', 'year', 'population'])
        missing = [col for col in required if col not in pop_df.columns]
        if missing:
            logger.error(f"Population file missing required columns: {missing}")
            return None
        
        return pop_df
    else:
        behavior = config.get('population', {}).get('missing_population_behavior', 'warn')
        if behavior == 'error':
            raise ValueError("Population file required but not provided")
        elif behavior == 'warn':
            logger.warning("Population file not provided - per-capita calculations will be skipped")
        return None


def main():
    """Main execution with full validation."""
    # Load configuration
    config = load_config()
    
    # Set logging level
    log_level = config.get('logging', {}).get('level', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Configuration
    INPUT_FILE = 'raw_energy_data.csv'
    POPULATION_FILE = None
    GDP_FILE = None
    
    # Create audit log
    audit_log = {
        'timestamp': datetime.now().isoformat(),
        'input_file': INPUT_FILE,
        'config': config,
        'steps': []
    }
    
    try:
        # Step 1: Load data
        logger.info("=" * 80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 80)
        df_raw = load_data(INPUT_FILE)
        audit_log['steps'].append({'step': 'load_data', 'rows': len(df_raw), 'columns': len(df_raw.columns)})
        
        # Step 2: Schema validation
        logger.info("=" * 80)
        logger.info("STEP 2: SCHEMA VALIDATION")
        logger.info("=" * 80)
        validator = SchemaValidator(config)
        is_valid, errors, warnings = validator.validate(df_raw)
        
        for error in errors:
            logger.error(f"SCHEMA ERROR: {error}")
        for warning in warnings:
            logger.warning(f"SCHEMA WARNING: {warning}")
        
        if not is_valid:
            logger.error("Schema validation failed. Please fix errors before proceeding.")
            return
        
        audit_log['steps'].append({
            'step': 'schema_validation',
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings
        })
        
        # Step 3: Standardize columns
        logger.info("=" * 80)
        logger.info("STEP 3: COLUMN STANDARDIZATION")
        logger.info("=" * 80)
        df_std = standardize_columns(df_raw)
        df_std = normalize_country_names(df_std, config.get('schema', {}).get('country_name_mapping_file'))
        
        # Enhanced validation: Country ISO codes
        if ENHANCED_VALIDATION_AVAILABLE and 'country' in df_std.columns:
            logger.info("Applying country ISO code validation...")
            country_validator = CountryValidator()
            df_std = country_validator.validate_countries(df_std)
            audit_log['steps'].append({
                'step': 'country_validation',
                'unmapped_countries': country_validator.unmapped_countries
            })
        
        # Data integrity checks
        integrity_checker = DataIntegrityChecker()
        input_checksum = integrity_checker.compute_checksum(df_std, 'standardized_data')
        audit_log['steps'].append({
            'step': 'data_integrity',
            'input_checksum': input_checksum
        })
        
        # Step 4: Unit conversion (EXPLICIT ONLY)
        logger.info("=" * 80)
        logger.info("STEP 4: UNIT CONVERSION (EXPLICIT)")
        logger.info("=" * 80)
        converter = UnitConverter(config)
        
        # Find energy columns
        exclude_cols = ['country', 'year', 'data_quality_flag', 'region']
        energy_cols = [col for col in df_std.columns 
                      if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_std[col])]
        
        if len(energy_cols) == 0:
            logger.error("No energy columns found!")
            return
        
        # Convert to TWh
        df_std['total_energy_twh'] = converter.create_unified_column(df_std, energy_cols)
        
        # Step 5: Outlier detection
        logger.info("=" * 80)
        logger.info("STEP 5: OUTLIER DETECTION")
        logger.info("=" * 80)
        detector = RobustOutlierDetector(config)
        outlier_mask = detector.detect(df_std['total_energy_twh'], df_std['country'])
        
        outlier_count = outlier_mask.sum()
        logger.info(f"Detected {outlier_count} outliers using {detector.method} method")
        
        if 'data_quality_flag' not in df_std.columns:
            df_std['data_quality_flag'] = FLAGS.ORIGINAL
        
        df_std.loc[outlier_mask, 'data_quality_flag'] = FLAGS.REMOVED_OUTLIER
        df_std.loc[outlier_mask, 'total_energy_twh'] = np.nan
        
        # Step 6: Safe interpolation
        logger.info("=" * 80)
        logger.info("STEP 6: SAFE INTERPOLATION")
        logger.info("=" * 80)
        interpolator = SafeInterpolator(config)
        df_interpolated = interpolator.interpolate(df_std, 'total_energy_twh')
        
        # Step 7: Population normalization
        logger.info("=" * 80)
        logger.info("STEP 7: POPULATION NORMALIZATION")
        logger.info("=" * 80)
        pop_df = load_population_data(POPULATION_FILE, config)
        if pop_df is not None:
            df_interpolated = df_interpolated.merge(
                pop_df[['country', 'year', 'population']],
                on=['country', 'year'],
                how='left'
            )
            df_interpolated['energy_per_capita_twh'] = (
                df_interpolated['total_energy_twh'] / df_interpolated['population']
            )
            df_interpolated['energy_per_capita_twh'] = df_interpolated['energy_per_capita_twh'].replace(
                [np.inf, -np.inf], np.nan
            )
            logger.info("Added energy_per_capita_twh column")
        else:
            logger.warning("Skipping per-capita calculations (no population data)")
        
        # Enhanced validation: Time series consistency
        if ENHANCED_VALIDATION_AVAILABLE and 'country' in df_interpolated.columns:
            logger.info("=" * 80)
            logger.info("STEP 7.5: TIME SERIES VALIDATION")
            logger.info("=" * 80)
            ts_validator = TimeSeriesValidator()
            
            # Check continuity
            continuity_issues = ts_validator.check_continuity(
                df_interpolated, 'country', 'year', 'total_energy_twh',
                max_gap=config.get('data_quality', {}).get('max_interpolation_gap', 3)
            )
            if continuity_issues:
                logger.warning(f"Found continuity issues in {len(continuity_issues)} countries")
            
            # Check for structural breaks
            structural_breaks = ts_validator.check_structural_breaks(
                df_interpolated, 'country', 'year', 'total_energy_twh',
                break_years=[2008, 2020]
            )
            if structural_breaks:
                logger.info(f"Detected structural breaks in {len(structural_breaks)} countries")
            
            audit_log['steps'].append({
                'step': 'time_series_validation',
                'continuity_issues': continuity_issues,
                'structural_breaks': structural_breaks
            })
        
        # GDP validation if GDP data available
        if ENHANCED_VALIDATION_AVAILABLE and GDP_FILE and os.path.exists(GDP_FILE):
            logger.info("Validating GDP data...")
            gdp_df = pd.read_csv(GDP_FILE)
            gdp_validator = GDPValidator()
            gdp_issues = gdp_validator.validate_gdp_data(gdp_df)
            audit_log['steps'].append({
                'step': 'gdp_validation',
                'issues': gdp_issues
            })
        
        # Step 8: Final integrity check
        output_checksum = integrity_checker.compute_checksum(df_interpolated, 'final_data')
        integrity_checker.check_row_count_consistency(df_raw, df_interpolated, 'Final processing')
        range_check = integrity_checker.check_value_range_consistency(
            df_interpolated, 'total_energy_twh',
            expected_min=0,
            expected_max=config.get('data_quality', {}).get('max_energy_twh', 50000) * 5
        )
        audit_log['steps'].append({
            'step': 'final_integrity_check',
            'output_checksum': output_checksum,
            'value_range_check': range_check
        })
        
        # Step 9: Export
        logger.info("=" * 80)
        logger.info("STEP 9: EXPORT")
        logger.info("=" * 80)
        os.makedirs('output_robust', exist_ok=True)
        df_interpolated.to_csv('output_robust/cleaned_energy_data.csv', index=False)
        
        # Export audit log
        import json
        with open('output_robust/audit_log.json', 'w') as f:
            json.dump(audit_log, f, indent=2, default=str)
        
        # Export integrity report
        integrity_checker.export_integrity_report('output_robust/integrity_report.json')
        
        # Export validation report if enhanced validation was used
        if ENHANCED_VALIDATION_AVAILABLE:
            validators_dict = {}
            if 'country_validator' in locals():
                validators_dict['country'] = country_validator
            if 'gdp_validator' in locals():
                validators_dict['gdp'] = gdp_validator
            if validators_dict:
                create_validation_report(validators_dict, 'output_robust/validation_report.json')
        
        logger.info("Processing complete!")
        logger.info(f"Results saved to output_robust/")
        logger.info(f"Input checksum: {input_checksum[:16]}...")
        logger.info(f"Output checksum: {output_checksum[:16]}...")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

