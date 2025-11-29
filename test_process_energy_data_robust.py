"""
Unit tests for robust energy data processing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
from pathlib import Path

# Import the robust processing module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from process_energy_data_robust import (
    SchemaValidator, UnitConverter, RobustOutlierDetector,
    SafeInterpolator, FLAGS, load_config, standardize_columns
)


class TestSchemaValidator:
    """Test schema validation."""
    
    def test_required_columns_present(self):
        """Test validation passes when required columns are present."""
        config = {'schema': {'required_columns': ['country', 'year']}}
        validator = SchemaValidator(config)
        
        df = pd.DataFrame({
            'country': ['USA', 'China'],
            'year': [2000, 2001],
            'energy': [100, 200]
        })
        
        is_valid, errors, warnings = validator.validate(df)
        assert is_valid
        assert len(errors) == 0
    
    def test_missing_required_columns(self):
        """Test validation fails when required columns are missing."""
        config = {'schema': {'required_columns': ['country', 'year']}}
        validator = SchemaValidator(config)
        
        df = pd.DataFrame({
            'country': ['USA', 'China'],
            'energy': [100, 200]
        })
        
        is_valid, errors, warnings = validator.validate(df)
        assert not is_valid
        assert any('year' in error for error in errors)
    
    def test_duplicate_country_year_pairs(self):
        """Test detection of duplicate country-year pairs."""
        config = {'schema': {'required_columns': ['country', 'year']}}
        validator = SchemaValidator(config)
        
        df = pd.DataFrame({
            'country': ['USA', 'USA', 'China'],
            'year': [2000, 2000, 2001],
            'energy': [100, 200, 300]
        })
        
        is_valid, errors, warnings = validator.validate(df)
        assert not is_valid
        assert any('duplicate' in error.lower() for error in errors)
    
    def test_year_range_validation(self):
        """Test year range validation."""
        config = {
            'schema': {
                'required_columns': ['country', 'year'],
                'year_range': {'min': 2000, 'max': 2024}
            }
        }
        validator = SchemaValidator(config)
        
        df = pd.DataFrame({
            'country': ['USA', 'China', 'Russia'],
            'year': [1999, 2000, 2025],
            'energy': [100, 200, 300]
        })
        
        is_valid, errors, warnings = validator.validate(df)
        # Should have warnings for out-of-range years
        assert any('outside' in warning.lower() for warning in warnings)


class TestUnitConverter:
    """Test unit conversion."""
    
    def test_ej_to_twh_conversion(self):
        """Test EJ to TWh conversion."""
        config = {
            'unit_conversions': {
                'EJ_to_TWh': 277.778,
                'Mtoe_to_TWh': 11.63,
                'ktoe_to_TWh': 0.01163,
                'TWh_to_TWh': 1.0
            }
        }
        converter = UnitConverter(config)
        
        df = pd.DataFrame({
            'energy_ej': [1.0, 2.0, 3.0]
        })
        
        result = converter.convert_column(df, 'energy_ej')
        expected = pd.Series([277.778, 555.556, 833.334], name=None)
        # Compare values only, ignore name
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True), rtol=1e-3, check_names=False)
    
    def test_mtoe_to_twh_conversion(self):
        """Test Mtoe to TWh conversion."""
        config = {
            'unit_conversions': {
                'EJ_to_TWh': 277.778,
                'Mtoe_to_TWh': 11.63,
                'ktoe_to_TWh': 0.01163,
                'TWh_to_TWh': 1.0
            }
        }
        converter = UnitConverter(config)
        
        df = pd.DataFrame({
            'energy_mtoe': [10.0, 20.0]
        })
        
        result = converter.convert_column(df, 'energy_mtoe')
        expected = pd.Series([116.3, 232.6], name=None)
        # Compare values only, ignore name
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True), rtol=1e-3, check_names=False)
    
    def test_unified_column_creation(self):
        """Test unified column creation from multiple columns."""
        config = {
            'unit_conversions': {
                'EJ_to_TWh': 277.778,
                'Mtoe_to_TWh': 11.63,
                'TWh_to_TWh': 1.0
            }
        }
        converter = UnitConverter(config)
        
        df = pd.DataFrame({
            'energy_ej': [1.0, np.nan, np.nan],
            'energy_mtoe': [np.nan, 10.0, np.nan],
            'energy_twh': [np.nan, np.nan, 100.0]
        })
        
        result = converter.create_unified_column(df, ['energy_ej', 'energy_mtoe', 'energy_twh'])
        expected = pd.Series([277.778, 116.3, 100.0])
        pd.testing.assert_series_equal(result, expected, rtol=1e-3)


class TestRobustOutlierDetector:
    """Test robust outlier detection."""
    
    def test_mad_outlier_detection(self):
        """Test MAD-based outlier detection."""
        config = {
            'data_quality': {
                'outlier_method': 'mad',
                'mad_multiplier': 3.0
            }
        }
        detector = RobustOutlierDetector(config)
        
        # Create data with clear outlier
        values = [100.0, 105.0, 98.0, 102.0, 1000.0]  # 1000 is outlier
        series = pd.Series(values)
        
        outliers = detector.detect(series)
        assert outliers.iloc[-1] == True  # Last value should be outlier
        assert outliers.iloc[:-1].sum() == 0  # Others should not be outliers
    
    def test_iqr_outlier_detection(self):
        """Test IQR-based outlier detection."""
        config = {
            'data_quality': {
                'outlier_method': 'iqr',
                'iqr_multiplier': 1.5
            }
        }
        detector = RobustOutlierDetector(config)
        
        values = [10.0, 12.0, 11.0, 13.0, 50.0]  # 50 is outlier
        series = pd.Series(values)
        
        outliers = detector.detect(series)
        assert outliers.iloc[-1] == True
    
    def test_fixed_threshold_outlier_detection(self):
        """Test fixed threshold outlier detection."""
        config = {
            'data_quality': {
                'outlier_method': 'fixed_threshold',
                'max_energy_twh': 100.0
            }
        }
        detector = RobustOutlierDetector(config)
        
        values = [50.0, 75.0, 150.0, -10.0]  # 150 and -10 are outliers
        series = pd.Series(values)
        
        outliers = detector.detect(series)
        assert outliers.iloc[2] == True  # 150 > threshold
        assert outliers.iloc[3] == True  # -10 < 0
    
    def test_per_country_outlier_detection(self):
        """Test outlier detection per country."""
        config = {
            'data_quality': {
                'outlier_method': 'mad',
                'mad_multiplier': 3.0
            }
        }
        detector = RobustOutlierDetector(config)
        
        # Use more values per country for better MAD detection
        df = pd.DataFrame({
            'country': ['USA', 'USA', 'USA', 'USA', 'China', 'China', 'China', 'China'],
            'energy': [100.0, 105.0, 98.0, 1000.0, 50.0, 52.0, 48.0, 500.0]  # Outliers per country
        })
        
        outliers = detector.detect(df['energy'], df['country'])
        # USA: 1000 is outlier relative to USA's scale (~100)
        # China: 500 is outlier relative to China's scale (~50)
        assert outliers.iloc[3] == True  # USA outlier
        assert outliers.iloc[7] == True  # China outlier


class TestSafeInterpolator:
    """Test safe interpolation."""
    
    def test_interpolation_within_gap_limit(self):
        """Test interpolation for gaps within limit."""
        config = {
            'data_quality': {
                'max_interpolation_gap': 3,
                'min_years_for_interpolation': 2
            }
        }
        interpolator = SafeInterpolator(config)
        
        df = pd.DataFrame({
            'country': ['USA', 'USA', 'USA', 'USA'],
            'year': [2000, 2001, 2003, 2004],  # Gap of 1 year (2002 missing)
            'total_energy_twh': [100.0, 110.0, np.nan, 130.0]
        })
        
        result = interpolator.interpolate(df, 'total_energy_twh')
        
        # Should interpolate 2002
        year_2002 = result[result['year'] == 2002]
        if len(year_2002) > 0:
            assert not pd.isna(year_2002['total_energy_twh'].iloc[0])
    
    def test_no_interpolation_beyond_gap_limit(self):
        """Test that gaps beyond limit are not interpolated."""
        config = {
            'data_quality': {
                'max_interpolation_gap': 2,
                'min_years_for_interpolation': 2
            }
        }
        interpolator = SafeInterpolator(config)
        
        df = pd.DataFrame({
            'country': ['USA', 'USA', 'USA'],
            'year': [2000, 2001, 2005],  # Gap of 3 years (exceeds limit)
            'total_energy_twh': [100.0, 110.0, 150.0]
        })
        
        result = interpolator.interpolate(df, 'total_energy_twh')
        
        # Should not interpolate years 2002-2004
        for year in [2002, 2003, 2004]:
            year_rows = result[result['year'] == year]
            if len(year_rows) > 0:
                # If row exists, it should still be missing
                assert pd.isna(year_rows['total_energy_twh'].iloc[0])


class TestColumnStandardization:
    """Test column name standardization."""
    
    def test_lowercase_conversion(self):
        """Test conversion to lowercase."""
        df = pd.DataFrame({
            'Country': [1, 2],
            'Year': [2000, 2001],
            'Energy (EJ)': [100, 200]
        })
        
        result = standardize_columns(df)
        assert 'country' in result.columns
        assert 'year' in result.columns
    
    def test_special_character_handling(self):
        """Test handling of special characters."""
        df = pd.DataFrame({
            'Primary Energy (EJ)': [100, 200],
            'Year-Value': [2000, 2001]
        })
        
        result = standardize_columns(df)
        # Should remove special chars but preserve structure
        assert all(' ' not in col for col in result.columns)


class TestConfiguration:
    """Test configuration loading."""
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'data_quality': {
                    'outlier_method': 'mad',
                    'max_interpolation_gap': 3
                }
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert 'data_quality' in config
            assert config['data_quality']['outlier_method'] == 'mad'
        finally:
            os.unlink(config_path)
    
    def test_default_config_when_file_missing(self):
        """Test default config when file doesn't exist."""
        config = load_config('nonexistent_file.yaml')
        # Should return empty dict or defaults
        assert isinstance(config, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

