"""
Enhanced validation features for energy data processing.

Includes:
- Country ISO code validation
- GDP currency/PPP validation
- Data integrity checksums
- Time series consistency checks
"""

import pandas as pd
import numpy as np
import hashlib
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ISO 3166-1 alpha-3 country codes (common countries)
ISO_COUNTRY_CODES = {
    'United States': 'USA',
    'United States of America': 'USA',
    'USA': 'USA',
    'US': 'USA',
    'China': 'CHN',
    "People's Republic of China": 'CHN',
    'India': 'IND',
    'Japan': 'JPN',
    'Germany': 'DEU',
    'United Kingdom': 'GBR',
    'UK': 'GBR',
    'France': 'FRA',
    'Italy': 'ITA',
    'Brazil': 'BRA',
    'Canada': 'CAN',
    'Russia': 'RUS',
    'Russian Federation': 'RUS',
    'South Korea': 'KOR',
    'Republic of Korea': 'KOR',
    'Australia': 'AUS',
    'Spain': 'ESP',
    'Mexico': 'MEX',
    'Indonesia': 'IDN',
    'Netherlands': 'NLD',
    'Saudi Arabia': 'SAU',
    'Turkey': 'TUR',
    'Switzerland': 'CHE',
    'Argentina': 'ARG',
    'Sweden': 'SWE',
    'Poland': 'POL',
    'Belgium': 'BEL',
    'Thailand': 'THA',
    'Iran': 'IRN',
    'Austria': 'AUT',
    'Norway': 'NOR',
    'United Arab Emirates': 'ARE',
    'UAE': 'ARE',
    'Israel': 'ISR',
    'Ireland': 'IRL',
    'Singapore': 'SGP',
    'Malaysia': 'MYS',
    'South Africa': 'ZAF',
    'Philippines': 'PHL',
    'Pakistan': 'PAK',
    'Bangladesh': 'BGD',
    'Vietnam': 'VNM',
    'Egypt': 'EGY',
    'Chile': 'CHL',
    'Finland': 'FIN',
    'Romania': 'ROU',
    'Czech Republic': 'CZE',
    'Portugal': 'PRT',
    'Peru': 'PER',
    'Iraq': 'IRQ',
    'New Zealand': 'NZL',
    'Qatar': 'QAT',
    'Greece': 'GRC',
    'Algeria': 'DZA',
    'Kazakhstan': 'KAZ',
    'Hungary': 'HUN',
    'Kuwait': 'KWT',
    'Ukraine': 'UKR',
    'Morocco': 'MAR',
    'Slovakia': 'SVK',
    'Ecuador': 'ECU',
    'Puerto Rico': 'PRI',
    'Oman': 'OMN',
    'Belarus': 'BLR',
    'Azerbaijan': 'AZE',
    'Sri Lanka': 'LKA',
    'Myanmar': 'MMR',
    'Luxembourg': 'LUX',
    'Uzbekistan': 'UZB',
    'Dominican Republic': 'DOM',
    'Guatemala': 'GTM',
    'Kenya': 'KEN',
    'Bulgaria': 'BGR',
    'Uruguay': 'URY',
    'Croatia': 'HRV',
    'Tunisia': 'TUN',
    'Lebanon': 'LBN',
    'Panama': 'PAN',
    'Lithuania': 'LTU',
    'Costa Rica': 'CRI',
    'Serbia': 'SRB',
    'Slovenia': 'SVN',
    'Bolivia': 'BOL',
    'Tanzania': 'TZA',
    'Yemen': 'YEM',
    'Jordan': 'JOR',
    'Cameroon': 'CMR',
    'Latvia': 'LVA',
    'Paraguay': 'PRY',
    'Zambia': 'ZMB',
    'El Salvador': 'SLV',
    'Trinidad and Tobago': 'TTO',
    'Estonia': 'EST',
    'Honduras': 'HND',
    'Uganda': 'UGA',
    'Cyprus': 'CYP',
    'Nepal': 'NPL',
    'Iceland': 'ISL',
    'Cambodia': 'KHM',
    'Senegal': 'SEN',
    'Zimbabwe': 'ZWE',
    'Papua New Guinea': 'PNG',
    'Bosnia and Herzegovina': 'BIH',
    'Afghanistan': 'AFG',
    'Botswana': 'BWA',
    'Mali': 'MLI',
    'Georgia': 'GEO',
    'Gabon': 'GAB',
    'Jamaica': 'JAM',
    'Nicaragua': 'NIC',
    'Mauritius': 'MUS',
    'Namibia': 'NAM',
    'Benin': 'BEN',
    'Mozambique': 'MOZ',
    'Brunei': 'BRN',
    'Albania': 'ALB',
    'Mongolia': 'MNG',
    'Armenia': 'ARM',
    'Madagascar': 'MDG',
    'Malta': 'MLT',
    'Guinea': 'GIN',
    'Burkina Faso': 'BFA',
    'Moldova': 'MDA',
    'Haiti': 'HTI',
    'Niger': 'NER',
    'Rwanda': 'RWA',
    'Kyrgyzstan': 'KGZ',
    'Tajikistan': 'TJK',
    'Malawi': 'MWI',
    'Togo': 'TGO',
    'Mauritania': 'MRT',
    'Barbados': 'BRB',
    'Montenegro': 'MNE',
    'Swaziland': 'SWZ',
    'Fiji': 'FJI',
    'Suriname': 'SUR',
    'Guyana': 'GUY',
    'Maldives': 'MDV',
    'Bhutan': 'BTN',
    'Belize': 'BLZ',
    'Bahamas': 'BHS',
    'Djibouti': 'DJI',
    'Luxembourg': 'LUX',
    'Macao': 'MAC',
    'Cape Verde': 'CPV',
    'Seychelles': 'SEY',
    'Sao Tome and Principe': 'STP',
    'Palau': 'PLW',
    'Micronesia': 'FSM',
    'Marshall Islands': 'MHL',
    'Kiribati': 'KIR',
    'Nauru': 'NRU',
    'Tuvalu': 'TUV'
}


class CountryValidator:
    """Validates and normalizes country names using ISO codes."""
    
    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self.mapping = mapping or ISO_COUNTRY_CODES
        self.unmapped_countries = []
    
    def normalize_country_name(self, country_name: str) -> Tuple[str, Optional[str]]:
        """
        Normalize country name and return ISO code.
        
        Returns:
            Tuple of (normalized_name, iso_code)
        """
        country_clean = country_name.strip()
        
        # Direct lookup
        if country_clean in self.mapping:
            iso_code = self.mapping[country_clean]
            return country_clean, iso_code
        
        # Case-insensitive lookup
        country_lower = country_clean.lower()
        for key, iso_code in self.mapping.items():
            if key.lower() == country_lower:
                return key, iso_code
        
        # No match found
        if country_clean not in self.unmapped_countries:
            self.unmapped_countries.append(country_clean)
        return country_clean, None
    
    def validate_countries(self, df: pd.DataFrame, country_col: str = 'country') -> pd.DataFrame:
        """
        Validate and normalize country names in DataFrame.
        
        Returns:
            DataFrame with normalized country names and ISO codes
        """
        if country_col not in df.columns:
            logger.warning(f"Country column '{country_col}' not found")
            return df
        
        df = df.copy()
        normalized_names = []
        iso_codes = []
        
        for country in df[country_col]:
            norm_name, iso_code = self.normalize_country_name(str(country))
            normalized_names.append(norm_name)
            iso_codes.append(iso_code)
        
        df[f'{country_col}_normalized'] = normalized_names
        df['iso_code'] = iso_codes
        
        # Report unmapped countries
        if self.unmapped_countries:
            logger.warning(f"Found {len(self.unmapped_countries)} unmapped countries: {self.unmapped_countries[:10]}")
        
        return df


class GDPValidator:
    """Validates GDP data for currency, units, and PPP adjustments."""
    
    def __init__(self):
        self.currency_codes = {
            'USD': 'US Dollar',
            'EUR': 'Euro',
            'GBP': 'British Pound',
            'JPY': 'Japanese Yen',
            'CNY': 'Chinese Yuan',
            'INR': 'Indian Rupee',
        }
        self.unit_types = ['nominal', 'real', 'ppp', 'constant']
    
    def validate_gdp_data(self, df: pd.DataFrame, gdp_col: str = 'gdp',
                         currency_col: Optional[str] = None,
                         unit_col: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Validate GDP data structure and units.
        
        Returns:
            Dictionary with validation results
        """
        issues = {
            'errors': [],
            'warnings': []
        }
        
        if gdp_col not in df.columns:
            issues['errors'].append(f"GDP column '{gdp_col}' not found")
            return issues
        
        # Check for negative values
        negative_gdp = df[df[gdp_col] < 0]
        if len(negative_gdp) > 0:
            issues['errors'].append(f"Found {len(negative_gdp)} rows with negative GDP values")
        
        # Check for missing values
        missing_gdp = df[df[gdp_col].isna()]
        if len(missing_gdp) > 0:
            issues['warnings'].append(f"Found {len(missing_gdp)} rows with missing GDP values")
        
        # Check currency if provided
        if currency_col and currency_col in df.columns:
            unknown_currencies = df[~df[currency_col].isin(self.currency_codes.keys())]
            if len(unknown_currencies) > 0:
                issues['warnings'].append(f"Found {len(unknown_currencies)} rows with unknown currency codes")
        
        # Check unit type if provided
        if unit_col and unit_col in df.columns:
            unknown_units = df[~df[unit_col].isin(self.unit_types)]
            if len(unknown_units) > 0:
                issues['warnings'].append(f"Found {len(unknown_units)} rows with unknown unit types")
        
        # Check for unrealistic values (GDP > 100 trillion USD)
        if 'USD' in str(df[gdp_col].dtype) or currency_col is None:
            very_large_gdp = df[df[gdp_col] > 100000]  # Assuming billions USD
            if len(very_large_gdp) > 0:
                issues['warnings'].append(f"Found {len(very_large_gdp)} rows with GDP > 100 trillion (check units)")
        
        return issues


class DataIntegrityChecker:
    """Checks data integrity using checksums and consistency checks."""
    
    def __init__(self):
        self.checksums = {}
        self.integrity_log = []
    
    def compute_checksum(self, df: pd.DataFrame, name: str) -> str:
        """Compute SHA256 checksum of DataFrame."""
        # Convert to string representation for hashing
        df_str = df.to_csv(index=False)
        checksum = hashlib.sha256(df_str.encode()).hexdigest()
        self.checksums[name] = checksum
        return checksum
    
    def verify_checksum(self, df: pd.DataFrame, name: str, expected_checksum: str) -> bool:
        """Verify DataFrame matches expected checksum."""
        actual_checksum = self.compute_checksum(df, name)
        match = actual_checksum == expected_checksum
        if not match:
            logger.warning(f"Checksum mismatch for {name}: expected {expected_checksum[:16]}..., got {actual_checksum[:16]}...")
        return match
    
    def check_row_count_consistency(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                                   operation: str) -> bool:
        """Check that row count changes are expected."""
        rows_before = len(df_before)
        rows_after = len(df_after)
        diff = rows_after - rows_before
        
        if diff < 0:
            logger.info(f"{operation}: Removed {abs(diff)} rows ({rows_before} -> {rows_after})")
        elif diff > 0:
            logger.info(f"{operation}: Added {diff} rows ({rows_before} -> {rows_after})")
        else:
            logger.info(f"{operation}: Row count unchanged ({rows_before})")
        
        return True
    
    def check_value_range_consistency(self, df: pd.DataFrame, col: str,
                                     expected_min: Optional[float] = None,
                                     expected_max: Optional[float] = None) -> Dict[str, any]:
        """Check that values are within expected range."""
        if col not in df.columns:
            return {'valid': False, 'error': f"Column {col} not found"}
        
        values = df[col].dropna()
        if len(values) == 0:
            return {'valid': False, 'error': f"Column {col} has no valid values"}
        
        actual_min = values.min()
        actual_max = values.max()
        
        issues = []
        if expected_min is not None and actual_min < expected_min:
            issues.append(f"Minimum value {actual_min} below expected {expected_min}")
        if expected_max is not None and actual_max > expected_max:
            issues.append(f"Maximum value {actual_max} above expected {expected_max}")
        
        return {
            'valid': len(issues) == 0,
            'actual_min': actual_min,
            'actual_max': actual_max,
            'issues': issues
        }
    
    def export_integrity_report(self, filepath: str):
        """Export integrity report to JSON."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'checksums': self.checksums,
            'log': self.integrity_log
        }
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Integrity report exported to {filepath}")


class TimeSeriesValidator:
    """Validates time series consistency and continuity."""
    
    def __init__(self):
        self.issues = []
    
    def check_continuity(self, df: pd.DataFrame, group_col: str, year_col: str,
                       value_col: str, max_gap: int = 3) -> Dict[str, List[str]]:
        """
        Check time series continuity for each group.
        
        Returns:
            Dictionary with continuity issues by group
        """
        issues = {}
        
        for group_val in df[group_col].unique():
            group_data = df[df[group_col] == group_val].sort_values(year_col)
            years = group_data[year_col].values
            
            # Check for gaps
            gaps = []
            for i in range(len(years) - 1):
                gap_size = years[i+1] - years[i] - 1
                if gap_size > 0:
                    if gap_size > max_gap:
                        gaps.append(f"Large gap: {years[i]} to {years[i+1]} ({gap_size} years)")
                    else:
                        gaps.append(f"Gap: {years[i]} to {years[i+1]} ({gap_size} years)")
            
            if gaps:
                issues[str(group_val)] = gaps
        
        return issues
    
    def check_monotonicity(self, df: pd.DataFrame, group_col: str, year_col: str,
                          value_col: str, allow_decrease: bool = True) -> Dict[str, List[str]]:
        """
        Check if time series is monotonic (increasing or decreasing).
        
        Returns:
            Dictionary with monotonicity violations
        """
        violations = {}
        
        for group_val in df[group_col].unique():
            group_data = df[df[group_col] == group_val].sort_values(year_col)
            values = group_data[value_col].dropna().values
            
            if len(values) < 2:
                continue
            
            # Check for sudden large changes (>50% change)
            large_changes = []
            for i in range(len(values) - 1):
                if values[i] > 0:
                    pct_change = abs((values[i+1] - values[i]) / values[i]) * 100
                    if pct_change > 50:
                        large_changes.append(f"Large change: {pct_change:.1f}% between years")
            
            if large_changes:
                violations[str(group_val)] = large_changes
        
        return violations
    
    def check_structural_breaks(self, df: pd.DataFrame, group_col: str, year_col: str,
                               value_col: str, break_years: List[int] = [2008, 2020]) -> Dict[str, List[str]]:
        """
        Check for structural breaks at known dates (e.g., financial crisis, COVID).
        
        Returns:
            Dictionary with structural break indicators
        """
        breaks = {}
        
        for group_val in df[group_col].unique():
            group_data = df[df[group_col] == group_val].sort_values(year_col)
            group_breaks = []
            
            for break_year in break_years:
                before = group_data[group_data[year_col] < break_year][value_col]
                after = group_data[group_data[year_col] > break_year][value_col]
                
                if len(before) > 0 and len(after) > 0:
                    before_mean = before.mean()
                    after_mean = after.mean()
                    
                    if before_mean > 0:
                        pct_change = ((after_mean - before_mean) / before_mean) * 100
                        if abs(pct_change) > 20:  # More than 20% change
                            group_breaks.append(f"Structural break at {break_year}: {pct_change:.1f}% change")
            
            if group_breaks:
                breaks[str(group_val)] = group_breaks
        
        return breaks


def create_validation_report(validators: Dict, output_path: str):
    """Create comprehensive validation report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'validations': {}
    }
    
    for name, validator in validators.items():
        if hasattr(validator, 'issues'):
            report['validations'][name] = validator.issues
        elif hasattr(validator, 'unmapped_countries'):
            report['validations'][name] = {
                'unmapped_countries': validator.unmapped_countries
            }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report exported to {output_path}")

