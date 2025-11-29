"""
Sample Data Generator for BP Statistical Review Energy Dataset

This script generates a sample CSV file that mimics the structure
of BP Statistical Review energy consumption data for testing purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(output_file='raw_energy_data.csv'):
    """
    Generate sample energy consumption data.
    
    Args:
        output_file: Output CSV file path
    """
    np.random.seed(42)
    
    # Countries with different characteristics
    countries = [
        'United States', 'China', 'India', 'Japan', 'Germany',
        'Russia', 'Brazil', 'South Korea', 'Canada', 'France',
        'United Kingdom', 'Italy', 'Australia', 'Mexico', 'Saudi Arabia',
        'Indonesia', 'South Africa', 'Turkey', 'Argentina', 'Thailand',
        'Nigeria', 'Bangladesh', 'Vietnam', 'Poland', 'Spain'
    ]
    
    years = range(2000, 2025)
    
    # Base energy consumption in EJ (will be converted by the script)
    # Realistic ranges based on actual data
    base_consumption = {
        'United States': 100,  # ~100 EJ
        'China': 80,
        'India': 20,
        'Japan': 25,
        'Germany': 15,
        'Russia': 30,
        'Brazil': 10,
        'South Korea': 12,
        'Canada': 15,
        'France': 12,
        'United Kingdom': 10,
        'Italy': 8,
        'Australia': 6,
        'Mexico': 8,
        'Saudi Arabia': 7,
        'Indonesia': 5,
        'South Africa': 6,
        'Turkey': 5,
        'Argentina': 4,
        'Thailand': 3,
        'Nigeria': 2,
        'Bangladesh': 1.5,
        'Vietnam': 2.5,
        'Poland': 4,
        'Spain': 7
    }
    
    data = []
    
    for country in countries:
        base = base_consumption.get(country, 5)
        
        for year in years:
            # Add some growth/decline trends
            years_since_2000 = year - 2000
            
            # Different growth rates for different countries
            if country in ['China', 'India', 'Indonesia', 'Vietnam', 'Bangladesh']:
                # Rapid growth for developing countries
                growth_rate = 0.05  # 5% annual growth
            elif country in ['United States', 'Germany', 'Japan', 'France']:
                # Slow growth/decline for developed countries
                growth_rate = -0.005  # -0.5% annual decline
            else:
                # Moderate growth
                growth_rate = 0.02  # 2% annual growth
            
            # Calculate consumption with some noise
            consumption = base * (1 + growth_rate) ** years_since_2000
            consumption += np.random.normal(0, consumption * 0.05)  # 5% noise
            
            # Introduce some missing data patterns
            # Some countries have missing early years (developing countries)
            if country in ['Nigeria', 'Bangladesh', 'Vietnam'] and year < 2005:
                if np.random.random() < 0.3:  # 30% chance of missing
                    consumption = np.nan
            
            # Some countries have missing recent years (data collection issues)
            if country in ['Argentina', 'Thailand'] and year > 2020:
                if np.random.random() < 0.2:  # 20% chance of missing
                    consumption = np.nan
            
            # Introduce some outliers (will be flagged)
            if country == 'Russia' and year == 2010:
                consumption = consumption * 10  # Outlier
            
            data.append({
                'Country': country,
                'Year': year,
                'Primary Energy Consumption (EJ)': consumption if not np.isnan(consumption) else np.nan
            })
    
    df = pd.DataFrame(data)
    
    # Add some rows with different unit formats (to test unit conversion)
    # Add a few rows with Mtoe
    sample_mtoe = df.sample(n=10, random_state=42)
    sample_mtoe['Primary Energy Consumption (Mtoe)'] = sample_mtoe['Primary Energy Consumption (EJ)'] * 23.88  # Approx conversion
    sample_mtoe['Primary Energy Consumption (EJ)'] = np.nan
    df = pd.concat([df, sample_mtoe], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated sample data with {len(df)} rows saved to {output_file}")
    print(f"Countries: {len(countries)}")
    print(f"Years: {min(years)} - {max(years)}")
    print(f"Missing values: {df['Primary Energy Consumption (EJ)'].isna().sum()}")
    
    return df


if __name__ == '__main__':
    generate_sample_data()

