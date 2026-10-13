"""
Helper functions for loading and processing test set data for map visualizations
"""

import pandas as pd
from functools import lru_cache

from utils.data_loader import load_flfp_data
from utils.modeling import build_display_set

@lru_cache(maxsize=1)
def load_test_set_data():
    """
    Build the display set: each country's FLFP in the latest year, with the
    most recent predictor values. Cached to avoid rebuilding on every call.

    Returns:
        pd.DataFrame: Test set with columns including country_name, iso3c, year,
                      flfp_15_64, and all predictor features
    """
    return build_display_set(load_flfp_data())


def get_latest_flfp_by_country():
    """
    Get dictionary mapping country names to their most recent FLFP values.
    
    Returns:
        dict: {country_name: flfp_value}
    """
    test_df = load_test_set_data()
    return dict(zip(test_df['country_name'], test_df['flfp_15_64']))


def get_iso_country_mapping():
    """
    Get dictionary mapping country names to ISO3C codes for choropleth.
    
    Returns:
        dict: {country_name: iso3c}
    """
    test_df = load_test_set_data()
    return dict(zip(test_df['country_name'], test_df['iso3c']))


def get_country_features(country_name):
    """
    Get all feature values for a specific country to populate sliders.
    
    Args:
        country_name (str): Name of the country
        
    Returns:
        dict: Dictionary with feature names as keys and values, or None if country not found.
              Includes keys for: all numeric features, income_level_encoded, and region dummies
    """
    test_df = load_test_set_data()
    
    # Find the country
    country_data = test_df[test_df['country_name'] == country_name]
    
    if len(country_data) == 0:
        return None
    
    # Get the first (and only) row as a dict
    country_row = country_data.iloc[0].to_dict()
    
    # Define all predictor features
    predictor_features = [
        'fertility_rate', 'fertility_adolescent', 'urban_population',
        'dependency_ratio', 'life_exp_female', 'infant_mortality',
        'population_total', 'secondary_enroll_fe', 'gdp_per_capita_const',
        'gdp_growth', 'services_gdp', 'industry_gdp', 'rule_of_law',
        'income_level_encoded'
    ]
    
    # Get region dummies
    region_cols = [col for col in test_df.columns if col.startswith('region_')]
    
    # Build feature dictionary
    features = {}
    
    # Add numeric features
    for feature in predictor_features:
        features[feature] = country_row[feature]
    
    # Add region dummies
    for region_col in region_cols:
        features[region_col] = country_row[region_col]
    
    # Add metadata for reference
    features['country_name'] = country_row['country_name']
    features['year'] = country_row['year']
    features['flfp_15_64'] = country_row['flfp_15_64']
    
    return features


def get_flfp_distribution():
    """
    Get array of all FLFP values for histogram visualization.
    
    Returns:
        tuple: (flfp_values, country_names) - arrays of FLFP values and corresponding country names
    """
    test_df = load_test_set_data()
    return test_df['flfp_15_64'].values, test_df['country_name'].values


def get_test_set_summary():
    """
    Get summary statistics about the test set.
    
    Returns:
        dict: Summary statistics including counts, ranges, etc.
    """
    test_df = load_test_set_data()
    
    return {
        'total_countries': len(test_df),
        'year_range': (int(test_df['year'].min()), int(test_df['year'].max())),
        'flfp_range': (float(test_df['flfp_15_64'].min()), float(test_df['flfp_15_64'].max())),
        'flfp_mean': float(test_df['flfp_15_64'].mean()),
        'flfp_median': float(test_df['flfp_15_64'].median()),
        'regions': test_df[[col for col in test_df.columns if col.startswith('region_')]].sum().to_dict()
    }
