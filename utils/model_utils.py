"""
Model loading and prediction utilities for Random Forest FLFP model
"""

import time
import numpy as np
import pandas as pd

from utils.data_loader import load_flfp_data
from utils.map_data_utils import load_test_set_data
from utils.modeling import train_model, compute_feature_stats

# Cache trained artifacts (train once at startup, use many times)
_cached_artifacts = None
_cached_stats = None


def load_model_artifacts():
    """
    Train the Random Forest once and cache it.
    Model is trained at app startup and reused for all predictions.

    Returns:
        dict: Model artifacts including trained model, feature names, and importance
    """
    global _cached_artifacts

    if _cached_artifacts is None:
        start = time.time()
        _cached_artifacts = train_model(load_flfp_data())
        first, last = _cached_artifacts['train_years']
        print(f"✓ Trained RF model on {first}-{last} with "
              f"{len(_cached_artifacts['feature_names'])} features in {time.time() - start:.1f}s")

    return _cached_artifacts


def load_feature_stats():
    """
    Compute feature statistics for slider ranges and default values.
    Used only for UI setup, not for prediction preprocessing.

    Returns:
        dict: Feature statistics including min, max, mean, and default values
    """
    global _cached_stats

    if _cached_stats is None:
        _cached_stats = compute_feature_stats(load_flfp_data(), load_test_set_data())

    return _cached_stats


def get_feature_importance_order():
    """
    Get features ordered by importance (descending).
    Used to order sliders in the UI.
    
    Returns:
        list: Feature names sorted by importance
    """
    artifacts = load_model_artifacts()
    feature_importance = artifacts['feature_importance']
    sorted_features = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
    return [f['feature'] for f in sorted_features]


def make_prediction(feature_values):
    """
    Make a prediction using the trained Random Forest model.
    This is very fast - just a forward pass through decision trees.
    No preprocessing needed as user provides all values via sliders.
    
    Args:
        feature_values (dict): Mapping of feature names to values
        
    Returns:
        float: Predicted FLFP rate (0-100)
    """
    artifacts = load_model_artifacts()
    model = artifacts['model']
    feature_names = artifacts['feature_names']
    
    # Create pandas DataFrame with proper column names to avoid sklearn warning
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # Make prediction - this is just evaluating decision trees (very fast!)
    prediction = model.predict(input_df)[0]
    
    # Clip to valid range [0, 100]
    return np.clip(prediction, 0, 100)


def get_population_slider_marks(log_min, log_max, n_marks=5):
    """
    Create logarithmically-spaced marks for population slider.
    Returns marks with actual population values as labels.
    
    Args:
        log_min (float): Minimum log(population) value
        log_max (float): Maximum log(population) value
        n_marks (int): Number of marks to create
        
    Returns:
        list: List of dicts with 'value' (log scale) and 'label' (actual population)
    """
    log_positions = np.linspace(log_min, log_max, n_marks)
    marks = []
    
    for log_pos in log_positions:
        actual_pop = np.exp(log_pos)
        
        # Format population nicely
        if actual_pop < 1e6:
            label = f"{actual_pop/1e3:.0f}K"
        elif actual_pop < 1e9:
            label = f"{actual_pop/1e6:.1f}M"
        else:
            label = f"{actual_pop/1e9:.2f}B"
        
        marks.append({
            "value": log_pos,
            "label": label
        })
    
    return marks


def format_population_value(log_pop):
    """
    Format a log(population) value into human-readable population string.
    
    Args:
        log_pop (float): Log-transformed population value
        
    Returns:
        str: Formatted population (e.g., "15.2M", "1.4B")
    """
    actual_pop = np.exp(log_pop)
    
    if actual_pop < 1e6:
        return f"{actual_pop/1e3:.1f}K"
    elif actual_pop < 1e9:
        return f"{actual_pop/1e6:.1f}M"
    else:
        return f"{actual_pop/1e9:.2f}B"


def get_feature_label(feature_name):
    """
    Convert feature name to human-readable label for display.
    
    Args:
        feature_name (str): Technical feature name
        
    Returns:
        str: Human-readable label
    """
    label_mapping = {
        # Numeric features
        'fertility_rate': 'Total Fertility Rate (births per woman)',
        'fertility_adolescent': 'Adolescent Fertility Rate (per 1,000 women 15-19)',
        'urban_population': 'Urban Population (%)',
        'dependency_ratio': 'Age Dependency Ratio (%)',
        'life_exp_female': 'Female Life Expectancy (years)',
        'infant_mortality': 'Infant Mortality Rate (per 1,000 births)',
        'population_total': 'Total Population (log scale)',
        'secondary_enroll_fe': 'Female Secondary Enrollment (gross %, can exceed 100)',
        'gdp_per_capita_const': 'GDP per Capita (constant 2015 USD)',
        'gdp_growth': 'GDP Growth (annual %)',
        'services_gdp': 'Services (% of GDP)',
        'industry_gdp': 'Industry (% of GDP)',
        'rule_of_law': 'Rule of Law Index (-2.5 to 2.5)',
        
        # Income level (encoded)
        'income_level_encoded': 'Income Level',
        
        # Region dummies
        'region_eap': 'East Asia & Pacific',
        'region_eca': 'Europe & Central Asia',
        'region_lac': 'Latin America & Caribbean',
        'region_mena_afpak': 'MENA + AfPak',
        'region_namerica': 'North America',
        'region_sasia': 'South Asia',
        'region_ssa': 'Sub-Saharan Africa',
    }
    
    return label_mapping.get(feature_name, feature_name.replace('_', ' ').title())


def get_region_features():
    """
    Get list of region feature names.
    
    Returns:
        list: Region feature names
    """
    return [
        'region_eap',
        'region_eca', 
        'region_lac',
        'region_namerica',
        'region_sasia',
        'region_ssa'
    ]


def get_region_options():
    """
    Get region options for radio button selection.
    Includes all regions, including the reference category (MENA + AfPak).
    
    Returns:
        list: List of dicts with 'value' and 'label' for each region
    """
    # Include the reference category first, then all dummy variables
    all_regions = ['region_mena_afpak'] + get_region_features()
    return [
        {'value': region, 'label': get_feature_label(region)}
        for region in all_regions
    ]

