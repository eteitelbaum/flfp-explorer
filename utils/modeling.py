"""
Model training and preprocessing for the Model Explorer.

Everything is built from data/flfp_dataset.parquet when the app starts:
  - the Random Forest is trained on all years before the display year
  - the display set holds each country's FLFP in the display year (the latest
    year in the data), with predictors taken from the most recent non-missing
    value within a short lookback window
  - slider ranges and defaults are derived from the same data

Hyperparameters were selected by grid search in models_time_split.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TARGET = 'flfp_15_64'

BASE_PREDICTORS = [
    'fertility_rate', 'fertility_adolescent', 'urban_population',
    'dependency_ratio', 'life_exp_female', 'infant_mortality',
    'population_total', 'secondary_enroll_fe', 'gdp_per_capita_const',
    'gdp_growth', 'services_gdp', 'industry_gdp', 'rule_of_law'
]

INCOME_MAPPING = {
    'Low income': 0,
    'Lower middle income': 1,
    'Upper middle income': 2,
    'High income': 3
}

# Middle East, North Africa, Afghanistan & Pakistan is the reference category
REGION_MAPPING = {
    'East Asia & Pacific': 'region_eap',
    'Europe & Central Asia': 'region_eca',
    'Latin America & Caribbean': 'region_lac',
    'North America': 'region_namerica',
    'South Asia': 'region_sasia',
    'Sub-Saharan Africa': 'region_ssa'
}
REGION_COLS = list(REGION_MAPPING.values())

FEATURE_NAMES = BASE_PREDICTORS + ['income_level_encoded'] + REGION_COLS

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.3,
    'random_state': 42,
    'n_jobs': -1,
}

# Years searched back from the display year for each predictor's latest value
LOOKBACK_YEARS = 2

DEFAULT_COUNTRY = 'Madagascar'


def prepare_features(df):
    """Encode income and region and log-transform population."""
    df = df[df['income_level'].isin(INCOME_MAPPING)].copy()
    df['region'] = df['region'].str.strip()
    df['income_level_encoded'] = df['income_level'].map(INCOME_MAPPING)
    for region, col in REGION_MAPPING.items():
        df[col] = (df['region'] == region).astype(int)
    df['population_total'] = np.log(df['population_total'])
    return df


def get_display_year(df):
    """The display year is the latest year with FLFP data."""
    return int(df.loc[df[TARGET].notna(), 'year'].max())


def split_training(df):
    """Rows with FLFP from every year before the display year."""
    display_year = get_display_year(df)
    train = df[df[TARGET].notna() & (df['year'] < display_year)]
    return prepare_features(train)


def compute_imputation_medians(train):
    """Country-level and global medians from the training data only."""
    return {
        'country': train.groupby('country_name')[BASE_PREDICTORS].median(),
        'global': train[BASE_PREDICTORS].median(),
    }


def impute(df, medians):
    """Fill gaps with the country's training median, then the global training median."""
    df = df.copy()
    country_medians = medians['country'].reindex(df['country_name'])
    country_medians.index = df.index
    df[BASE_PREDICTORS] = df[BASE_PREDICTORS].fillna(country_medians)
    df[BASE_PREDICTORS] = df[BASE_PREDICTORS].fillna(medians['global'])
    return df


def build_training_set(df):
    """Imputed training features and target, ordered by year then country."""
    train = split_training(df)
    train = impute(train, compute_imputation_medians(train))
    train = train.sort_values(['year', 'country_name'])
    return train[FEATURE_NAMES], train[TARGET]


def train_model(df):
    """
    Fit the Random Forest on all years before the display year.

    Returns:
        dict: model, feature_names, feature_importance, train_years
    """
    X_train, y_train = build_training_set(df)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    feature_importance = (
        pd.DataFrame({'feature': FEATURE_NAMES, 'importance': model.feature_importances_})
        .sort_values('importance', ascending=False)
        .to_dict('records')
    )
    display_year = get_display_year(df)
    return {
        'model': model,
        'feature_names': FEATURE_NAMES,
        'feature_importance': feature_importance,
        'train_years': (int(df['year'].min()), display_year - 1),
    }


def build_display_set(df):
    """
    One row per country with FLFP in the display year. Each predictor takes its
    most recent non-missing value within the lookback window; remaining gaps are
    filled with training medians.
    """
    display_year = get_display_year(df)
    medians = compute_imputation_medians(split_training(df))

    window = prepare_features(df[df['year'].between(display_year - LOOKBACK_YEARS, display_year)])
    current = window[(window['year'] == display_year) & window[TARGET].notna()]

    latest = (
        window[window['country_name'].isin(current['country_name'])]
        .sort_values('year')
        .groupby('country_name')[BASE_PREDICTORS]
        .last()
    )

    display = current[
        ['country_name', 'iso3c', 'year', TARGET, 'income_level_encoded'] + REGION_COLS
    ].set_index('country_name')
    display = display.join(latest).reset_index()
    display = impute(display, medians)

    columns = ['country_name', 'iso3c', 'year', TARGET] + FEATURE_NAMES
    return display[columns].sort_values('country_name').reset_index(drop=True)


def compute_feature_stats(df, display):
    """
    Slider ranges and defaults. Means come from the training data; ranges also
    cover the display set so every country's values fit on the sliders.
    """
    X_train, _ = build_training_set(df)
    stats = {}
    for feature in FEATURE_NAMES:
        stats[feature] = {
            'min': float(min(X_train[feature].min(), display[feature].min())),
            'max': float(max(X_train[feature].max(), display[feature].max())),
            'mean': float(X_train[feature].mean()),
        }

    default_row = display[display['country_name'] == DEFAULT_COUNTRY]
    if len(default_row) == 0:
        default_row = display.iloc[[0]]
    stats['default_values'] = {
        feature: float(default_row[feature].iloc[0]) for feature in FEATURE_NAMES
    }
    return stats
