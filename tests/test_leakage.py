"""
Unit tests for data leakage prevention and feature quarantine.
"""

import pytest
import pandas as pd
from src.features import get_feature_splits, NUMERIC_FEATURES, CATEGORICAL_FEATURES, EXCLUDED_COLUMNS
from src.data_loader import load_cleaned_data


def test_target_quarantine():
    """Verify target variable is strictly separated from feature set."""
    df = load_cleaned_data('cleaned_data.csv')
    X, y = get_feature_splits(df)
    
    assert 'happiness_score' not in X.columns, "Target variable must not be present in feature matrix X."
    assert 'happiness_score' == y.name, "Target series name must be happiness_score."


def test_no_rank_or_decomposed_columns():
    """Verify that rank and algebraic decomposed factors are absent from features."""
    df = load_cleaned_data('cleaned_data.csv')
    X, _ = get_feature_splits(df)
    
    for col in X.columns:
        assert 'rank' not in col.lower(), f"Rank column {col} found in features (Target Leakage risk)."
        assert 'explained' not in col.lower(), f"Explained factor {col} found in features (Target Leakage risk)."
        assert 'dystopia' not in col.lower(), f"Dystopia residual {col} found in features."


def test_country_identifier_excluded():
    """Verify high-cardinality country names are excluded from features to prevent memorization."""
    df = load_cleaned_data('cleaned_data.csv')
    X, _ = get_feature_splits(df)
    
    assert 'country' not in X.columns, "Country identifier should not be a raw feature in model training."
