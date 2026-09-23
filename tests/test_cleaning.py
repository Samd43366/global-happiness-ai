"""
Unit tests for data loading, data cleaning, and schema integrity.
"""

import os
import pandas as pd
import pytest
from src.data_loader import load_raw_data, load_cleaned_data, EXPECTED_COLUMNS
from src.cleaning import clean_dataset


def test_raw_data_loading():
    """Verify raw data loading successfully aggregates all annual files."""
    df = load_raw_data('data/raw')
    assert len(df) > 1000, "Raw dataset should contain over 1,000 panel records."
    assert 'year' in df.columns, "Year column must be present."
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Expected column {col} missing from raw data."


def test_clean_dataset_zero_nulls():
    """Verify clean_dataset completely resolves all missing values."""
    raw = load_raw_data('data/raw')
    cleaned, report = clean_dataset(raw)
    
    assert cleaned.isnull().sum().sum() == 0, "Cleaned dataset must contain zero NaN values."
    assert len(cleaned) == len(raw), "Cleaned dataset should preserve all observations."
    assert len(report) >= 9, "Data Quality Report must document all major features."


def test_target_variable_bounds():
    """Verify the target happiness_score is within realistic Cantril Ladder bounds (0 to 10)."""
    df = load_cleaned_data('cleaned_data.csv')
    assert df['happiness_score'].min() >= 0.0, "Happiness score cannot be negative."
    assert df['happiness_score'].max() <= 10.0, "Happiness score cannot exceed 10.0."


def test_region_normalization():
    """Verify regional categories are properly normalized."""
    df = load_cleaned_data('cleaned_data.csv')
    regions = df['region'].unique().tolist()
    assert 'Africa' not in regions, "Raw 'Africa' label must be consolidated into 'Sub-Saharan Africa'."
    assert 'Sub-Saharan Africa' in regions
