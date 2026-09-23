"""
Feature Engineering and Preprocessing Pipeline Module.

Defines input feature subsets and builds reproducible scikit-learn
ColumnTransformer pipelines ensuring zero data leakage.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


TARGET_COLUMN: str = 'happiness_score'

NUMERIC_FEATURES: List[str] = [
    'gdp_per_capita',
    'social_support',
    'healthy_life_expectancy',
    'freedom_to_make_life_choices',
    'generosity',
    'perceptions_of_corruption'
]

CATEGORICAL_FEATURES: List[str] = [
    'region'
]

# Columns explicitly quarantined to prevent target leakage or memorization
EXCLUDED_COLUMNS: List[str] = [
    'country',
    'year',
    'happiness_score'
]


def get_feature_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictive features from the target variable.
    
    Args:
        df: Cleaned World Happiness dataset.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X_features, y_target)
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for preprocessing.
    
    Features:
        - Numeric: Median Imputation + StandardScaler
        - Categorical: OneHotEncoder with handle_unknown='ignore'
        
    Returns:
        ColumnTransformer: Preprocessing transformer pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor
