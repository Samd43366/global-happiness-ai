"""
Data Ingestion and Loading Module for World Happiness Analysis.

Provides validated loading routines for raw multi-year CSV files and cleaned datasets.
"""

import os
import glob
from typing import List, Optional
import pandas as pd


EXPECTED_COLUMNS: List[str] = [
    'country',
    'region',
    'happiness_score',
    'gdp_per_capita',
    'social_support',
    'healthy_life_expectancy',
    'freedom_to_make_life_choices',
    'generosity',
    'perceptions_of_corruption'
]


def load_raw_data(data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Load and concatenate all annual World Happiness Report CSV files.
    
    Args:
        data_dir: Directory containing raw WHR_YYYY.csv files.
        
    Returns:
        pd.DataFrame: Concatenated panel dataset with a 'year' column.
    
    Raises:
        FileNotFoundError: If no matching CSV files are located.
        ValueError: If any file fails schema validation.
    """
    pattern = os.path.join(data_dir, 'WHR_*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No WHR_*.csv files found in directory: {data_dir}")
    
    dfs = []
    for file_path in files:
        # Extract 4-digit year from filename (e.g. WHR_2023.csv -> 2023)
        base_name = os.path.basename(file_path)
        year_str = base_name.replace('WHR_', '').replace('.csv', '')
        try:
            year = int(year_str)
        except ValueError:
            raise ValueError(f"Unable to parse year from filename: {base_name}")
            
        df_year = pd.read_csv(file_path)
        
        # Verify required columns exist
        missing_cols = set(EXPECTED_COLUMNS) - set(df_year.columns)
        if missing_cols:
            raise ValueError(f"File {base_name} is missing expected columns: {missing_cols}")
            
        df_year = df_year[EXPECTED_COLUMNS].copy()
        df_year['year'] = year
        dfs.append(df_year)
        
    unified_df = pd.concat(dfs, ignore_index=True)
    return unified_df


def load_cleaned_data(file_path: str = 'data/cleaned_data.csv') -> pd.DataFrame:
    """
    Load the finalized cleaned dataset for modeling and analysis.
    
    Args:
        file_path: Path to cleaned_data.csv.
        
    Returns:
        pd.DataFrame: Validated cleaned dataset.
    """
    if not os.path.exists(file_path):
        # Fallback to root directory if called from root or subfolder
        alt_path = os.path.join(os.path.dirname(__file__), '..', file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Cleaned dataset not found at {file_path}")
            
    df = pd.read_csv(file_path)
    return df
