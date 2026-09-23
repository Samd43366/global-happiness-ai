"""
UI and Application Utility Functions.

Provides cached resource loaders for the serialized model artifact and dataset.
"""

import os
import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_trained_model(model_path: str = 'models/final_model.pkl'):
    """Load serialized pipeline artifact with caching."""
    if not os.path.exists(model_path):
        alt = os.path.join(os.path.dirname(__file__), '..', model_path)
        if os.path.exists(alt):
            model_path = alt
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


@st.cache_data
def load_data(data_path: str = 'cleaned_data.csv'):
    """Load cleaned dataset with caching."""
    if not os.path.exists(data_path):
        alt = os.path.join(os.path.dirname(__file__), '..', data_path)
        if os.path.exists(alt):
            data_path = alt
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}")
    return pd.read_csv(data_path)
