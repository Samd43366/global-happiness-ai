"""
Model Interpretability and Explainability Module.

Leverages SHAP (SHapley Additive exPlanations) to explain global feature
importance and local individual predictions for policy analysis.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def get_tree_explainer(pipeline) -> Tuple[shap.TreeExplainer, List[str]]:
    """
    Extract tree model and feature names from the fitted scikit-learn Pipeline.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['regressor']
    
    # Extract feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(['region']).tolist()
    all_feature_names = [
        'gdp_per_capita',
        'social_support',
        'healthy_life_expectancy',
        'freedom_to_make_life_choices',
        'generosity',
        'perceptions_of_corruption'
    ] + cat_features
    
    explainer = shap.TreeExplainer(model)
    return explainer, all_feature_names


def explain_prediction(pipeline, input_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate local SHAP values for a single prediction instance.
    
    Args:
        pipeline: Fitted end-to-end Pipeline.
        input_df: Single row DataFrame of raw inputs.
        
    Returns:
        Dict[str, Any]: Base value, prediction, and feature contributions.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['regressor']
    
    # Transform input through preprocessor
    X_trans = preprocessor.transform(input_df)
    
    explainer, feature_names = get_tree_explainer(pipeline)
    shap_values = explainer.shap_values(X_trans)
    base_value = float(explainer.expected_value)
    predicted_value = float(pipeline.predict(input_df)[0])
    
    contributions = []
    for name, val in zip(feature_names, shap_values[0]):
        contributions.append({
            'feature': name,
            'shap_value': round(float(val), 4)
        })
        
    # Sort contributions by absolute impact
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    return {
        'base_value': round(base_value, 4),
        'predicted_value': round(predicted_value, 4),
        'contributions': contributions
    }
