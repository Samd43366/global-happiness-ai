"""
Unit tests for serialized model artifact inference and explainability.
"""

import os
import joblib
import pandas as pd
import pytest
from src.explainability import explain_prediction


def test_model_artifact_exists():
    """Verify serialized model artifact exists and is loadable."""
    model_path = os.path.join('models', 'final_model.pkl')
    assert os.path.exists(model_path), f"Serialized model artifact missing at {model_path}"
    
    pipeline = joblib.load(model_path)
    assert hasattr(pipeline, 'predict'), "Loaded artifact must implement predict method."


def test_end_to_end_inference():
    """Verify pipeline predicts reasonable values from raw input dictionary."""
    model_path = os.path.join('models', 'final_model.pkl')
    pipeline = joblib.load(model_path)
    
    sample_input = pd.DataFrame([{
        'gdp_per_capita': 1.45,
        'social_support': 1.25,
        'healthy_life_expectancy': 0.75,
        'freedom_to_make_life_choices': 0.60,
        'generosity': 0.15,
        'perceptions_of_corruption': 0.08,
        'region': 'Western Europe'
    }])
    
    pred = pipeline.predict(sample_input)
    assert len(pred) == 1, "Expected single prediction output."
    assert 2.0 <= pred[0] <= 9.0, f"Predicted score {pred[0]} out of expected Cantril ladder range [2.0, 9.0]."


def test_shap_explanation_structure():
    """Verify SHAP explanation generates valid contributions."""
    model_path = os.path.join('models', 'final_model.pkl')
    pipeline = joblib.load(model_path)
    
    sample_input = pd.DataFrame([{
        'gdp_per_capita': 1.45,
        'social_support': 1.25,
        'healthy_life_expectancy': 0.75,
        'freedom_to_make_life_choices': 0.60,
        'generosity': 0.15,
        'perceptions_of_corruption': 0.08,
        'region': 'Western Europe'
    }])
    
    explanation = explain_prediction(pipeline, sample_input)
    assert 'base_value' in explanation, "Explanation must contain base_value."
    assert 'predicted_value' in explanation, "Explanation must contain predicted_value."
    assert len(explanation['contributions']) > 0, "Explanation must provide feature contributions."
