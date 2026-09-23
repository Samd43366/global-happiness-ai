"""
Option A: Policy Simulation & Explainable AI Component.

Provides interactive sliders for socio-economic parameters and renders
real-time happiness score predictions alongside SHAP waterfall explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.explainability import explain_prediction


PRESETS = {
    'Custom Scenario': None,
    'Finland (High Trust & Social Capital)': {
        'gdp_per_capita': 1.88,
        'social_support': 1.58,
        'healthy_life_expectancy': 0.77,
        'freedom_to_make_life_choices': 0.74,
        'generosity': 0.13,
        'perceptions_of_corruption': 0.04,
        'region': 'Western Europe'
    },
    'United States (High GDP, Moderate Trust)': {
        'gdp_per_capita': 1.98,
        'social_support': 1.33,
        'healthy_life_expectancy': 0.63,
        'freedom_to_make_life_choices': 0.57,
        'generosity': 0.22,
        'perceptions_of_corruption': 0.17,
        'region': 'North America and ANZ'
    },
    'Costa Rica (Latin American High Well-Being)': {
        'gdp_per_capita': 1.48,
        'social_support': 1.41,
        'healthy_life_expectancy': 0.73,
        'freedom_to_make_life_choices': 0.68,
        'generosity': 0.08,
        'perceptions_of_corruption': 0.10,
        'region': 'Latin America and Caribbean'
    },
    'Global Median Benchmark': {
        'gdp_per_capita': 1.04,
        'social_support': 1.08,
        'healthy_life_expectancy': 0.58,
        'freedom_to_make_life_choices': 0.44,
        'generosity': 0.17,
        'perceptions_of_corruption': 0.10,
        'region': 'Central and Eastern Europe'
    }
}


def render_simulator(pipeline, df: pd.DataFrame):
    """Render the interactive policy simulation interface."""
    st.markdown("### 🎛️ National Policy Simulator & Real-Time Forecast")
    st.markdown(
        "Adjust socio-economic indicators below to simulate a hypothetical nation's "
        "well-being profile and observe real-time predictions with **SHAP factor attributions**."
    )
    
    # Preset Selector
    selected_preset = st.selectbox("📌 Select a Preset Profile or Choose Custom:", list(PRESETS.keys()))
    preset_vals = PRESETS[selected_preset]
    
    col1, col2 = st.columns([1, 1])
    
    regions = sorted(df['region'].unique().tolist())
    
    with col1:
        region = st.selectbox(
            "Geopolitical Region",
            regions,
            index=regions.index(preset_vals['region']) if preset_vals else 0
        )
        gdp = st.slider(
            "GDP per Capita Index",
            min_value=0.0, max_value=2.30,
            value=float(preset_vals['gdp_per_capita']) if preset_vals else 1.05,
            step=0.02,
            help="Purchasing Power Parity (PPP) economic output index"
        )
        support = st.slider(
            "Social Support Index",
            min_value=0.0, max_value=1.70,
            value=float(preset_vals['social_support']) if preset_vals else 1.10,
            step=0.02,
            help="Perceived reliability of relatives/friends in times of crisis"
        )
        health = st.slider(
            "Healthy Life Expectancy Index",
            min_value=0.0, max_value=1.05,
            value=float(preset_vals['healthy_life_expectancy']) if preset_vals else 0.60,
            step=0.01,
            help="Normalized healthy life expectancy at birth"
        )

    with col2:
        freedom = st.slider(
            "Freedom of Choice Index",
            min_value=0.0, max_value=0.80,
            value=float(preset_vals['freedom_to_make_life_choices']) if preset_vals else 0.45,
            step=0.02,
            help="Perceived liberty to make life decisions"
        )
        generosity = st.slider(
            "Generosity Index",
            min_value=-0.10, max_value=0.60,
            value=float(preset_vals['generosity']) if preset_vals else 0.15,
            step=0.01,
            help="Charitable donation and mutual civic assistance behavior"
        )
        corruption = st.slider(
            "Perceptions of Corruption",
            min_value=0.0, max_value=0.60,
            value=float(preset_vals['perceptions_of_corruption']) if preset_vals else 0.10,
            step=0.01,
            help="Surveyed perception of government and business corruption (Lower = Cleaner)"
        )
        
    input_df = pd.DataFrame([{
        'gdp_per_capita': gdp,
        'social_support': support,
        'healthy_life_expectancy': health,
        'freedom_to_make_life_choices': freedom,
        'generosity': generosity,
        'perceptions_of_corruption': corruption,
        'region': region
    }])
    
    # Run prediction and SHAP explanation
    pred_score = float(pipeline.predict(input_df)[0])
    explanation = explain_prediction(pipeline, input_df)
    
    global_mean = 5.441
    delta = pred_score - global_mean
    
    st.divider()
    
    # Display Forecast Metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric(
            label="Predicted Happiness Score",
            value=f"{pred_score:.3f} / 10.0",
            delta=f"{delta:+.3f} vs Global Avg ({global_mean:.2f})"
        )
    with mcol2:
        tier = "Elite Flourishing (>7.0)" if pred_score >= 7.0 else ("Moderate Well-Being (5.0–7.0)" if pred_score >= 5.0 else "Vulnerable Well-Being (<5.0)")
        st.metric(label="Societal Tier", value=tier)
    with mcol3:
        st.metric(label="Model Prior (Base Value)", value=f"{explanation['base_value']:.3f}")
        
    st.divider()
    
    # SHAP Feature Attribution Waterfall
    st.markdown("### 🔍 Explainable AI: SHAP Feature Attribution")
    st.markdown(
        "The waterfall below displays how each specific input pushed the prediction "
        "**above** (green) or **below** (crimson) the model baseline."
    )
    
    contributions = explanation['contributions'][:8]
    labels = [c['feature'].replace('region_', 'Region: ') for c in contributions][::-1]
    values = [c['shap_value'] for c in contributions][::-1]
    colors = ['forestgreen' if v > 0 else 'crimson' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_xlabel('SHAP Impact on Cantril Ladder Score (Points)')
    ax.set_title('Local Feature Contributions to Current Prediction', pad=10)
    
    for bar in bars:
        width = bar.get_width()
        ha = 'left' if width >= 0 else 'right'
        offset = 0.01 if width >= 0 else -0.01
        ax.annotate(
            f"{width:+.3f}",
            xy=(width + offset, bar.get_y() + bar.get_height() / 2),
            xytext=(0, 0), textcoords="offset points",
            ha=ha, va='center', fontsize=9, fontweight='bold'
        )
        
    plt.tight_layout()
    st.pyplot(fig)
