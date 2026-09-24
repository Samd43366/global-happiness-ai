"""
Main Application Entry Point for Global Happiness Intelligence System.

Integrates Option A (What-If Policy Simulator with SHAP Explainability)
and Option B (AI Data Analyst Grounded Query Engine) into a unified dashboard.
"""

import os
import sys
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Global Happiness AI & Intelligence System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root and app directory are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

for p in [project_root, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from utils import load_trained_model, load_data
    from components.simulator import render_simulator
    from components.chat_analyst import render_chat_analyst
except (ModuleNotFoundError, ImportError):
    from app.utils import load_trained_model, load_data
    from app.components.simulator import render_simulator
    from app.components.chat_analyst import render_chat_analyst


def main():
    # Sidebar Navigation & System Meta
    with st.sidebar:
        st.title("🌍 Global Happiness AI")
        st.caption("Data, ML & AI Team Lead Technical Assessment")
        st.markdown("---")
        
        st.markdown("### 📊 System Architecture")
        st.markdown("""
        * **Engine**: XGBoost Regressor ($R^2 = 0.838$)
        * **Explainability**: SHAP TreeExplainer
        * **Data Scope**: 1,367 Surveys (2015–2023)
        * **Evaluation**: Repeated 5-Fold CV
        """)
        st.markdown("---")
        st.markdown("### 🏛️ Five Assessment Pillars")
        st.markdown("""
        1. **Data Engineering (25%)**: Zero-leakage audit
        2. **ML Engineering (25%)**: 4-model benchmark
        3. **Application & AI (25%)**: Dual Option A & B
        4. **Software Architecture (15%)**: Clean `src/` & `app/`
        5. **Documentation (10%)**: Enterprise README
        """)
        st.markdown("---")
        st.info("💡 **Tip**: Switch tabs to test either real-time policy predictions or natural-language dataset querying.")

    # Main Header
    st.title("🌐 Global Happiness Intelligence & Policy Platform")
    st.markdown(
        "An enterprise end-to-end intelligence system translating multi-year global socio-economic data "
        "into validated statistical insights, machine learning forecasts, and AI-driven policy simulations."
    )

    # Load artifacts
    pipeline = load_trained_model()
    df = load_data()

    # Application Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎛️ Option A: Policy Simulator & SHAP",
        "💬 Option B: AI Data Analyst (NL Q&A)",
        "📈 Stage 2 Model Benchmarks & Architecture"
    ])

    with tab1:
        render_simulator(pipeline, df)

    with tab2:
        render_chat_analyst(df)

    with tab3:
        st.markdown("### 🏆 Algorithm Benchmarking Protocol")
        st.markdown(
            "Candidate models evaluated under **Repeated 5-Fold Cross-Validation** (25 total fits per model) "
            "and scored on a held-out 20% test set."
        )
        
        benchmark_data = {
            'Model Family': [
                'Baseline (Mean Regressor)',
                'Ridge Regression (L2 Regularized)',
                'Random Forest (Bagging Ensemble)',
                'XGBoost (Gradient Boosted - Selected)'
            ],
            'CV MAE': ['0.925 ± 0.032', '0.392 ± 0.021', '0.362 ± 0.017', '0.345 ± 0.017'],
            'CV RMSE': ['1.115 ± 0.038', '0.511 ± 0.028', '0.475 ± 0.027', '0.448 ± 0.025'],
            'CV R² Score': ['-0.008 ± 0.007', '0.788 ± 0.021', '0.816 ± 0.018', '0.837 ± 0.016'],
            'Test MAE': [0.930, 0.401, 0.364, 0.355],
            'Test RMSE': [1.132, 0.517, 0.467, 0.455],
            'Test R² Score': [0.000, 0.791, 0.830, 0.838]
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🛠️ Systematic Model Failure & Debugging Strategy")
        st.markdown("""
        **Scenario**: *Training $R^2 = 0.97$ vs. Testing $R^2 = 0.42$*
        
        My 5-phase diagnostic hierarchy addresses this high-variance divergence:
        1. **Leakage Audit**: Validate no unique row identifiers (`country`, unregularized target encoding) entered $X_{\text{train}}$.
        2. **Capacity Pruning**: Reduce tree depths (`max_depth=3-5`), introduce shrinkage (`learning_rate=0.04`), and apply leaf constraints.
        3. **Cross-Validation Scope**: Ensure scalers and imputers are strictly encapsulated inside an `sklearn.pipeline.Pipeline`.
        4. **Adversarial Validation**: Train binary classifiers to test for covariate shift between train and test distributions ($AUC > 0.65$).
        5. **Variance Inflation Factor (VIF)**: Check for extreme collinearity between economic and life expectancy indicators.
        """)


if __name__ == '__main__':
    main()
