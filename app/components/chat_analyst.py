"""
Option B: AI Data Analyst Chat Component.

Provides a natural-language query interface backed by deterministic structured
query execution on the World Happiness dataset with zero hallucination.
"""

import streamlit as st
import pandas as pd
from src.llm_analyst import AIDataAnalyst


def render_chat_analyst(df: pd.DataFrame):
    """Render the AI Data Analyst query interface."""
    st.markdown("### 💬 AI Data Analyst (Zero-Hallucination Query Engine)")
    st.markdown(
        "Ask questions about the global dataset in natural language. Queries are parsed and "
        "executed deterministically against verified data tables to prevent hallucination."
    )
    
    analyst = AIDataAnalyst(data_path='cleaned_data.csv')
    
    # Quick Action Prompt Buttons
    st.markdown("**Quick Query Prompts:**")
    qcols = st.columns(4)
    quick_query = None
    with qcols[0]:
        if st.button("🏆 Top 5 Happiest"):
            quick_query = "What are the top 5 happiest countries in 2023?"
    with qcols[1]:
        if st.button("🌍 Regional Rankings"):
            quick_query = "Show regional happiness rankings and averages"
    with qcols[2]:
        if st.button("🌴 Latin America Paradox"):
            quick_query = "Explain the Latin America happiness anomaly"
    with qcols[3]:
        if st.button("📈 Feature Correlations"):
            quick_query = "What factors have the highest correlation with happiness?"

    # Session Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your AI Data Analyst. Ask me anything about global happiness trends, "
                    "regional disparities, economic trade-offs, or country performance."
                ),
                "data": None
            }
        ]

    # Handle quick query or user input
    user_input = st.chat_input("Ask a question about the dataset (e.g. 'Which countries have the highest social support?')...")
    query_to_run = quick_query if quick_query else user_input

    if query_to_run:
        st.session_state.messages.append({"role": "user", "content": query_to_run, "data": None})
        result = analyst.query(query_to_run)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "data": result['data'],
            "method": result['method']
        })

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("data") is not None:
                st.dataframe(msg["data"], use_container_width=True)
            if msg.get("method"):
                st.caption(f"🔬 *Execution Method: {msg['method']}*")
