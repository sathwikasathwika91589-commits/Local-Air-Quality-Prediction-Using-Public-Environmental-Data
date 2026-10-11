import streamlit as st
st.title("Model Insights")
trained = st.session_state.get("trained_pollutants", {})
if not trained:
    st.info("Load live station history on Home, then train pollutant models on Forecast to inspect real validation and test metrics.")
else:
    from src.explainability import feature_importance
    for pollutant, result in trained.items():
        st.subheader(f"{pollutant.upper()} · selected {result['selected_model']}")
        st.dataframe(result["validation_metrics"], use_container_width=True, hide_index=True)
        st.write("Held-out chronological test metrics", result["test_metrics"])
        importance = feature_importance(result["models"][result["selected_model"]], result["feature_columns"])
        if importance is not None:
            st.caption("Tree impurity importance is descriptive and can favor high-cardinality or correlated features.")
            st.bar_chart(importance.head(20))
        else:
            st.info("The selected estimator does not expose built-in feature importance.")
