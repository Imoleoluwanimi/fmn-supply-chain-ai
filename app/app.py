"""
Project 1: Supply Chain Demand and Risk
Streamlit app entry point.

This app shows which SKUs need attention, why, and lets a business user
ask free-text questions about the data. It loads a pre-trained model and
the pre-computed flags by default, and also accepts a freshly uploaded
CSV in the same format to recompute today's flags on the spot.
"""

import pandas as pd
import joblib
import streamlit as st

import data_prep as dp
from explain import generate_explanation, answer_question

st.set_page_config(page_title="FMN Supply Chain Risk Dashboard", layout="wide")

# FMN official brand colors: green as primary, yellow as accent, white and
# charcoal for contrast and typography.
GREEN = "#18864B"
GREEN_DARK = "#198A4D"
YELLOW = "#FFD500"
YELLOW_LIGHT = "#FFDD00"
WHITE = "#FFFFFF"
CHARCOAL = "#1A1A1A"
BACKGROUND = "#F6FAF7"

CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: {BACKGROUND};
}}

.fmn-header {{
    background: linear-gradient(120deg, {GREEN} 0%, {GREEN_DARK} 100%);
    padding: 1.75rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    border-bottom: 4px solid {YELLOW};
}}
.fmn-header h1 {{
    color: {WHITE};
    margin: 0;
    font-size: 1.6rem;
}}
.fmn-header p {{
    color: {YELLOW};
    margin: 0.25rem 0 0 0;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}

div[data-testid="stMetric"] {{
    background-color: {WHITE};
    border: 1px solid #E3ECE6;
    border-left: 5px solid {GREEN};
    border-radius: 8px;
    padding: 1rem;
}}
div[data-testid="stMetricValue"] {{
    color: {CHARCOAL};
}}

h2, h3 {{
    color: {CHARCOAL};
}}

/* Sidebar: dark charcoal so the yellow accents and white text stay visible */
section[data-testid="stSidebar"] {{
    background-color: {CHARCOAL};
}}
section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}

/* File uploader dropzone was invisible before, blending white on white.
   Give it a dark panel with a yellow dashed border so it clearly reads
   as an upload target. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background-color: #2A2A2A;
    border: 2px dashed {YELLOW};
    border-radius: 8px;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
    background-color: {YELLOW};
    color: {CHARCOAL} !important;
    font-weight: 700;
    border: none;
}}

.stButton button {{
    background-color: {YELLOW};
    color: {CHARCOAL};
    font-weight: 700;
    border: none;
}}
.stButton button:hover {{
    background-color: {YELLOW_LIGHT};
    color: {CHARCOAL};
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid #E3ECE6;
    border-radius: 8px;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="fmn-header">
        <h1>FMN Supply Chain Risk Dashboard</h1>
        <p>FEEDING THE NATION, EVERY DAY &nbsp;|&nbsp; Project 1: Demand and Inventory Risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("app/demand_forecast_model.joblib")


@st.cache_data
def load_bundled_flags():
    established = pd.read_csv("app/established_sku_flags.csv")
    new_skus = pd.read_csv("app/new_sku_flags.csv")
    
    # Work with whatever format these CSVs are currently in. Older versions
    # do not have a separate confidence column, and instead bake the
    # "(early estimate)" note directly into risk_flag. Handle both cases
    # here so the app works without needing the notebook rerun.
    if "confidence" not in established.columns:
        established["confidence"] = "Model-based"

    if "confidence" not in new_skus.columns:
        new_skus["confidence"] = "Early estimate, limited history"
        new_skus["risk_flag"] = (
            new_skus["risk_flag"].str.replace(r"\s*\(early estimate\)", "", regex=True)
        )

    return established, new_skus


def run_pipeline_on_upload(uploaded_file, model):
    raw = pd.read_csv(uploaded_file)
    return dp.run_full_pipeline(raw, model)


def flag_style(value):
    if "stockout" in value:
        return "background-color: #f8d7da"
    if "overstock" in value:
        return "background-color: #fff3cd"
    return ""


def main():
    model = load_model()

    with st.sidebar:
        st.header("Data source")
        uploaded_file = st.file_uploader(
            "Upload a fresh daily export (same column format)",
            type="csv",
        )
        st.caption(
            "If no file is uploaded, the dashboard uses the historical "
            "dataset provided with this assessment."
        )

    if uploaded_file is not None:
        with st.spinner("Processing uploaded data..."):
            established, new_skus = run_pipeline_on_upload(uploaded_file, model)
        st.success("Using freshly uploaded data.")
    else:
        established, new_skus = load_bundled_flags()

    all_flags = pd.concat(
        [
            established[["sku_id", "category", "closing_stock", "forecasted_demand",
                          "days_of_cover", "lead_time_days", "risk_flag", "confidence"]],
            new_skus.rename(columns={"latest_stock": "closing_stock",
                                      "avg_daily_sales": "forecasted_demand"})
                    [["sku_id", "category", "closing_stock", "forecasted_demand",
                      "days_of_cover", "lead_time_days", "risk_flag", "confidence"]],
        ],
        ignore_index=True,
    )

    flagged_count = all_flags["risk_flag"].apply(lambda x: "ok" not in x).sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs", len(all_flags))
    col2.metric("Flagged for attention", int(flagged_count))
    col3.metric("Healthy", len(all_flags) - int(flagged_count))

    st.markdown("### All SKUs")
    with st.container(border=True):
        sort_order = all_flags["risk_flag"].apply(lambda x: 0 if "ok" not in x else 1)
        display_table = all_flags.assign(_sort=sort_order).sort_values("_sort").drop(columns="_sort")
        st.dataframe(
            display_table.style.map(flag_style, subset=["risk_flag"]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### SKU detail")
    with st.container(border=True):
        selected_sku = st.selectbox("Select a SKU to see the full reasoning", all_flags["sku_id"])
        sku_row = all_flags[all_flags["sku_id"] == selected_sku].iloc[0]

        detail_col, explain_col = st.columns([1, 2])

        with detail_col:
            st.metric("Risk flag", sku_row["risk_flag"])
            st.caption(sku_row["confidence"])
            st.metric("Days of cover", f"{sku_row['days_of_cover']:.1f}")
            st.metric("Lead time (days)", int(sku_row["lead_time_days"]))
            st.metric("Forecasted daily demand", f"{sku_row['forecasted_demand']:.0f}")
            st.metric("Current stock", f"{sku_row['closing_stock']:.0f}")

        with explain_col:
            st.markdown("**Why this SKU is flagged**")
            if st.button("Generate AI explanation"):
                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(sku_row.to_dict())
                st.write(explanation)

    st.markdown("### Ask a question")
    with st.container(border=True):
        question = st.text_input(
            "e.g. \"why is SKU-1004 flagged\" or \"which SKUs need attention this week\""
        )
        if question:
            with st.spinner("Thinking..."):
                answer = answer_question(question, all_flags)
            st.write(answer)


if __name__ == "__main__":
    main()
