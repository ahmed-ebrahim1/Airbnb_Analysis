import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(page_title="Bivariate Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    # Construct path to the CSV file located in the parent directory
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Airbnb NYC 2019.csv')
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Convert pandas nullable ints to float for compatibility with Plotly/Streamlit
    for col in df.columns:
        if str(df[col].dtype).startswith('Int'):
            df[col] = df[col].astype('float64')
    return df

df = load_data()

st.title("🔗 Bivariate Analysis Phase")
st.markdown("---")

with st.expander("ℹ️ What we do in the Bivariate Phase", expanded=True):
    st.markdown(
        """
        Bivariate analysis examines relationships between two variables. Common goals:
        - Quantify correlation between numeric variables (Pearson / Spearman).
        - Visualize relationships (scatter plots, line plots, regression trendlines).
        - Compare numeric distributions across categories (box plots, violin plots).
        - Analyze association between categorical variables (contingency tables, stacked bars).
        """
    )

st.markdown("---")

# Prepare variable lists
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# --- Section 1: Numeric vs Numeric ---
st.header("1️⃣ Numeric vs Numeric")
col1, col2 = st.columns([1, 2])

with col1:
    x_var = st.selectbox("X variable (numeric)", numerical_cols, key="biv_x_numeric")
    y_var = st.selectbox("Y variable (numeric)", numerical_cols, index=1 if len(numerical_cols) > 1 else 0, key="biv_y_numeric")
    corr_method = st.selectbox("Correlation method", ['pearson', 'spearman'], key="biv_corr_method")
    show_trend = st.checkbox("Show regression trendline", value=True, key="biv_show_trend")
    
    if st.button("Compute correlation", key="btn_compute_corr"):
        pair = df[[x_var, y_var]].dropna()
        if len(pair) < 2:
            st.warning("Not enough data to compute correlation.")
        else:
            corr = pair.corr(method=corr_method).iloc[0,1]
            st.metric(f"{corr_method.title()} correlation", f"{corr:.3f}")

with col2:
    pair_plot_data = df[[x_var, y_var]].dropna().copy()
    pair_plot_data[x_var] = pair_plot_data[x_var].astype('float64')
    pair_plot_data[y_var] = pair_plot_data[y_var].astype('float64')
    fig = px.scatter(pair_plot_data, x=x_var, y=y_var, trendline='ols' if show_trend else None,
                     title=f"Scatter Plot: {x_var} vs {y_var}")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Section 2: Numeric vs Categorical ---
st.header("2️⃣ Numeric vs Categorical")
col3, col4 = st.columns(2)

with col3:
    num_var = st.selectbox("Select Numeric Variable", numerical_cols, key="num_select_biv")
    cat_var = st.selectbox("Select Categorical Variable", categorical_cols, key="cat_select_biv")
    agg_func = st.selectbox("Choose Aggregation", ['mean', 'median', 'count'], key="agg_select_biv")

with col4:
    plot_df = df[[num_var, cat_var]].dropna().copy()
    plot_df[num_var] = plot_df[num_var].astype('float64')
    
    if len(plot_df) == 0:
        st.warning("No data available for this pair.")
    else:
        st.subheader(f"Distribution: {num_var} by {cat_var}")
        fig_box = px.box(plot_df, x=cat_var, y=num_var, points='outliers', title="Box Plot")
        st.plotly_chart(fig_box, use_container_width=True)

# Grouped summary logic (Placed outside columns for better width)
if st.button("Show Grouped Summary Table", key="btn_show_summary"):
    grp = df.groupby(cat_var)[num_var].agg(agg_func).sort_values(ascending=False)
    st.dataframe(grp.reset_index().rename(columns={num_var: f"{agg_func}_{num_var}"}), use_container_width=True)
    
    fig_bar = px.bar(grp, x=grp.index, y=grp.values, title=f"{agg_func.title()} of {num_var} by {cat_var}")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --- Section 3: Categorical vs Categorical ---
st.header("3️⃣ Categorical vs Categorical")
col5, col6 = st.columns(2)

with col5:
    cat_a = st.selectbox("Category A", categorical_cols, key="cat_a_select")
    cat_b = st.selectbox("Category B", [c for c in categorical_cols if c != cat_a], key="cat_b_select")

with col6:
    ct = pd.crosstab(df[cat_a], df[cat_b], normalize='index')
    if ct.size == 0:
        st.warning("Not enough data for contingency table.")
    else:
        st.subheader("Contingency Table (Proportions)")
        st.dataframe(ct.round(3), use_container_width=True)
        
        ct_plot = pd.crosstab(df[cat_a], df[cat_b])
        fig_stack = px.bar(ct_plot, barmode='stack', title=f"Stacked Counts: {cat_a} by {cat_b}")
        st.plotly_chart(fig_stack, use_container_width=True)

st.markdown("---")

# --- Section 4: Correlation Matrix ---
st.header("4️⃣ Correlation Matrix & Heatmap")
if st.button("Generate Heatmap", key="btn_heatmap"):
    num_df = df[numerical_cols].copy().dropna()
    if num_df.empty:
        st.warning("No numeric data available.")
    else:
        corr_matrix = num_df.corr().fillna(0)
        fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                             title="Numeric Correlation Matrix")
        st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# Export Option
with st.expander("📥 Export Current Pair Data"):
    export_df = df[[x_var, y_var]].dropna().head(1000)
    csv_data = export_df.to_csv(index=False)
    st.download_button("Download CSV", data=csv_data, file_name=f"bivariate_{x_var}_{y_var}.csv", mime='text/csv', key="btn_download")