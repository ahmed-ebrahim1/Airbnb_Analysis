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
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Airbnb NYC 2019.csv')
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    # convert pandas nullable ints to float
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
        - Identify interaction effects and conditional patterns.

        Typical steps:
        1. Choose variable pairs to investigate (numeric-numeric, numeric-categorical, categorical-categorical).
        2. Clean and align data (drop NaNs for chosen pair).
        3. Visualize with appropriate chart and, where useful, add summary statistics.
        4. Quantify strength and direction (correlation coefficients, group means, chi-square tests).
        5. Interpret results and flag possible causal hypotheses or data issues.
        """
    )

st.markdown("---")

# Prepare variable lists
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Section: Numeric vs Numeric
st.header("1️⃣ Numeric vs Numeric")
col1, col2 = st.columns([1, 2])

with col1:
    x_var = st.selectbox("X variable (numeric)", numerical_cols, key="biv_x")
    y_var = st.selectbox("Y variable (numeric)", numerical_cols, index=1 if len(numerical_cols) > 1 else 0, key="biv_y")
    corr_method = st.selectbox("Correlation method", ['pearson', 'spearman'], key="corr_method")
    show_trend = st.checkbox("Show regression trendline", value=True)
    if st.button("Compute correlation"):
        pair = df[[x_var, y_var]].dropna()
        if len(pair) < 2:
            st.warning("Not enough data after dropping NaNs to compute correlation.")
        else:
            corr = pair.corr(method=corr_method).iloc[0,1]
            st.metric(f"{corr_method.title()} correlation", f"{corr:.3f}")

with col2:
    pair_plot_data = df[[x_var, y_var]].dropna().copy()
    # ensure numeric types
    pair_plot_data[x_var] = pair_plot_data[x_var].astype('float64')
    pair_plot_data[y_var] = pair_plot_data[y_var].astype('float64')
    fig = px.scatter(pair_plot_data, x=x_var, y=y_var, trendline='ols' if show_trend else None,
                     title=f"Scatter: {x_var} vs {y_var}")
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Section: Numeric vs Categorical
st.header("2️⃣ Numeric vs Categorical")
col1, col2 = st.columns(2)
with col1:
    num_var = st.selectbox("Numeric variable", numerical_cols, key="num_for_cat")
    cat_var = st.selectbox("Categorical variable", categorical_cols, key="cat_for_num")
    agg_func = st.selectbox("Aggregation", ['mean', 'median', 'count'], key="agg_func")

with col2:
    plot_df = df[[num_var, cat_var]].dropna().copy()
    # ensure numeric type for plotting
    plot_df[num_var] = plot_df[num_var].astype('float64')
    if len(plot_df) == 0:
        st.warning("No data available for the selected pair after dropping missing values.")
    else:
        st.subheader(f"Distribution of {num_var} across {cat_var}")
        fig1 = px.box(plot_df, x=cat_var, y=num_var, points='outliers', title=f"Box plot of {num_var} by {cat_var}")
        st.plotly_chart(fig1, width='stretch')
        # Violin plot
        fig2 = px.violin(plot_df, x=cat_var, y=num_var, box=True, points=False, title=f"Violin plot of {num_var} by {cat_var}")
        st.plotly_chart(fig2, width='stretch')
# (Removed duplicated Numeric-vs-Categorical control block — first instance above is retained)

# Grouped summary
if st.button("Show grouped summary"):
    grp = df.groupby(cat_var)[num_var].agg(agg_func).sort_values(ascending=False)
    grp = grp.astype('float64')
    st.dataframe(grp.reset_index().rename(columns={num_var: f"{agg_func}_{num_var}"}), width='stretch')
    fig_bar = px.bar(grp, x=grp.index, y=grp.values, title=f"{agg_func.title()} of {num_var} by {cat_var}")
    st.plotly_chart(fig_bar, width='stretch')

st.markdown("---")

# Section: Categorical vs Categorical
st.header("3️⃣ Categorical vs Categorical")
col1, col2 = st.columns(2)
with col1:
    cat_a = st.selectbox("Category A", categorical_cols, key="cat_a")
    cat_b = st.selectbox("Category B", [c for c in categorical_cols if c != cat_a], key="cat_b")

with col2:
    ct = pd.crosstab(df[cat_a], df[cat_b], normalize='index')
    if ct.size == 0:
        st.warning("Not enough data to compute contingency table.")
    else:
        st.subheader("Contingency Table (proportions by row)")
        st.dataframe(ct.round(3), width='stretch')
        # Stacked bar chart
        ct_plot = pd.crosstab(df[cat_a], df[cat_b])
        fig = px.bar(ct_plot, barmode='stack', title=f"Stacked Counts: {cat_a} by {cat_b}")
        st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Section: Correlation Matrix and Heatmap
st.header("4️⃣ Correlation Matrix & Heatmap")
if st.button("Compute correlation matrix"):
    num_df = df[numerical_cols].copy().dropna()
    if num_df.shape[0] == 0:
        st.warning("No numeric data available to compute correlations.")
    else:
        corr = num_df.corr().fillna(0)
        fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu', zmin=-1, zmax=1,
                        title="Correlation matrix (numeric features)")
        st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Interpretation and Next Steps
st.header("📝 Interpretation & Next Steps")
st.markdown(
    """
    - Strong correlations may indicate predictive relationships but do not imply causation.
    - Differences in distribution across categories can inform feature engineering (e.g., encoding, binning).
    - Significant association between categories may suggest interaction terms for modeling.
    - Use these bivariate insights to prioritize features for modeling and deeper multivariate analysis.
    """
)

# Export option
with st.expander("📥 Export pair summary"):
    sample_pairs = []
    sample_pairs.append({'x': x_var, 'y': y_var})
    sample_df = df[[x_var, y_var]].dropna().head(1000)
    csv = sample_df.to_csv(index=False)
    st.download_button("Download sample pair CSV", data=csv, file_name=f"pair_{x_var}_{y_var}.csv", mime='text/csv')
