import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Bivariate Analysis", layout="wide")

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Airbnb NYC 2019.csv')
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if str(df[col].dtype).startswith('Int'):
            df[col] = df[col].astype('float64')
    return df

df = load_data()

st.title("🔗 Bivariate Analysis")
st.markdown("Explore relationships between pairs of variables")
st.markdown("---")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# 1. Numeric vs Numeric
st.header("1️⃣ Numeric vs Numeric")
col1, col2 = st.columns(2)

with col1:
    x_var = st.selectbox("X variable:", numerical_cols, key="x_var")
    y_var = st.selectbox("Y variable:", numerical_cols, index=1 if len(numerical_cols) > 1 else 0, key="y_var")
    show_trend = st.checkbox("Add trendline", value=True)

with col2:
    pair_data = df[[x_var, y_var]].dropna().copy().astype('float64')
    if len(pair_data) > 0:
        corr = pair_data.corr().iloc[0, 1]
        st.metric("Pearson Correlation", f"{corr:.3f}")
        
        fig = px.scatter(pair_data, x=x_var, y=y_var, 
                        trendline='ols' if show_trend else None,
                        title=f"{x_var} vs {y_var}")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 2. Numeric vs Categorical
st.header("2️⃣ Numeric vs Categorical")
cat_cols_short = [c for c in categorical_cols if df[c].nunique() < 30]

col1, col2 = st.columns(2)
with col1:
    num_var = st.selectbox("Numeric:", numerical_cols, key="num_cat")
    if cat_cols_short:
        cat_var = st.selectbox("Categorical:", cat_cols_short, key="cat_num")
    else:
        st.warning("No suitable categorical columns")
        cat_var = None

with col2:
    if cat_var:
        plot_df = df[[num_var, cat_var]].dropna().copy().astype({num_var: 'float64'})
        if len(plot_df) > 0:
            fig = px.box(plot_df, x=cat_var, y=num_var, title=f"{num_var} by {cat_var}")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 3. Categorical vs Categorical
st.header("3️⃣ Categorical vs Categorical")
if len(cat_cols_short) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        cat_a = st.selectbox("Category A:", cat_cols_short, key="cat_a")
        cat_b = st.selectbox("Category B:", [c for c in cat_cols_short if c != cat_a], key="cat_b")
    
    with col2:
        ct = pd.crosstab(df[cat_a], df[cat_b])
        if ct.size > 0:
            fig = px.bar(ct, barmode='stack', title=f"{cat_a} vs {cat_b}")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need at least 2 categorical columns with <30 unique values")

st.markdown("---")

# 4. Correlation Matrix
st.header("4️⃣ Correlation Heatmap")
if st.button("Generate Heatmap"):
    num_df = df[numerical_cols].dropna().astype('float64')
    if len(num_df) > 0 and len(numerical_cols) > 1:
        corr_matrix = num_df.corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', 
                       zmin=-1, zmax=1, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric columns")