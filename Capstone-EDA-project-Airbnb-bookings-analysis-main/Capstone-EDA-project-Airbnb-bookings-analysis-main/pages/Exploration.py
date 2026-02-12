import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import streamlit.components.v1 as components

# --- Setup and Data Loading (Kept from your original) ---
st.set_page_config(page_title="Data Exploration & EDA", layout="wide")

@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..', 'Airbnb NYC 2019.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(current_dir, 'Airbnb NYC 2019.csv')
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float64')
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.title("🔍 Data Exploration & EDA Phase")

# --- Section 1: Understanding Data Structure ---
st.header("1️⃣ Understanding the Data Structure")
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("🔢 Total Rows", f"{len(df):,}")
with col2: st.metric("📊 Total Columns", df.shape[1])

st.subheader("Data Types Overview")
c1, c2 = st.columns(2)
with c1:
    dtype_counts = df.dtypes.astype(str).value_counts()
    fig_pie_dtypes = px.pie(values=dtype_counts.values, names=dtype_counts.index, title="Data Types")
    # FIX: Added unique key
    st.plotly_chart(fig_pie_dtypes, use_container_width=True, key="dtype_pie_chart")

# --- Section 2: Univariate Analysis ---
st.header("2️⃣ Univariate Analysis")
tab1, tab2 = st.tabs(["Numerical Variables", "Categorical Variables"])

with tab1:
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    st.write("**Statistical Summary:**")
    st.dataframe(df[numerical_cols].describe().T, use_container_width=True)
    
    # FIX: Removed the stray st.plotly_chart(fig) that was here before fig was defined
    st.write("**Distributions (Top 4 Variables):**")
    cols_to_plot = numerical_cols[:4]
    fig_multi = make_subplots(rows=2, cols=2, subplot_titles=[f"Dist of {col}" for col in cols_to_plot])
    
    for idx, col in enumerate(cols_to_plot):
        row, cp = (idx // 2) + 1, (idx % 2) + 1
        fig_multi.add_trace(go.Histogram(x=df[col].dropna().values.astype('float64'), name=col), row=row, col=cp)
    
    fig_multi.update_layout(height=600, showlegend=False)
    # FIX: Added unique key
    st.plotly_chart(fig_multi, use_container_width=True, key="multi_hist_chart")
    
    st.write("**Box Plot for Outlier Detection:**")
    sel_num = st.selectbox("Select variable:", numerical_cols, key="univariate_num_box_sel")
    fig_box_uni = px.box(y=df[sel_num].astype('float64'), title=f"Box Plot of {sel_num}")
    # FIX: Added unique key
    st.plotly_chart(fig_box_uni, use_container_width=True, key="uni_box_plot")

with tab2:
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        sel_cat = st.selectbox("Select variable:", categorical_cols, key="univariate_cat_sel")
        v_counts = df[sel_cat].value_counts()
        fig_cat_bar = px.bar(x=v_counts.index, y=v_counts.values.astype('float64'), title=f"Counts: {sel_cat}")
        # FIX: Added unique key
        st.plotly_chart(fig_cat_bar, use_container_width=True, key="cat_bar_chart")

# --- Section 3: Bivariate Analysis ---
st.header("3️⃣ Bivariate Analysis")
b1, b2, b3 = st.tabs(["Correlation", "Price vs Room Type", "Borough Analysis"])

with b1:
    num_df = df.select_dtypes(include=['number']).astype('float64').dropna()
    if not num_df.empty:
        fig_corr = px.imshow(num_df.corr().fillna(0), text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu")
        # FIX: Added unique key
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_heatmap")

with b2:
    fig_price_room = px.box(df, x='room_type', y='price', title="Price by Room Type")
    # FIX: Added unique key
    st.plotly_chart(fig_price_room, use_container_width=True, key="price_room_box")

with b3:
    col_a, col_b = st.columns(2)
    with col_a:
        b_counts = df['neighbourhood_group'].value_counts()
        fig_b_pie = px.pie(values=b_counts.values.astype('float64'), names=b_counts.index, title="Listings by Borough")
        # FIX: Added unique key
        st.plotly_chart(fig_b_pie, use_container_width=True, key="borough_pie")
    with col_b:
        avg_p = df.groupby('neighbourhood_group')['price'].mean().astype('float64').sort_values()
        fig_b_bar = px.bar(x=avg_p.index, y=avg_p.values, title="Avg Price by Borough")
        # FIX: Added unique key
        st.plotly_chart(fig_b_bar, use_container_width=True, key="borough_price_bar")

# --- Section 5: Outlier Detection ---
st.header("5️⃣ Outlier Detection")
sel_out = st.selectbox("Variable:", numerical_cols, key="outlier_sel_final")
if sel_out:
    data_s = df[sel_out].astype('float64').dropna()
    fig_out_hist = px.histogram(data_s, title=f"Dist of {sel_out}")
    # FIX: Added unique key
    st.plotly_chart(fig_out_hist, use_container_width=True, key="outlier_hist_chart")