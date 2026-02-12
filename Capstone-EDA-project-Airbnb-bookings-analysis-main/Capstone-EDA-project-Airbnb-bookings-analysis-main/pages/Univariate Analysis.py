import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Univariate Analysis", layout="wide")

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

st.title("📊 Univariate Analysis")
st.markdown("Quickly explore individual variable distributions and key statistics")
st.markdown("---")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Numerical Variables
st.header("1️⃣ Numerical Variables")
selected_num = st.selectbox("Select variable:", numerical_cols, key="num_var")

if selected_num:
    clean_data = df[selected_num].dropna().astype('float64')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{clean_data.mean():.2f}")
        st.metric("Median", f"{clean_data.median():.2f}")
    with col2:
        st.metric("Std Dev", f"{clean_data.std():.2f}")
        st.metric("Range", f"{clean_data.max() - clean_data.min():.2f}")
    with col3:
        st.metric("Min", f"{clean_data.min():.2f}")
        st.metric("Max", f"{clean_data.max():.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(data_frame=pd.DataFrame({selected_num: clean_data}), 
                          x=selected_num, nbins=40, title="Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        box_data = pd.DataFrame({selected_num: clean_data})
        fig = px.box(box_data, y=selected_num, title="Box Plot")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Categorical Variables
st.header("2️⃣ Categorical Variables")
selected_cat = st.selectbox("Select variable:", categorical_cols, key="cat_var")

if selected_cat:
    value_counts = df[selected_cat].value_counts().head(12)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Values", df[selected_cat].nunique())
    with col2:
        st.metric("Most Common", value_counts.index[0])
    with col3:
        st.metric("Missing", df[selected_cat].isnull().sum())
    
    freq_df = pd.DataFrame({'Category': value_counts.index, 'Count': value_counts.values})
    fig = px.bar(freq_df, x='Category', y='Count', title="Top Categories")
    st.plotly_chart(fig, use_container_width=True)
