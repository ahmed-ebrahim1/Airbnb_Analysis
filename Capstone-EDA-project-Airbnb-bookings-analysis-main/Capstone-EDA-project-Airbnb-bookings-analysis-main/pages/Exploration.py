import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import streamlit.components.v1 as components

# Safe import of ydata-profiling: avoid crashing the app if package or dependencies
# (e.g. pkg_resources/setuptools) are missing in the runtime environment.
ProfileReport = None
_profile_import_error = None
try:
    from ydata_profiling import ProfileReport
except Exception as e:
    _profile_import_error = e

# Set page configuration
st.set_page_config(page_title="Data Exploration & EDA", layout="wide")

# Load the data
@st.cache_data
def load_data():
    # Adjusted path handling for robustness
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..', 'Airbnb NYC 2019.csv')
    
    # If the file isn't found in the parent dir, check current dir
    if not os.path.exists(csv_path):
        csv_path = os.path.join(current_dir, 'Airbnb NYC 2019.csv')
        
    df = pd.read_csv(csv_path)
    
    # CRITICAL FIX: Convert all pandas-specific nullable types to standard numpy types
    # Plotly cannot serialize 'Int64' (nullable integers) to JSON
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            # Convert to float64 to safely handle potential NaNs and avoid serialization errors
            df[col] = df[col].astype('float64')
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float64')
            
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

# Initial Load
try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Title
st.title("🔍 Data Exploration & EDA Phase")
st.markdown("---")

# Add ydata-profiling section
st.header("📋 Automated Data Profiling Report (YData Profiling)")

with st.expander("🔧 Generate Comprehensive Data Profile Report", expanded=False):
    st.write("""
    YData Profiling generates a comprehensive statistical summary of your dataset including:
    - Variable types and missing values
    - Univariate and multivariate statistics
    - Correlations and interactions
    - Missing data patterns
    - Sample records
    """)
    
    if st.button("🚀 Generate YData Profile Report"):
        with st.spinner("Generating comprehensive profile report... This may take a moment..."):
            # If the import failed earlier, show the error and actionable instructions
            if ProfileReport is None:
                st.error("ydata-profiling is not available in this environment.")
                st.write("Import error details:")
                st.code(str(_profile_import_error))
                st.info(
                    "To fix this, ensure your runtime includes the required packages.\n"
                    "If you're running locally: `pip install ydata-profiling setuptools`\n"
                    "If deploying (Streamlit Cloud or similar), add `ydata-profiling` and `setuptools` to your requirements.txt and redeploy."
                )
            else:
                try:
                    # Generate the profile report
                    profile = ProfileReport(df, title="Airbnb NYC 2019 Dataset Report", minimal=False)
                    profile_html = profile.to_html()

                    # Display the report
                    components.html(profile_html, height=800, scrolling=True)
                    st.success("✅ Profile report generated successfully!")

                    # Option to download
                    st.download_button(
                        label="📥 Download Full Report (HTML)",
                        data=profile_html,
                        file_name="Airbnb_NYC_2019_Profile_Report.html",
                        mime="text/html"
                    )

                except Exception as e:
                    st.error(f"❌ Error generating profile: {str(e)}")

st.markdown("---")

# Explanation of EDA
st.header("📚 What is the Exploration Phase?")
with st.expander("ℹ️ What we do in the Exploration Phase", expanded=True):
    st.markdown("""
    ### Exploratory Data Analysis (EDA) is the process of:
    1. **Understanding the Data Structure**
    2. **Analyzing Individual Variables**
    3. **Exploring Relationships Between Variables**
    4. **Identifying Patterns & Insights**
    5. **Data Quality Assessment**
    6. **Feature Discovery**
    """)

st.markdown("---")

# Section 1: Understanding Data Structure
st.header("1️⃣ Understanding the Data Structure")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🔢 Total Rows", f"{len(df):,}")
with col2:
    st.metric("📊 Total Columns", df.shape[1])
with col3:
    st.metric("❌ Missing Values", f"{df.isnull().sum().sum():,}")
with col4:
    st.metric("💾 Memory Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
with col5:
    st.metric("🎯 Duplicate Rows", f"{df.duplicated().sum():,}")

# Data types overview
st.subheader("Data Types Overview")
col1, col2 = st.columns(2)

with col1:
    # Ensure types are strings for Plotly names
    dtype_counts = df.dtypes.astype(str).value_counts()
    fig = px.pie(values=dtype_counts.values, names=dtype_counts.index,
                 title="Distribution of Data Types")
    st.plotly_chart(fig, width='stretch')

with col2:
    st.write("**Column Types Summary:**")
    type_summary = pd.DataFrame({
        'Data Type': dtype_counts.index,
        'Count': dtype_counts.values
    })
    st.dataframe(type_summary, width='stretch')

st.markdown("---")

# Section 2: Univariate Analysis
st.header("2️⃣ Univariate Analysis (Individual Variables)")

tab1, tab2 = st.tabs(["Numerical Variables", "Categorical Variables"])

with tab1:
    st.subheader("📊 Numerical Variables Analysis")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    st.write("**Statistical Summary:**")
    st.dataframe(df[numerical_cols].describe().T, width='stretch')
    
    st.write("**Distributions:**")
    st.plotly_chart(fig, width='stretch')
    
    cols_to_plot = numerical_cols[:4]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Distribution of {col}" for col in cols_to_plot]
    )
    
    for idx, col in enumerate(cols_to_plot):
        row = (idx // 2) + 1
        col_pos = (idx % 2) + 1
        # Use standard numpy float64 to avoid serialization issues
        col_data = df[col].dropna().values.astype('float64')
        fig.add_trace(
            go.Histogram(x=col_data, name=col),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    
    st.write("**Box Plot for Outlier Detection:**")
    selected_num_col = st.selectbox("Select a numerical variable:", numerical_cols)
    
    # FIX: Corrected px.box arguments to avoid "multiple values for data_frame"
    fig = px.box(
        data_frame=df.assign(**{selected_num_col: df[selected_num_col].astype('float64')}),
        y=selected_num_col, 
        title=f"Box Plot of {selected_num_col}"
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("🏷️ Categorical Variables Analysis")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        selected_cat_col = st.selectbox("Select a categorical variable:", categorical_cols)
        value_counts = df[selected_cat_col].value_counts()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = px.bar(x=value_counts.index, y=value_counts.values.astype('float64'),
                        title=f"Value Counts: {selected_cat_col}",
                        labels={'x': selected_cat_col, 'y': 'Count'})
            st.plotly_chart(fig, width='stretch')
        with col2:
            st.write(f"**Summary for {selected_cat_col}:**")
            summary_cat = pd.DataFrame({
                'Unique Values': [df[selected_cat_col].nunique()],
                'Most Common': [value_counts.index[0] if not value_counts.empty else "N/A"],
                'Missing Values': [df[selected_cat_col].isnull().sum()]
            })
            st.dataframe(summary_cat, width='stretch')

st.markdown("---")

# Section 3: Bivariate Analysis
st.header("3️⃣ Bivariate Analysis")

tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Price vs Room Type", "Borough Analysis"])

with tab1:
    st.subheader("🔗 Correlation Analysis")
    # Cast everything to float64 to ensure it's serializable
    numerical_df = df.select_dtypes(include=['number']).astype('float64').dropna()
    if not numerical_df.empty:
        corr_matrix = numerical_df.corr().fillna(0)
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("💰 Price vs Room Type")
    # Clean data specifically for this chart
    plot_df = df[['room_type', 'price']].copy()
    plot_df['price'] = plot_df['price'].astype('float64')
    
    fig = px.box(plot_df, x='room_type', y='price', title="Price by Room Type")
    st.plotly_chart(fig, width='stretch')

with tab3:
    st.subheader("🗺️ Borough Analysis")
    col1, col2 = st.columns(2)
    with col1:
        counts = df['neighbourhood_group'].value_counts()
        fig = px.pie(values=counts.values.astype('float64'), names=counts.index, title="Listings by Borough")
        st.plotly_chart(fig, width='stretch')
    with col2:
        # Cast price to float for mean calculation
        avg_p = df.groupby('neighbourhood_group')['price'].mean().astype('float64').sort_values()
        fig = px.bar(x=avg_p.index, y=avg_p.values, title="Avg Price by Borough")
        st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Section 5: Outlier Detection (Corrected Types)
st.header("5️⃣ Outlier Detection")
num_cols = df.select_dtypes(include=['number']).columns.tolist()
selected_col = st.selectbox("Variable for outlier detection:", num_cols, key="outlier_sel")

if selected_col:
    data_series = df[selected_col].astype('float64').dropna()
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Outliers Count", f"{((data_series > upper) | (data_series < lower)).sum():,}")
    with col2:
        st.metric("Upper Bound", f"{upper:.2f}")

    fig = px.histogram(data_series, title=f"Distribution of {selected_col}")
    st.plotly_chart(fig, width='stretch')

st.info("Exploration complete. Standardized all data types to float64 to prevent JSON serialization errors.")