import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Set page configuration
st.set_page_config(page_title="Airbnb NYC 2019 - Data Description", layout="wide")

# Title and description
st.title("📊 Airbnb NYC 2019 Dataset - Data Description")
st.markdown("---")

# Load the data
@st.cache_data
def load_data():
    import os
    csv_path = os.path.join(os.path.dirname(__file__), 'Airbnb NYC 2019.csv')
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# Display basic information
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Total Columns", df.shape[1])
with col3:
    st.metric("Missing Values", df.isnull().sum().sum())
with col4:
    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

st.markdown("---")

# Show the first few rows
st.subheader("📋 First Few Rows of Data")
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# Data types and missing values
st.subheader("📌 Data Types and Missing Values")
data_info = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values,
    'Non-Null Count': df.count().values,
    'Null Count': df.isnull().sum().values,
    'Null %': (df.isnull().sum().values / len(df) * 100).round(2)
})
st.dataframe(data_info, use_container_width=True)

st.markdown("---")

# Detailed column descriptions
st.subheader("📝 Detailed Column Descriptions")

column_descriptions = {
    'id': 'Unique identifier for each listing',
    'name': 'Name/title of the Airbnb listing',
    'host_id': 'Unique identifier for the host',
    'host_name': 'Name of the host',
    'neighbourhood_group': 'General area/borough (e.g., Brooklyn, Manhattan)',
    'neighbourhood': 'Specific neighbourhood within the borough',
    'latitude': 'Latitude coordinate of the listing',
    'longitude': 'Longitude coordinate of the listing',
    'room_type': 'Type of room (Entire home/apt, Private room, Shared room)',
    'price': 'Price per night in USD',
    'minimum_nights': 'Minimum number of nights required for booking',
    'number_of_reviews': 'Total number of reviews received',
    'last_review': 'Date of the most recent review (YYYY-MM-DD)',
    'reviews_per_month': 'Average reviews per month',
    'calculated_host_listings_count': 'Total number of listings by this host',
    'availability_365': 'Number of days available in a year (0-365)'
}

for col in df.columns:
    with st.expander(f"**{col}** - {column_descriptions.get(col, 'N/A')}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Type:**", df[col].dtype)
            st.write("**Non-Null Count:**", df[col].count())
            st.write("**Null Count:**", df[col].isnull().sum())
        
        with col2:
            if df[col].dtype in ['int64', 'float64']:
                st.write("**Min:**", df[col].min())
                st.write("**Max:**", df[col].max())
                st.write("**Mean:**", df[col].mean().round(2))
        
        # Display unique values or statistical summary
        if df[col].dtype == 'object':
            st.write(f"**Unique Values:** {df[col].nunique()}")
            if df[col].nunique() <= 20:
                st.write("**Value Counts:**")
                st.write(df[col].value_counts())
        else:
            st.write("**Statistical Summary:**")
            st.write(df[col].describe().round(2))

st.markdown("---")

# Statistical summary for numerical columns
st.subheader("📊 Statistical Summary of Numerical Columns")
st.dataframe(df.describe().T, use_container_width=True)

st.markdown("---")

# Visualizations
st.subheader("📈 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("**Room Type Distribution**")
    room_counts = df['room_type'].value_counts()
    fig1 = px.pie(values=room_counts.values, names=room_counts.index, 
                   title="Distribution of Room Types")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.write("**Listings by Neighbourhood Group**")
    neighbourhood_counts = df['neighbourhood_group'].value_counts()
    fig2 = px.bar(x=neighbourhood_counts.index, y=neighbourhood_counts.values,
                   title="Number of Listings by Neighbourhood Group",
                   labels={'x': 'Neighbourhood Group', 'y': 'Count'})
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.write("**Price Distribution**")
    fig3 = px.histogram(df, x='price', nbins=50, 
                        title="Distribution of Prices",
                        labels={'price': 'Price (USD)'})
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.write("**Reviews per Month Distribution**")
    fig4 = px.histogram(df, x='reviews_per_month', nbins=50,
                        title="Distribution of Reviews per Month",
                        labels={'reviews_per_month': 'Reviews per Month'})
    st.plotly_chart(fig4, use_container_width=True)

# Geographic map
st.subheader("🗺️ Geographic Distribution of Listings")
st.map(df[['latitude', 'longitude']])

st.markdown("---")

# Download option
st.subheader("📥 Download Data")
csv = df.to_csv(index=False)
st.download_button(
    label="Download dataset as CSV",
    data=csv,
    file_name="Airbnb_NYC_2019.csv",
    mime="text/csv"
)