import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy import stats
from scipy.stats import shapiro, normaltest, skew, kurtosis

# Set page configuration
st.set_page_config(page_title="Univariate Analysis", layout="wide")

# Load the data
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Airbnb NYC 2019.csv')
    df = pd.read_csv(csv_path)
    
    # Clean data: Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Convert pandas nullable types to standard types
    for col in df.columns:
        if df[col].dtype == 'Int64':
            df[col] = df[col].astype('float64')
        elif df[col].dtype == 'Int32':
            df[col] = df[col].astype('float64')
    
    return df

df = load_data()

# Title
st.title("📊 Univariate Analysis Phase")
st.markdown("---")

# Explanation of Univariate Analysis
st.header("📚 What is Univariate Analysis?")

with st.expander("ℹ️ Understanding Univariate Analysis", expanded=True):
    st.markdown("""
    ### Univariate Analysis examines ONE variable at a time to understand:
    
    **1. Central Tendency (Location)**
    - **Mean**: Average value
    - **Median**: Middle value (resistant to outliers)
    - **Mode**: Most frequently occurring value
    
    **2. Spread/Dispersion (Variability)**
    - **Range**: Max - Min
    - **Variance**: Average squared deviation from mean
    - **Standard Deviation**: Square root of variance
    - **IQR (Interquartile Range)**: Q3 - Q1 (middle 50% of data)
    - **Coefficient of Variation**: SD / Mean (standardized measure of dispersion)
    
    **3. Distribution Shape**
    - **Skewness**: Asymmetry of distribution
      - Skewness = 0: Symmetric
      - Skewness > 0: Positively skewed (right tail)
      - Skewness < 0: Negatively skewed (left tail)
    - **Kurtosis**: Peakedness of distribution
      - Kurtosis = 3: Normal distribution
      - Kurtosis > 3: Leptokurtic (sharp peak, heavy tails)
      - Kurtosis < 3: Platykurtic (flat peak, light tails)
    
    **4. Normality Assessment**
    - **Shapiro-Wilk Test**: Tests if data follows normal distribution
    - **Anderson-Darling Test**: Another normality test
    - **Q-Q Plot**: Visual assessment of normality
    
    **5. Outlier Detection**
    - **IQR Method**: Values beyond Q1-1.5*IQR or Q3+1.5*IQR
    - **Z-Score Method**: Values with |z-score| > 3
    - **Visual Inspection**: Box plots and histograms
    
    **6. Frequency & Value Distribution**
    - **Value Counts**: Frequency of categorical values
    - **Percentiles**: Distribution across data range
    - **Unique Values**: Number of distinct values
    
    **Why is Univariate Analysis Important?**
    - Foundation for data cleaning
    - Understand individual variable behavior
    - Detect data quality issues
    - Choose appropriate transformations
    - Prepare data for modeling
    """)

st.markdown("---")

# Get column types
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Section 1: Numerical Variables Analysis
st.header("1️⃣ Numerical Variables Analysis")

selected_num = st.selectbox("Select a numerical variable:", numerical_cols, key="num_var")

if selected_num:
    col1, col2, col3 = st.columns(3)
    
    # Get clean data
    clean_data = df[selected_num].dropna().astype('float64')
    
    # Basic Statistics
    with col1:
        st.subheader("📊 Central Tendency")
        st.metric("Mean", f"{clean_data.mean():.2f}")
        st.metric("Median", f"{clean_data.median():.2f}")
        st.metric("Mode", f"{clean_data.mode()[0]:.2f}" if len(clean_data.mode()) > 0 else "N/A")
    
    with col2:
        st.subheader("📏 Dispersion")
        st.metric("Std Dev", f"{clean_data.std():.2f}")
        st.metric("Variance", f"{clean_data.var():.2f}")
        st.metric("Range", f"{clean_data.max() - clean_data.min():.2f}")
    
    with col3:
        st.subheader("📐 Shape")
        st.metric("Skewness", f"{skew(clean_data):.3f}")
        st.metric("Kurtosis", f"{kurtosis(clean_data):.3f}")
        st.metric("CV", f"{(clean_data.std() / clean_data.mean()):.3f}" if clean_data.mean() != 0 else "N/A")
    
    st.markdown("---")
    
    # Detailed Statistics Table
    st.subheader("📋 Detailed Statistical Summary")
    stats_df = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'Max', 'IQR'],
        'Value': [
            len(clean_data),
            f"{clean_data.mean():.2f}",
            f"{clean_data.median():.2f}",
            f"{clean_data.std():.2f}",
            f"{clean_data.min():.2f}",
            f"{clean_data.quantile(0.25):.2f}",
            f"{clean_data.quantile(0.50):.2f}",
            f"{clean_data.quantile(0.75):.2f}",
            f"{clean_data.max():.2f}",
            f"{clean_data.quantile(0.75) - clean_data.quantile(0.25):.2f}"
        ]
    })
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📈 Distribution Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Histogram with Normal Curve**")
        fig = px.histogram(data_frame=pd.DataFrame({selected_num: clean_data}), 
                          x=selected_num, nbins=50, 
                          title=f"Distribution of {selected_num}",
                          labels={selected_num: selected_num})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Box Plot (Outlier Detection)**")
        box_data = pd.DataFrame({selected_num: clean_data})
        fig = px.box(box_data, y=selected_num,
                    title=f"Box Plot of {selected_num}",
                    points="all")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Q-Q Plot (Normality Assessment)**")
        q_q_data = pd.DataFrame({
            'Theoretical': stats.probplot(clean_data)[0][0],
            'Sample': stats.probplot(clean_data)[0][1]
        })
        fig = px.scatter(q_q_data, x='Theoretical', y='Sample',
                        title="Q-Q Plot: Normal vs Actual Distribution",
                        labels={'Theoretical': 'Theoretical Quantiles', 
                               'Sample': 'Sample Quantiles'},
                        trendline="ols")
        # Add diagonal line
        fig.add_shape(type="line", x0=-3, y0=-3, x1=3, y1=3,
                     line=dict(color="red", width=2, dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Cumulative Distribution Function**")
        sorted_data = np.sort(clean_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        cdf_df = pd.DataFrame({'Value': sorted_data, 'CDF': cdf})
        fig = px.line(cdf_df, x='Value', y='CDF',
                     title="Cumulative Distribution Function",
                     labels={'Value': selected_num, 'CDF': 'Cumulative Probability'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Normality Tests
    st.subheader("🔬 Normality Tests")
    
    # Shapiro-Wilk Test
    if len(clean_data) <= 5000:  # Shapiro-Wilk works best with smaller samples
        stat_sw, p_sw = shapiro(clean_data)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shapiro-Wilk Test Statistic", f"{stat_sw:.4f}")
        with col2:
            st.metric("P-value", f"{p_sw:.6f}")
        
        if p_sw > 0.05:
            st.success("✅ Data appears to be normally distributed (p > 0.05)")
        else:
            st.warning("⚠️ Data does NOT appear to be normally distributed (p ≤ 0.05)")
    else:
        # Use Anderson-Darling for large samples
        stat_ad = normaltest(clean_data)[0]
        p_ad = normaltest(clean_data)[1]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anderson-Darling Test Statistic", f"{stat_ad:.4f}")
        with col2:
            st.metric("P-value", f"{p_ad:.6f}")
        
        if p_ad > 0.05:
            st.success("✅ Data appears to be normally distributed (p > 0.05)")
        else:
            st.warning("⚠️ Data does NOT appear to be normally distributed (p ≤ 0.05)")
    
    st.markdown("---")
    
    # Outlier Analysis
    st.subheader("🎯 Outlier Detection")
    
    Q1 = clean_data.quantile(0.25)
    Q3 = clean_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
    
    # Z-score method
    z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
    outliers_zscore = clean_data[z_scores > 3]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("IQR Method - Outliers", len(outliers_iqr))
        st.write(f"Lower Bound: {lower_bound:.2f}")
        st.write(f"Upper Bound: {upper_bound:.2f}")
    
    with col2:
        st.metric("Z-Score Method (|z| > 3)", len(outliers_zscore))
        st.write(f"Mean: {clean_data.mean():.2f}")
        st.write(f"Std Dev: {clean_data.std():.2f}")
    
    with col3:
        st.metric("Outliers %", f"{len(outliers_iqr)/len(clean_data)*100:.2f}%")
        st.metric("Non-Outliers %", f"{(len(clean_data)-len(outliers_iqr))/len(clean_data)*100:.2f}%")
    
    # Show outlier visualization
    st.write("**Outliers Visualization** - Red zone shows outlier range")
    outlier_plot = pd.DataFrame({selected_num: clean_data})
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean_data, name=selected_num, nbinsx=50, opacity=0.7))
    fig.add_vline(x=lower_bound, line_dash="dash", line_color="red",
                 annotation_text=f"Lower Bound: {lower_bound:.2f}", annotation_position="top left")
    fig.add_vline(x=upper_bound, line_dash="dash", line_color="red",
                 annotation_text=f"Upper Bound: {upper_bound:.2f}", annotation_position="top right")
    fig.update_layout(title=f"Distribution with Outlier Bounds - {selected_num}",
                     xaxis_title=selected_num, yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Section 2: Categorical Variables Analysis
st.header("2️⃣ Categorical Variables Analysis")

selected_cat = st.selectbox("Select a categorical variable:", categorical_cols, key="cat_var")

if selected_cat:
    st.subheader(f"Analysis of '{selected_cat}'")
    
    value_counts = df[selected_cat].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unique Values", df[selected_cat].nunique())
    with col2:
        st.metric("Most Frequent", value_counts.index[0])
    with col3:
        st.metric("Frequency of Mode", value_counts.values[0])
    with col4:
        st.metric("Missing Values", df[selected_cat].isnull().sum())
    
    st.markdown("---")
    
    # Value Counts Table
    st.subheader("📋 Frequency Distribution")
    
    freq_df = pd.DataFrame({
        'Category': value_counts.index,
        'Count': value_counts.values,
        'Percentage': (value_counts.values / len(df) * 100).round(2),
        'Cumulative %': (value_counts.values.cumsum() / len(df) * 100).round(2)
    })
    
    st.dataframe(freq_df, use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Categorical Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Bar Chart - Frequencies**")
        fig = px.bar(freq_df, x='Category', y='Count',
                    title=f"Frequency Distribution of {selected_cat}",
                    labels={'Category': selected_cat, 'Count': 'Frequency'},
                    text='Count')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Pie Chart - Proportions**")
        fig = px.pie(freq_df, values='Count', names='Category',
                    title=f"Proportion Distribution of {selected_cat}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Cumulative percentage chart
    st.write("**Cumulative Percentage Chart (Pareto Analysis)**")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=freq_df['Category'], y=freq_df['Percentage'],
                        name='Percentage', yaxis='y1',
                        marker=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=freq_df['Category'], y=freq_df['Cumulative %'],
                            name='Cumulative %', yaxis='y2',
                            mode='lines+markers', line=dict(color='red', width=3)))
    fig.update_layout(
        title=f"Pareto Analysis: {selected_cat}",
        yaxis=dict(title="Percentage (%)", side='left'),
        yaxis2=dict(title="Cumulative % (%)", overlaying='y', side='right'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Diversity Metrics
    st.subheader("📈 Categorical Diversity Metrics")
    
    # Shannon Entropy (Diversity index)
    total = len(df)
    proportions = value_counts.values / total
    entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
    max_entropy = np.log2(len(value_counts))
    normalized_entropy = entropy / max_entropy
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Shannon Entropy", f"{entropy:.3f}")
        st.caption("Measure of diversity (higher = more diverse)")
    
    with col2:
        st.metric("Max Possible Entropy", f"{max_entropy:.3f}")
        st.caption("Entropy if all categories equally likely")
    
    with col3:
        st.metric("Normalized Entropy", f"{normalized_entropy:.3f}")
        st.caption("0-1 scale (1 = maximum diversity)")
    
    # Dominance
    dominance = value_counts.values[0] / total
    st.metric("Dominance (Mode %)", f"{dominance*100:.2f}%")
    st.caption(f"The most frequent category ({value_counts.index[0]}) represents {dominance*100:.2f}% of data")

st.markdown("---")

# Section 3: Key Insights
st.header("3️⃣ Key Insights & Interpretation")

st.markdown("""
### What to Look For in Univariate Analysis:

**For Numerical Variables:**
- **Skewness**: 
  - Near 0: Symmetric data (good for many models)
  - > 0: Right-skewed (may need transformation)
  - < 0: Left-skewed (may need transformation)
  
- **Outliers**: 
  - Few outliers: Use as-is or investigate
  - Many outliers: May indicate data quality issues or special populations
  
- **Normality**:
  - Normal distribution: Can use parametric tests
  - Non-normal: May need transformation or non-parametric tests

**For Categorical Variables:**
- **Imbalance**: 
  - Balanced categories: Good for classification
  - Imbalanced: May need sampling or reweighting
  
- **Dominance**:
  - One category dominates: May not be informative
  - Multiple categories: Better predictive potential

### Next Steps:
1. **Data Cleaning**: Handle outliers and missing values
2. **Transformation**: Apply log, sqrt, or box-cox to normalize skewed data
3. **Scaling**: Standardize or normalize for certain algorithms
4. **Feature Engineering**: Create new features based on insights
5. **Bivariate Analysis**: Examine relationships between variables
""")

st.markdown("---")

# Summary Statistics Export
st.header("4️⃣ Export Summary Statistics")

with st.expander("📥 Download Complete Statistical Summary"):
    
    # Create comprehensive summary
    summary_data = []
    
    for col in numerical_cols:
        clean = df[col].dropna()
        summary_data.append({
            'Variable': col,
            'Type': 'Numerical',
            'Count': len(clean),
            'Mean': f"{clean.mean():.2f}",
            'Median': f"{clean.median():.2f}",
            'Std Dev': f"{clean.std():.2f}",
            'Min': f"{clean.min():.2f}",
            'Max': f"{clean.max():.2f}",
            'Skewness': f"{skew(clean):.3f}",
            'Kurtosis': f"{kurtosis(clean):.3f}"
        })
    
    for col in categorical_cols:
        summary_data.append({
            'Variable': col,
            'Type': 'Categorical',
            'Unique': df[col].nunique(),
            'Most Frequent': df[col].value_counts().index[0],
            'Mode Frequency': df[col].value_counts().values[0],
            'Missing': df[col].isnull().sum(),
            'Missing %': f"{df[col].isnull().sum()/len(df)*100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display
    st.dataframe(summary_df, use_container_width=True)
    
    # Download option
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Summary Statistics (CSV)",
        data=csv,
        file_name="univariate_analysis_summary.csv",
        mime="text/csv"
    )
