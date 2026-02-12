import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# إعداد الصفحة
st.set_page_config(page_title="Bivariate Analysis", layout="wide")

# دالة تحميل البيانات
@st.cache_data
def load_data():
    # بناء المسار للوصول لملف الـ CSV في المجلد الرئيسي
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Airbnb NYC 2019.csv')
    try:
        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # تحويل أنواع البيانات لضمان التوافق مع Plotly
        for col in df.columns:
            if str(df[col].dtype).startswith('Int'):
                df[col] = df[col].astype('float64')
        return df
    except FileNotFoundError:
        st.error(f"لم يتم العثور على ملف البيانات في المسار: {csv_path}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.title("🔗 Bivariate Analysis Phase")
    st.markdown("---")

    with st.expander("ℹ️ What we do in the Bivariate Phase", expanded=False):
        st.markdown(
            """
            تحليل المتغيرات الثنائية (Bivariate Analysis) يهدف لفهم العلاقة بين متغيرين:
            - **Numeric vs Numeric**: قياس الارتباط (Correlation) وتشتت البيانات.
            - **Numeric vs Categorical**: مقارنة التوزيعات (Box plots) وحساب المتوسطات لكل فئة.
            - **Categorical vs Categorical**: تحليل التكرارات والنسب المئوية بين الفئات المختلفة.
            """
        )

    # تجهيز قوائم الأعمدة
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- القسم الأول: Numeric vs Numeric ---
    st.header("1️⃣ Numeric vs Numeric")
    col1, col2 = st.columns([1, 2])

    with col1:
        x_var = st.selectbox("X variable (numeric)", numerical_cols, key="biv_x_num")
        y_var = st.selectbox("Y variable (numeric)", numerical_cols, index=1 if len(numerical_cols) > 1 else 0, key="biv_y_num")
        corr_method = st.selectbox("Correlation method", ['pearson', 'spearman'], key="biv_corr")
        show_trend = st.checkbox("Show regression trendline", value=True, key="biv_trend")
        
        if st.button("Compute correlation", key="btn_corr"):
            pair = df[[x_var, y_var]].dropna()
            if len(pair) < 2:
                st.warning("بيانات غير كافية لحساب الارتباط.")
            else:
                corr = pair.corr(method=corr_method).iloc[0,1]
                st.metric(f"{corr_method.title()} correlation", f"{corr:.3f}")

    with col2:
        pair_plot_data = df[[x_var, y_var]].dropna().copy()
        fig_scatter = px.scatter(pair_plot_data, x=x_var, y=y_var, trendline='ols' if show_trend else None,
                         title=f"Scatter Plot: {x_var} vs {y_var}")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # --- القسم الثاني: Numeric vs Categorical ---
    st.header("2️⃣ Numeric vs Categorical")
    
    # فلترة الأعمدة الفئوية لاستبعاد الأعمدة التي تحتوي على قيم فريدة كثيرة جداً (مثل ID أو Name)
    # هذا يمنع ازدحام الرسم البياني
    filtered_cat_biv = [col for col in categorical_cols if df[col].nunique() < 50]

    col3, col4 = st.columns(2)
    with col3:
        num_var = st.selectbox("Select Numeric Variable", numerical_cols, key="num_select_v2")
        cat_var = st.selectbox("Select Categorical Variable", filtered_cat_biv, key="cat_select_v2")
        agg_func = st.selectbox("Choose Aggregation", ['mean', 'median', 'count'], key="agg_select_v2")

    with col4:
        plot_df = df[[num_var, cat_var]].dropna().copy()
        if plot_df.empty:
            st.warning("لا توجد بيانات متاحة لهذا الزوج.")
        else:
            fig_box = px.box(plot_df, x=cat_var, y=num_var, title=f"Box Plot: {num_var} by {cat_var}")
            st.plotly_chart(fig_box, use_container_width=True)

    if st.button("Show Grouped Summary Table", key="btn_summary"):
        grp = df.groupby(cat_var)[num_var].agg(agg_func).sort_values(ascending=False)
        st.dataframe(grp.reset_index(), use_container_width=True)

    st.markdown("---")

    # --- القسم الثالث: Categorical vs Categorical ---
    st.header("3️⃣ Categorical vs Categorical")
    
    # فلترة الأعمدة (أهم خطوة لحل مشكلة الـ Memory Error)
    # سنختار فقط الأعمدة التي تحتوي على أقل من 30 قيمة فريدة
    short_cat_cols = [col for col in categorical_cols if df[col].nunique() < 30]

    if len(short_cat_cols) < 2:
        st.warning("لا توجد أعمدة فئوية كافية (بقيم فريدة قليلة) لإجراء هذا التحليل.")
    else:
        col5, col6 = st.columns(2)
        with col5:
            cat_a = st.selectbox("Category A", short_cat_cols, key="cat_a_final")
            cat_b = st.selectbox("Category B", [c for c in short_cat_cols if c != cat_a], key="cat_b_final")

        with col6:
            # حساب جدول التكرار
            ct = pd.crosstab(df[cat_a], df[cat_b], normalize='index')
            st.subheader("Contingency Table (Proportions)")
            st.dataframe(ct.round(3), use_container_width=True)
            
            ct_plot = pd.crosstab(df[cat_a], df[cat_b])
            fig_stack = px.bar(ct_plot, barmode='stack', title=f"Stacked Counts: {cat_a} by {cat_b}")
            st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")

    # --- القسم الرابع: Correlation Matrix ---
    st.header("4️⃣ Correlation Matrix & Heatmap")
    if st.button("Generate Heatmap", key="btn_heat"):
        num_df = df[numerical_cols].dropna()
        if not num_df.empty:
            corr_matrix = num_df.corr().fillna(0)
            fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.error("لا توجد بيانات رقمية كافية.")

    # خيار التصدير
    with st.expander("📥 Export Current Analysis"):
        export_df = df[[x_var, y_var]].dropna().head(1000)
        csv = export_df.to_csv(index=False)
        st.download_button("Download Sample CSV", data=csv, file_name="bivariate_data.csv", mime='text/csv', key="btn_dl")

else:
    st.warning("يرجى التأكد من وجود ملف Airbnb NYC 2019.csv في المجلد الصحيح.")