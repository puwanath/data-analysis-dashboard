import streamlit as st
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.visualization import Visualizer
from typing import Dict, List, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
import logging
from datetime import datetime

# Initialize components
logger = logging.getLogger(__name__)
data_processor = DataProcessor()
visualizer = Visualizer()

def show_data_statistics(df: pd.DataFrame):
    """Show comprehensive data statistics"""
    st.write("### 📊 Data Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    with col4:
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Missing Values", f"{missing_percentage:.1f}%")

    # For the detailed statistics section:
    st.write("📈 Detailed Statistics")
    tab_numeric, tab_categorical, tab_temporal = st.tabs(["Numeric", "Categorical", "Temporal"])
    
    with tab_numeric:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            st.write("Numeric Columns Statistics:")
            st.dataframe(df[numeric_cols].describe())
            
            for col in numeric_cols:
                st.write(f"Distribution: {col}")
                fig = px.histogram(df, x=col, marginal="box")
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical tests
                skewness = stats.skew(df[col].dropna())
                kurtosis = stats.kurtosis(df[col].dropna())
                normality = stats.normaltest(df[col].dropna())
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Skewness", f"{skewness:.3f}")
                col2.metric("Kurtosis", f"{kurtosis:.3f}")
                col3.metric("Normality p-value", f"{normality.pvalue:.3f}")

    # Detailed statistics
    # with st.expander("📈 Detailed Statistics", expanded=True):
    #     tabs = st.tabs(["Numeric", "Categorical", "Temporal"])
        
    #     with tabs[0]:
    #         numeric_cols = df.select_dtypes(include=[np.number]).columns
    #         if not numeric_cols.empty:
    #             st.write("Numeric Columns Statistics:")
    #             st.dataframe(df[numeric_cols].describe())
                
    #             st.write("Distribution Analysis:")
    #             for col in numeric_cols:
    #                 st.write(f"### Distribution: {col}")
    #                 fig = px.histogram(df, x=col, marginal="box")
    #                 st.plotly_chart(fig, use_container_width=True)
                    
    #                 # Add statistical tests
    #                 skewness = stats.skew(df[col].dropna())
    #                 kurtosis = stats.kurtosis(df[col].dropna())
    #                 normality = stats.normaltest(df[col].dropna())
                    
    #                 st.write(f"Skewness: {skewness:.3f}")
    #                 st.write(f"Kurtosis: {kurtosis:.3f}")
    #                 st.write(f"Normality test p-value: {normality.pvalue:.3f}")
            
    #     with tabs[1]:
    #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    #         if not categorical_cols.empty:
    #             st.write("Categorical Columns Analysis:")
    #             for col in categorical_cols:
    #                 with st.expander(f"Analysis: {col}"):
    #                     value_counts = df[col].value_counts()
    #                     unique_count = df[col].nunique()
                        
    #                     st.write(f"Unique values: {unique_count}")
                        
    #                     # Bar chart of value counts
    #                     fig = px.bar(
    #                         x=value_counts.index[:20],
    #                         y=value_counts.values[:20],
    #                         title=f"Top 20 values in {col}"
    #                     )
    #                     st.plotly_chart(fig, use_container_width=True)
                        
    #                     # Show full value counts
    #                     st.dataframe(pd.DataFrame({
    #                         'Value': value_counts.index,
    #                         'Count': value_counts.values,
    #                         'Percentage': (value_counts.values / len(df) * 100).round(2)
    #                     }))
        
    #     with tabs[2]:
    #         datetime_cols = df.select_dtypes(include=['datetime64']).columns
    #         if not datetime_cols.empty:
    #             st.write("Temporal Analysis:")
    #             for col in datetime_cols:
    #                 with st.expander(f"Analysis: {col}"):
    #                     temporal_stats = pd.DataFrame({
    #                         'Metric': ['Min', 'Max', 'Range', 'Unique Values'],
    #                         'Value': [
    #                             df[col].min(),
    #                             df[col].max(),
    #                             df[col].max() - df[col].min(),
    #                             df[col].nunique()
    #                         ]
    #                     })
    #                     st.dataframe(temporal_stats)
                        
    #                     # Timeline visualization
    #                     fig = px.scatter(
    #                         df,
    #                         x=col,
    #                         title=f"Timeline of {col}"
    #                     )
    #                     st.plotly_chart(fig, use_container_width=True)

def analyze_correlations(df: pd.DataFrame):
    """Analyze and visualize correlations between variables"""
    st.write("### 🔗 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
        return
        
    # Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Heatmap
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        labels=dict(color="Correlation")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Strong correlations
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.5:
                strong_corr.append({
                    'Variables': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                    'Correlation': corr_matrix.iloc[i,j]
                })
    
    if strong_corr:
        st.write("### Strong Correlations Found")
        st.dataframe(pd.DataFrame(strong_corr))
        
        # Scatter plots for strong correlations
        for corr in strong_corr:
            var1, var2 = corr['Variables'].split(' vs ')
            fig = px.scatter(
                df,
                x=var1,
                y=var2,
                title=f"Scatter plot: {var1} vs {var2} (correlation: {corr['Correlation']:.2f})",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)

def analyze_distributions(df: pd.DataFrame):
    """Analyze variable distributions"""
    st.write("### 📊 Distribution Analysis")
    
    # Select columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if numeric_cols.empty:
        st.warning("No numeric columns available for distribution analysis")
        return
        
    selected_cols = st.multiselect(
        "Select columns for distribution analysis",
        numeric_cols
    )
    
    if selected_cols:
        # Distribution plots
        for col in selected_cols:
            st.write(f"#### Distribution of {col}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    df,
                    x=col,
                    title=f"Histogram of {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Box plot
                fig = px.box(
                    df,
                    y=col,
                    title=f"Box plot of {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            st.write("Statistical Tests:")
            
            # Normality test
            stat, p_value = stats.normaltest(df[col].dropna())
            st.write(f"Normality test p-value: {p_value:.4f}")
            
            # Basic statistics
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
                'Value': [
                    df[col].mean(),
                    df[col].median(),
                    df[col].std(),
                    stats.skew(df[col].dropna()),
                    stats.kurtosis(df[col].dropna())
                ]
            })
            st.dataframe(stats_df)

def analyze_outliers(df: pd.DataFrame):
    """Analyze outliers in the dataset"""
    st.write("### 🎯 Outlier Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        st.warning("No numeric columns available for outlier analysis")
        return
        
    selected_col = st.selectbox(
        "Select column for outlier analysis",
        numeric_cols
    )
    
    if selected_col:
        # Calculate outlier boundaries
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df[
            (df[selected_col] < lower_bound) |
            (df[selected_col] > upper_bound)
        ]
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Outliers", len(outliers))
        with col2:
            st.metric("Lower Bound", f"{lower_bound:.2f}")
        with col3:
            st.metric("Upper Bound", f"{upper_bound:.2f}")
        
        # Box plot with outliers
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df[selected_col],
            name=selected_col,
            boxpoints='outliers'
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Show outlier values
        if not outliers.empty:
            st.write("### Outlier Values")
            st.dataframe(outliers[[selected_col]])

def analyze_missing_values(df: pd.DataFrame):
    """Analyze missing values in the dataset"""
    st.write("### ❓ Missing Values Analysis")
    
    # Calculate missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing Percentage': missing_percent.values
    })
    
    # Filter columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if missing_df.empty:
        st.success("No missing values found in the dataset!")
        return
        
    # Display missing values statistics
    st.dataframe(missing_df)
    
    # Visualization of missing values
    fig = px.bar(
        missing_df,
        x='Column',
        y='Missing Percentage',
        title="Missing Values by Column"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values patterns
    if len(missing_df) > 1:
        st.write("### Missing Values Patterns")
        # Create missing value pattern matrix
        pattern_matrix = df[missing_df['Column']].isnull()
        patterns = pattern_matrix.value_counts()
        
        st.write(f"Found {len(patterns)} different missing value patterns")
        st.dataframe(pd.DataFrame({
            'Pattern': [str(p) for p in patterns.index],
            'Count': patterns.values
        }))

def show_page():
    """Main function to show data explorer page"""
    st.write("## 🔍 Data Explorer")
    
    if not st.session_state.uploaded_files:
        st.warning("Please upload some data files first!")
        return
    
    # Select dataset
    selected_file = st.selectbox(
        "Select dataset to explore:",
        list(st.session_state.uploaded_files.keys())
    )
    
    if selected_file:
        df = st.session_state.uploaded_files[selected_file]['data']
        
        # Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Statistics",
            "Correlations",
            "Distributions",
            "Outliers",
            "Missing Values"
        ])
        
        with tab1:
            show_data_statistics(df)
            
        with tab2:
            analyze_correlations(df)
            
        with tab3:
            analyze_distributions(df)
            
        with tab4:
            analyze_outliers(df)
            
        with tab5:
            analyze_missing_values(df)

if __name__ == "__main__":
    show_page()