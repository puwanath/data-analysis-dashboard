import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st
import requests
from pandas.api.types import is_numeric_dtype

class DataProcessor:
    @staticmethod
    def load_file(file, file_type: str) -> pd.DataFrame:
        """Load data from uploaded file"""
        try:
            if file_type in ['xlsx', 'xls']:
                return pd.read_excel(file)
            elif file_type == 'csv':
                return pd.read_csv(file)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    @staticmethod
    def analyze_relationships(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Analyze potential relationships between datasets"""
        relationships = {}
        for name1, df1 in dfs.items():
            relationships[name1] = []
            for name2, df2 in dfs.items():
                if name1 != name2:
                    common_cols = set(df1.columns) & set(df2.columns)
                    if common_cols:
                        relationships[name1].append({
                            'dataset': name2,
                            'common_columns': list(common_cols)
                        })
        return relationships

    @staticmethod
    def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, 
                      merge_columns: List[str], 
                      merge_type: str = 'inner') -> pd.DataFrame:
        """Merge two datasets based on specified columns"""
        try:
            return pd.merge(df1, df2, on=merge_columns, how=merge_type)
        except Exception as e:
            st.error(f"Error merging datasets: {str(e)}")
            return None

    @staticmethod
    def group_data(df: pd.DataFrame, 
                   group_columns: List[str], 
                   agg_functions: Dict[str, str]) -> pd.DataFrame:
        """Group data based on specified columns and aggregation functions"""
        try:
            return df.groupby(group_columns).agg(agg_functions).reset_index()
        except Exception as e:
            st.error(f"Error grouping data: {str(e)}")
            return None

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            return df[numeric_cols].corr()
        return None

    @staticmethod
    def detect_data_type(df: pd.DataFrame) -> Dict[str, str]:
        """Detect data types for each column"""
        data_types = {}
        for column in df.columns:
            if is_numeric_dtype(df[column]):
                if df[column].nunique() < 10:
                    data_types[column] = 'categorical_numeric'
                else:
                    data_types[column] = 'continuous'
            else:
                if df[column].nunique() < 10:
                    data_types[column] = 'categorical'
                else:
                    data_types[column] = 'text'
        return data_types

    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get comprehensive summary statistics"""
        summary = {
            'numeric': df.describe(),
            'missing': df.isnull().sum(),
            'unique_counts': df.nunique(),
            'data_types': df.dtypes
        }
        return summary

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by handling missing values and duplicates"""
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric columns with median
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill categorical columns with mode
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            
            return df
        except Exception as e:
            st.error(f"Error cleaning data: {str(e)}")
            return None

    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, List[int]]:
        """Detect outliers in specified columns using IQR or Z-score method"""
        outliers = {}
        for col in columns:
            if not is_numeric_dtype(df[col]):
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_indices = df[
                    (df[col] < (Q1 - 1.5 * IQR)) | 
                    (df[col] > (Q3 + 1.5 * IQR))
                ].index.tolist()
            else:  # z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_indices = df[z_scores > 3].index.tolist()
                
            outliers[col] = outlier_indices
            
        return outliers

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing columns"""
        df_new = df.copy()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create interaction features for numeric columns
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                df_new[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                df_new[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                
        return df_new
    
    def load_excel(self, file) -> pd.DataFrame:
        """Load data from Excel file"""
        return pd.read_excel(file)
    
    def load_csv(self, file) -> pd.DataFrame:
        """Load data from CSV file"""
        return pd.read_csv(file)

    def load_json(self, file) -> pd.DataFrame:
        """Load data from JSON file"""
        return pd.read_json(file)

    def load_sql(self, query) -> pd.DataFrame:
        """Load data from SQL query"""
        return pd.read_sql(query, self.engine)

    def load_api(self, url) -> pd.DataFrame:
        """Load data from API"""
        response = requests.get(url)
        return pd.DataFrame(response.json())

    def load_web(self, url) -> pd.DataFrame:
        """Load data from web page"""
        response = requests.get(url)
        return pd.read_html(response.text)[0]