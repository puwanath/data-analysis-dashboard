import streamlit as st
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.cache_manager import CacheManager
from src.security_manager import SecurityManager
from typing import Dict, List, Optional, Tuple
import json
import yaml
import os
from pathlib import Path
import logging
from datetime import datetime

# Initialize components
logger = logging.getLogger(__name__)
data_processor = DataProcessor()
cache_manager = CacheManager()
security_manager = SecurityManager()

def load_file_config() -> Dict:
    """Load file configuration"""
    try:
        with open("config/data_sources.yaml", 'r') as f:
            config = yaml.safe_load(f)
            return config['storage']['local']
    except Exception as e:
        logger.error(f"Error loading file config: {str(e)}")
        return {
            'allowed_extensions': ['.csv', '.xlsx', '.xls', '.json', '.parquet'],
            'max_file_size': '100mb'
        }

def validate_file(file, config: Dict) -> Tuple[bool, str]:
    """Validate uploaded file"""
    try:
        # Check file extension
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in config['allowed_extensions']:
            return False, f"Unsupported file type. Allowed types: {', '.join(config['allowed_extensions'])}"
        
        # Check file size
        max_size = int(config['max_file_size'].replace('mb', '')) * 1024 * 1024
        if file.size > max_size:
            return False, f"File too large. Maximum size: {config['max_file_size']}"
        
        return True, "File valid"
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return False, "Error validating file"

def process_uploaded_file(file) -> Optional[pd.DataFrame]:
    """Process uploaded file and return DataFrame"""
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        
        if file_ext == '.csv':
            df = data_processor.load_csv(file)
        elif file_ext in ['.xlsx', '.xls']:
            df = data_processor.load_excel(file)
        elif file_ext == '.json':
            df = data_processor.load_json(file)
        elif file_ext == '.parquet':
            df = data_processor.load_parquet(file)
        else:
            st.error("Unsupported file type")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {str(e)}")
        return None

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """Analyze data quality metrics"""
    try:
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'column_types': df.dtypes.astype(str).to_dict()
        }
        
        # Calculate unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        metrics['unique_values'] = {
            col: df[col].nunique()
            for col in categorical_cols
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing data quality: {str(e)}")
        return {}

def detect_relationships(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
    """Detect potential relationships between datasets"""
    try:
        relationships = {}
        for name1, df1 in dfs.items():
            relationships[name1] = []
            for name2, df2 in dfs.items():
                if name1 != name2:
                    common_cols = set(df1.columns) & set(df2.columns)
                    if common_cols:
                        rel_strength = analyze_relationship_strength(
                            df1, df2, list(common_cols)
                        )
                        relationships[name1].append({
                            'dataset': name2,
                            'common_columns': list(common_cols),
                            'relationship_strength': rel_strength
                        })
        return relationships
        
    except Exception as e:
        logger.error(f"Error detecting relationships: {str(e)}")
        return {}

def analyze_relationship_strength(df1: pd.DataFrame, 
                                df2: pd.DataFrame,
                                common_cols: List[str]) -> float:
    """Analyze strength of relationship between datasets"""
    try:
        # Calculate overlap in values
        overlap_scores = []
        for col in common_cols:
            set1 = set(df1[col].dropna().astype(str))
            set2 = set(df2[col].dropna().astype(str))
            if set1 and set2:
                overlap = len(set1 & set2) / len(set1 | set2)
                overlap_scores.append(overlap)
        
        return np.mean(overlap_scores) if overlap_scores else 0.0
        
    except Exception as e:
        logger.error(f"Error analyzing relationship strength: {str(e)}")
        return 0.0

def show_data_preview(df: pd.DataFrame):
    """Show data preview with tabs for different views"""
    tabs = st.tabs(["Preview", "Info", "Statistics"])
    
    with tabs[0]:
        st.dataframe(df.head(100))
        
    with tabs[1]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
            
        st.write("### Column Information")
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info)
        
    with tabs[2]:
        st.write("### Numeric Statistics")
        st.dataframe(df.describe())
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            st.write("### Category Statistics")
            for col in categorical_cols:
                st.write(f"**Value Counts: {col}**")
                st.dataframe(df[col].value_counts().head(10))

def manage_uploaded_files():
    """Manage previously uploaded files"""
    if st.session_state.uploaded_files:
        st.write("### Uploaded Files")
        
        for filename in list(st.session_state.uploaded_files.keys()):
            with st.expander(f"📄 {filename}"):
                file_info = st.session_state.uploaded_files[filename]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"Rows: {file_info['shape'][0]}")
                with col2:
                    st.write(f"Columns: {file_info['shape'][1]}")
                with col3:
                    if st.button("Remove", key=f"remove_{filename}"):
                        del st.session_state.uploaded_files[filename]
                        st.rerun()
                with col4:
                    if st.button("Preview", key=f"preview_{filename}"):
                        st.dataframe(file_info['data'].head())
                        
def handle_file_upload():
    """Handle file upload process"""
    config = load_file_config()
    
    uploaded_files = st.file_uploader(
        "Upload Data Files",
        type=[ext.replace('.', '') for ext in config['allowed_extensions']],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                # Validate file
                is_valid, message = validate_file(file, config)
                if not is_valid:
                    st.error(message)
                    continue
                
                with st.spinner(f'Processing {file.name}...'):
                    # Process file
                    df = process_uploaded_file(file)
                    if df is not None:
                        # Store in session state
                        st.session_state.uploaded_files[file.name] = {
                            'data': df,
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'upload_time': datetime.now().isoformat(),
                            'quality_metrics': analyze_data_quality(df)
                        }
                        st.success(f"Successfully processed {file.name}")
                        
                        # Show preview
                        with st.expander("Show Data Preview"):
                            show_data_preview(df)

def handle_data_relationships():
    """Handle data relationship detection and management"""
    if len(st.session_state.uploaded_files) > 1:
        st.write("### Data Relationships")
        
        relationships = detect_relationships(
            {name: info['data'] 
             for name, info in st.session_state.uploaded_files.items()}
        )
        
        for file1, related in relationships.items():
            if related:
                with st.expander(f"🔗 Relationships for {file1}"):
                    for rel in related:
                        st.write(f"Connected to: {rel['dataset']}")
                        st.write(f"Common columns: {', '.join(rel['common_columns'])}")
                        st.write(f"Relationship strength: {rel['relationship_strength']:.2f}")
                        
                        if st.button(
                            "Merge Datasets",
                            key=f"merge_{file1}_{rel['dataset']}"
                        ):
                            merge_datasets(file1, rel['dataset'], rel['common_columns'])

def merge_datasets(file1: str, file2: str, common_cols: List[str]):
    """Merge two datasets"""
    try:
        merge_type = st.selectbox(
            "Select merge type:",
            ['inner', 'outer', 'left', 'right']
        )
        
        if st.button("Confirm Merge"):
            df1 = st.session_state.uploaded_files[file1]['data']
            df2 = st.session_state.uploaded_files[file2]['data']
            
            merged_df = data_processor.merge_datasets(
                df1, df2, common_cols, merge_type
            )
            
            if merged_df is not None:
                new_filename = f"merged_{file1}_{file2}.csv"
                st.session_state.uploaded_files[new_filename] = {
                    'data': merged_df,
                    'shape': merged_df.shape,
                    'columns': list(merged_df.columns),
                    'upload_time': datetime.now().isoformat(),
                    'quality_metrics': analyze_data_quality(merged_df)
                }
                st.success("Datasets merged successfully!")
                st.rerun()
                
    except Exception as e:
        st.error(f"Error merging datasets: {str(e)}")
        logger.error(f"Dataset merge error: {str(e)}")

def show_page():
    """Main function to show data upload page"""

    # Initialize session state for uploaded files if it doesn't exist
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}

    st.write("## 📤 Data Upload and Management")
    
    # File upload section
    st.write("### Upload New Files")
    handle_file_upload()
    
    # Manage existing files
    manage_uploaded_files()
    
    # Handle data relationships
    handle_data_relationships()

if __name__ == "__main__":
    show_page()