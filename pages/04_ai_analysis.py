import streamlit as st
import pandas as pd
import numpy as np
from src.llm_analyzer import LLMAnalyzer
from src.data_processor import DataProcessor
from src.visualization import Visualizer
from typing import Dict, List, Optional, Any
import plotly.express as px
import logging
from datetime import datetime
import yaml
import json

# Initialize components
logger = logging.getLogger(__name__)
llm_analyzer = LLMAnalyzer()
data_processor = DataProcessor()
visualizer = Visualizer()

def load_prompt_templates() -> Dict[str, str]:
    """Load prompt templates from configuration"""
    try:
        with open("config/models.yaml", 'r') as f:
            config = yaml.safe_load(f)
            return config.get('prompts', {})
    except Exception as e:
        logger.error(f"Error loading prompt templates: {str(e)}")
        return {}

def prepare_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare dataset information for analysis"""
    try:
        info = {
            'basic_info': {
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'memory_usage': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
                'column_types': {k: str(v) for k, v in df.dtypes.astype(str).to_dict().items()}
            },
            'statistics': {
                'numeric': {col: {k: float(v) if isinstance(v, (np.integer, np.floating)) else str(v) 
                           for k, v in stats.items()}
                           for col, stats in df.describe().to_dict().items()},
                'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
                'unique_counts': {k: int(v) for k, v in df.nunique().to_dict().items()}
            }
        }
        
        # Add column-specific information
        info['columns'] = {}
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            if np.issubdtype(df[col].dtype, np.number):
                info['columns'][col] = {
                    'type': 'numeric',
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                info['columns'][col] = {
                    'type': 'datetime',
                    'min': str(df[col].min()),
                    'max': str(df[col].max()),
                    'unique_values': int(df[col].nunique())
                }
            else:
                info['columns'][col] = {
                    'type': 'categorical',
                    'unique_values': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in df[col].value_counts().head(5).to_dict().items()}
                }
                
        return info
        
    except Exception as e:
        logger.error(f"Error preparing dataset info: {str(e)}")
        return {}
    
def show_overview_analysis(df: pd.DataFrame, model: str):
    """Show overview analysis of the dataset"""
    st.write("### 📊 Data Overview Analysis")
    
    templates = load_prompt_templates()
    
    if not templates:
        st.error("Error loading prompt templates")
        return
        
    # Prepare dataset information
    dataset_info = prepare_dataset_info(df)
    
    # Create analysis prompt
    prompt = templates['data_overview'].format(
        dataset_info=json.dumps(dataset_info, indent=2)
    )
    
    with st.spinner("Analyzing dataset..."):
        analysis = llm_analyzer.query_model(prompt, model=model)
        if analysis:
            st.markdown(analysis)
            
            # Save analysis
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
                
            st.session_state.analysis_history.append({
                'type': 'overview',
                'timestamp': datetime.now().isoformat(),
                'model': model,
                'result': analysis
            })

def show_relationship_analysis(df: pd.DataFrame, model: str):
    """Show relationship analysis between variables"""
    st.write("### 🔗 Relationship Analysis")
    
    # Select columns for analysis
    cols = st.multiselect(
        "Select columns to analyze relationships:",
        df.columns
    )
    
    if cols:
        # Prepare relationship information
        relationship_info = {}
        
        # Numeric correlations
        numeric_cols = [col for col in cols 
                       if np.issubdtype(df[col].dtype, np.number)]
        if len(numeric_cols) > 1:
            relationship_info['correlations'] = df[numeric_cols].corr().to_dict()
        
        # Categorical associations
        categorical_cols = [col for col in cols 
                          if col not in numeric_cols]
        if categorical_cols:
            relationship_info['categorical_associations'] = {}
            for col1 in categorical_cols:
                for col2 in categorical_cols:
                    if col1 < col2:
                        contingency = pd.crosstab(df[col1], df[col2])
                        relationship_info['categorical_associations'][
                            f"{col1}_vs_{col2}"
                        ] = contingency.to_dict()
        
        # Create analysis prompt
        templates = load_prompt_templates()
        prompt = templates['relationships'].format(
            relationships_info=json.dumps(relationship_info, indent=2)
        )
        
        if st.button("Analyze Relationships"):
            with st.spinner("Analyzing relationships..."):
                analysis = llm_analyzer.query_model(prompt, model=model)
                if analysis:
                    st.markdown(analysis)
                    
                    # Save analysis
                    if 'analysis_history' not in st.session_state:
                        st.session_state.analysis_history = []
                        
                    st.session_state.analysis_history.append({
                        'type': 'relationships',
                        'timestamp': datetime.now().isoformat(),
                        'model': model,
                        'result': analysis
                    })

def show_time_series_analysis(df: pd.DataFrame, model: str):
    """Show time series analysis"""
    st.write("### 📈 Time Series Analysis")
    
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
    
    if datetime_cols.empty:
        st.warning("No datetime columns found in the dataset")
        return
        
    time_col = st.selectbox("Select time column:", datetime_cols)
    metric_cols = st.multiselect(
        "Select metrics to analyze:",
        df.select_dtypes(include=np.number).columns
    )
    
    if time_col and metric_cols:
        ts_data = {
            'time_column': str(time_col),
            'metrics': {}
        }
        
        for col in metric_cols:
            values_list = []
            for _, row in df.sort_values(time_col)[[time_col, col]].iterrows():
                values_list.append({
                    'time': str(row[time_col]),
                    'value': float(row[col])
                })
                
            ts_data['metrics'][str(col)] = {
                'values': values_list,
                'statistics': {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'trend': 'increasing' if float(df[col].corr(pd.to_numeric(df[time_col]))) > 0 else 'decreasing'
                }
            }
        
        prompt = f"""Analyze this time series data:
        
        Time Series Data:
        {json.dumps(ts_data, indent=2)}
        
        Please provide:
        1. Trend analysis for each metric
        2. Seasonality patterns if any
        3. Notable patterns or anomalies
        4. Potential correlations between metrics
        5. Recommendations for forecasting
        
        Format the response in markdown."""
        
        if st.button("Analyze Time Series"):
            with st.spinner("Analyzing time series..."):
                analysis = llm_analyzer.query_model(prompt, model=model)
                if analysis:
                    st.markdown(analysis)
                    
                    for col in metric_cols:
                        fig = px.line(
                            df.sort_values(time_col),
                            x=time_col,
                            y=col,
                            title=f"{col} over time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if 'analysis_history' not in st.session_state:
                        st.session_state.analysis_history = []
                        
                    st.session_state.analysis_history.append({
                        'type': 'time_series',
                        'timestamp': datetime.now().isoformat(),
                        'model': model,
                        'result': analysis
                    })
                    
def show_custom_analysis(df: pd.DataFrame, model: str):
    """Show custom analysis interface"""
    st.write("### 🤔 Custom Analysis")
    
    # Context preparation
    st.write("#### Analysis Context")
    include_sections = {
        'basic_info': st.checkbox("Include basic dataset information", True),
        'statistics': st.checkbox("Include statistical summary", True),
        'correlations': st.checkbox("Include correlations", True),
        'sample_data': st.checkbox("Include sample data", True)
    }
    
    # Custom prompt input first
    prompt = st.text_area(
        "Enter your analysis prompt:",
        help="Describe what you want to analyze in the dataset"
    )

    def prepare_context():
        context = "Please analyze this dataset:\n\n"
        
        if include_sections['basic_info']:
            context += f"""Basic Information:
            - Rows: {len(df)}
            - Columns: {len(df.columns)}
            - Column Types: {df.dtypes.to_dict()}\n\n"""
            
        if include_sections['statistics']:
            # Filter statistics based on prompt keywords
            stats_df = df.describe()
            if prompt:
                keywords = set(prompt.lower().split())
                relevant_cols = [col for col in stats_df.columns 
                               if any(kw in col.lower() for kw in keywords)]
                if relevant_cols:
                    stats_df = stats_df[relevant_cols]
            context += f"""Statistical Summary:
            {stats_df.to_string()}\n\n"""
            
        if include_sections['correlations']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_df = df[numeric_cols].corr()
                if prompt:
                    # Filter correlations based on prompt keywords
                    keywords = set(prompt.lower().split())
                    relevant_cols = [col for col in corr_df.columns 
                                   if any(kw in col.lower() for kw in keywords)]
                    if relevant_cols:
                        corr_df = corr_df[relevant_cols][relevant_cols]
                context += f"""Correlations:
                {corr_df.to_string()}\n\n"""
                
        if include_sections['sample_data']:
            sample_df = df.head()
            if prompt:
                # Filter sample data based on prompt keywords
                keywords = set(prompt.lower().split())
                relevant_cols = [col for col in sample_df.columns 
                               if any(kw in col.lower() for kw in keywords)]
                if relevant_cols:
                    sample_df = sample_df[relevant_cols]
            context += f"""Sample Data:
            {sample_df.to_string()}\n\n"""
            
        return context

    # Add Run Analysis button
    if st.button("Run Analysis", key="run_custom_analysis"):
        if prompt:
            with st.spinner("Running analysis..."):
                context = prepare_context()
                full_prompt = context + f"\n{prompt}\n\nFormat the response in markdown."
                analysis = llm_analyzer.query_model(full_prompt, model=model)
                
                if analysis:
                    st.markdown(analysis)
                    
                    # Save analysis
                    if 'analysis_history' not in st.session_state:
                        st.session_state.analysis_history = []
                        
                    st.session_state.analysis_history.append({
                        'type': 'custom',
                        'timestamp': datetime.now().isoformat(),
                        'model': model,
                        'prompt': prompt,
                        'result': analysis
                    })
        else:
            st.warning("Please enter an analysis prompt first")

def show_analysis_history():
    """Show analysis history"""
    st.write("### 📚 Analysis History")
    
    if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
        st.info("No analysis history available")
        return
        
    for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(
            f"Analysis {i+1}: {analysis['type'].title()} "
            f"({analysis['timestamp']})"
        ):
            st.write(f"Model: {analysis['model']}")
            if 'prompt' in analysis:
                st.write("Prompt:", analysis['prompt'])
            st.markdown(analysis['result'])

def show_page():
    """Main function to show AI analysis page"""
    # Initialize session state for uploaded files if not exists
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
        
    st.write("## 🤖 AI-Powered Analysis")
    
    # Check Ollama connection
    if not llm_analyzer.test_connection():
        st.error("Could not connect to Ollama API. Please make sure it's running.")
        return
    
    # Get available models
    models = llm_analyzer.get_available_models()
    if not models:
        st.error("No models available. Please install some models in Ollama.")
        return
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        models,
        index=models.index('llama2') if 'llama2' in models else 0
    )
    
    # Dataset selection
    if not st.session_state.uploaded_files:
        st.warning("Please upload some data files first!")
        return
        
    selected_file = st.selectbox(
        "Select dataset to analyze:",
        list(st.session_state.uploaded_files.keys())
    )
    
    if selected_file:
        df = st.session_state.uploaded_files[selected_file]['data']
        
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Relationships",
            "Time Series",
            "Custom Analysis",
            "History"
        ])
        
        with tab1:
            show_overview_analysis(df, model)
            
        with tab2:
            show_relationship_analysis(df, model)
            
        with tab3:
            show_time_series_analysis(df, model)
            
        with tab4:
            show_custom_analysis(df, model)
            
        with tab5:
            show_analysis_history()

if __name__ == "__main__":
    show_page()