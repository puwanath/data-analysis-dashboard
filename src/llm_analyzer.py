import requests
from typing import Dict, List, Optional
import pandas as pd
import json
import streamlit as st
import time

class LLMAnalyzer:
    def __init__(self, api_url: str = "http://localhost:11434"):
        """Initialize LLM Analyzer with Ollama API"""
        self.api_url = api_url
        self.default_model = "llama2"
        
    def query_model(self, 
                    prompt: str, 
                    model: Optional[str] = None,
                    temperature: float = 0.3,
                    max_tokens: int = 2000) -> str:
        """Query the Ollama API with a prompt"""
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": model or self.default_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                st.error(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error querying LLM: {str(e)}")
            return None

    def analyze_data_overview(self, df: pd.DataFrame) -> str:
        """Generate an overview analysis of the dataset"""
        prompt = f"""As a data analyst, analyze this dataset and provide key insights:

Dataset Info:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Columns: {', '.join(df.columns)}

Basic statistics:
{df.describe().to_string()}

Please provide:
1. Key observations about the data
2. Potential patterns or trends
3. Data quality issues if any
4. Recommendations for further analysis
5. Business insights and decision support recommendations

Format the response with clear sections using markdown."""

        return self.query_model(prompt)

    def analyze_relationships(self, 
                            df: pd.DataFrame,
                            relationships: Dict[str, List[str]]) -> str:
        """Analyze relationships between different variables"""
        # Calculate correlations for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numeric_cols].corr().round(2) if len(numeric_cols) > 1 else pd.DataFrame()
        
        prompt = f"""Analyze the relationships in this dataset:

Correlation Matrix:
{corr_matrix.to_string() if not corr_matrix.empty else "No numeric correlations available"}

Detected Relationships:
{json.dumps(relationships, indent=2)}

Please provide:
1. Strong correlations identified (if any)
2. Potential causal relationships
3. Interesting patterns or clusters
4. Recommendations for feature engineering
5. Suggestions for further statistical analysis

Format the response with clear sections using markdown."""

        return self.query_model(prompt)

    def generate_dashboard_insights(self, 
                                  charts_data: Dict[str, pd.DataFrame]) -> str:
        """Generate insights for dashboard visualizations"""
        charts_summary = ""
        for name, data in charts_data.items():
            charts_summary += f"\n{name}:\n{data.describe().to_string()}\n"
        
        prompt = f"""Analyze these dashboard visualizations:

Charts Data Summary:
{charts_summary}

Please provide:
1. Key insights from each visualization
2. Relationships between different metrics
3. Actionable recommendations
4. Suggested areas for deeper analysis
5. Business impact assessment

Format the response with clear sections and bullet points."""

        return self.query_model(prompt)

    def generate_report_content(self, 
                              data: Dict[str, pd.DataFrame],
                              analysis_results: Dict[str, str]) -> str:
        """Generate content for automated reports"""
        prompt = f"""Create a comprehensive analysis report based on this data:

Dataset Overview:
{json.dumps({name: {'rows': len(df), 'columns': list(df.columns)} 
            for name, df in data.items()}, indent=2)}

Analysis Results:
{json.dumps(analysis_results, indent=2)}

Please generate:
1. Executive Summary (2-3 paragraphs)
2. Key Findings
3. Detailed Analysis
4. Recommendations
5. Next Steps

Format the response in markdown with clear sections."""

        return self.query_model(prompt)

    def suggest_visualizations(self, df: pd.DataFrame) -> str:
        """Suggest appropriate visualizations for the data"""
        data_types = df.dtypes.to_dict()
        unique_counts = df.nunique().to_dict()
        
        prompt = f"""As a data visualization expert, suggest appropriate visualizations for this dataset:

Column Data Types:
{json.dumps({col: str(dtype) for col, dtype in data_types.items()}, indent=2)}

Unique Values per Column:
{json.dumps(unique_counts, indent=2)}

Please suggest:
1. Most appropriate chart types for each column/relationship
2. Interesting combinations of variables to visualize
3. Advanced visualization techniques that might be useful
4. Dashboard layout recommendations
5. Interactive visualization features to consider

Format the response with clear sections and examples."""

        return self.query_model(prompt)

    def explain_anomalies(self, anomalies: pd.DataFrame, 
                         total_records: int) -> str:
        """Explain detected anomalies"""
        prompt = f"""Analyze these anomalies in the dataset:

Total Records: {total_records}
Anomalies Detected: {len(anomalies)}

Anomaly Statistics:
{anomalies.describe().to_string()}

Please provide:
1. Analysis of the anomaly patterns
2. Potential causes of the anomalies
3. Impact assessment
4. Recommendations for handling these anomalies
5. Prevention strategies

Format the response with clear sections using markdown."""

        return self.query_model(prompt)

    def show_llm_interface(self):
        """Show LLM analysis interface in Streamlit"""
        st.subheader("🤖 AI Analysis Settings")
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            ["llama2", "mistral", "codellama", "vicuna"]
        )
        
        # Analysis type
        # analysis_type = st.selectbox(
        #     "Analysis Type",
        #     ["Data Overview", "Relationships", "Custom Analysis"]
        # )
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Data Overview", "Relationships", "Time Series", "Segment Performance", "Custom Analysis"]
        )
        
        # Model parameters
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature",
                0.0, 1.0, 0.7,
                help="Higher values make output more random"
            )
        with col2:
            max_tokens = st.slider(
                "Max Tokens",
                100, 4000, 2000,
                help="Maximum length of response"
            )

        # Data selection
        if not st.session_state.uploaded_files:
            st.warning("Please upload some data files first!")
            return

        selected_file = st.selectbox(
            "Select Dataset",
            list(st.session_state.uploaded_files.keys())
        )

        if selected_file:
            df = st.session_state.uploaded_files[selected_file]['data']

            if analysis_type == "Data Overview":
                if st.button("Generate Overview Analysis"):
                    with st.spinner("Analyzing data..."):
                        result = self.analyze_data_overview(df)
                        if result:
                            st.markdown(result)
                            
                            # Save analysis result
                            if 'analysis_history' not in st.session_state:
                                st.session_state.analysis_history = []
                            st.session_state.analysis_history.append({
                                'type': 'overview',
                                'timestamp': pd.Timestamp.now(),
                                'dataset': selected_file,
                                'result': result
                            })

            elif analysis_type == "Relationships":
                # Analyze relationships between columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns

                col1, col2 = st.columns(2)
                with col1:
                    selected_numeric = st.multiselect(
                        "Select Numeric Columns",
                        numeric_cols
                    )
                with col2:
                    selected_categorical = st.multiselect(
                        "Select Categorical Columns",
                        categorical_cols
                    )

                if st.button("Analyze Relationships"):
                    with st.spinner("Analyzing relationships..."):
                        # Create relationships dictionary
                        relationships = {}
                        if selected_numeric:
                            relationships['numeric_correlations'] = df[selected_numeric].corr().to_dict()
                        if selected_categorical:
                            # Calculate categorical associations
                            for col1 in selected_categorical:
                                relationships[col1] = {}
                                for col2 in selected_categorical:
                                    if col1 != col2:
                                        contingency = pd.crosstab(df[col1], df[col2])
                                        relationships[col1][col2] = contingency.to_dict()

                        result = self.analyze_relationships(df, relationships)
                        if result:
                            st.markdown(result)
                            
                            # Save analysis result
                            if 'analysis_history' not in st.session_state:
                                st.session_state.analysis_history = []
                            st.session_state.analysis_history.append({
                                'type': 'relationships',
                                'timestamp': pd.Timestamp.now(),
                                'dataset': selected_file,
                                'result': result
                            })

            elif analysis_type == "Custom Analysis":
                # Custom prompt for analysis
                custom_prompt = st.text_area(
                    "Enter your analysis prompt",
                    help="Describe what you want to analyze in the data"
                )

                if custom_prompt and st.button("Run Analysis"):
                    with st.spinner("Running custom analysis..."):
                        # Prepare context
                        context = f"""
                        Dataset Info:
                        - Columns: {', '.join(df.columns)}
                        - Sample Data: {df.head().to_string()}
                        - Basic Stats: {df.describe().to_string()}
                        
                        User Question: {custom_prompt}
                        
                        Please provide a detailed analysis addressing the question.
                        Include relevant statistical observations and recommendations.
                        """

                        result = self.query_model(
                            context,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        if result:
                            st.markdown(result)
                            
                            # Save analysis result
                            if 'analysis_history' not in st.session_state:
                                st.session_state.analysis_history = []
                            st.session_state.analysis_history.append({
                                'type': 'custom',
                                'timestamp': pd.Timestamp.now(),
                                'dataset': selected_file,
                                'prompt': custom_prompt,
                                'result': result
                            })

            elif analysis_type == "Time Series":
                date_column = st.selectbox(
                    "Select Date Column",
                    df.select_dtypes(include=['datetime64', 'object']).columns
                )
                if st.button("Analyze Time Series"):
                    with st.spinner("Analyzing time patterns..."):
                        result = self.analyze_time_series(df, date_column)
                        if result:
                            st.markdown(result)

            elif analysis_type == "Segment Performance":
                segment_column = st.selectbox(
                    "Select Segment Column",
                    df.select_dtypes(include=['object', 'category']).columns
                )
                metric_columns = st.multiselect(
                    "Select Metric Columns",
                    df.select_dtypes(include=['int64', 'float64']).columns
                )
                if st.button("Analyze Segments"):
                    with st.spinner("Analyzing segments..."):
                        result = self.analyze_segment_performance(df, segment_column, metric_columns)
                        if result:
                            st.markdown(result)

        # Show analysis history
        if 'analysis_history' in st.session_state and st.session_state.analysis_history:
            with st.expander("Analysis History"):
                for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
                    st.write(f"### Analysis {i+1}")
                    st.write(f"Type: {analysis['type']}")
                    st.write(f"Dataset: {analysis['dataset']}")
                    st.write(f"Time: {analysis['timestamp']}")
                    if 'prompt' in analysis:
                        st.write(f"Prompt: {analysis['prompt']}")
                    if st.button(f"Show Result {i+1}"):
                        st.markdown(analysis['result'])

    def test_connection(self) -> bool:
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            st.error(f"Could not connect to Ollama API: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            st.error(f"Error getting models: {str(e)}")
            return []

    def validate_model(self, model_name: str) -> bool:
        """Validate if model is available in Ollama"""
        try:
            response = requests.post(
                f"{self.api_url}/api/tags/{model_name}"
            )
            return response.status_code == 200
        except Exception:
            return False
        
    def analyze_time_series(self, df: pd.DataFrame, date_column: str) -> str:
        """Analyze time series patterns in the dataset"""
        df[date_column] = pd.to_datetime(df[date_column])
        time_stats = {
            'start_date': df[date_column].min(),
            'end_date': df[date_column].max(),
            'time_span': (df[date_column].max() - df[date_column].min()).days,
            'frequency': df[date_column].diff().mode()[0]
        }
        
        prompt = f"""Analyze the time series patterns in this dataset:

    Time Range Statistics:
    {json.dumps(time_stats, default=str, indent=2)}

    Please provide:
    1. Temporal patterns and seasonality
    2. Trend analysis
    3. Anomalous periods
    4. Forecasting recommendations
    5. Key time-based insights

    Format the response with clear sections using markdown."""

        return self.query_model(prompt)

    def analyze_segment_performance(self, 
                                df: pd.DataFrame,
                                segment_column: str,
                                metric_columns: List[str]) -> str:
        """Analyze performance across different segments"""
        segment_stats = df.groupby(segment_column)[metric_columns].agg(['mean', 'std', 'min', 'max'])
        
        prompt = f"""Analyze segment performance in this dataset:

    Segment Statistics:
    {segment_stats.to_string()}

    Please provide:
    1. Top performing segments
    2. Underperforming segments
    3. Key performance drivers
    4. Segment-specific recommendations
    5. Optimization opportunities

    Format the response with clear sections using markdown."""

        return self.query_model(prompt)

    def batch_analyze(self, 
                    dfs: Dict[str, pd.DataFrame],
                    analysis_types: List[str]) -> Dict[str, str]:
        """Run batch analysis on multiple datasets"""
        results = {}
        
        for name, df in dfs.items():
            results[name] = {}
            for analysis_type in analysis_types:
                if analysis_type == 'overview':
                    results[name]['overview'] = self.analyze_data_overview(df)
                elif analysis_type == 'visualizations':
                    results[name]['visualizations'] = self.suggest_visualizations(df)
                elif analysis_type == 'relationships':
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    relationships = {'numeric_correlations': df[numeric_cols].corr().to_dict()}
                    results[name]['relationships'] = self.analyze_relationships(df, relationships)
                    
        return results