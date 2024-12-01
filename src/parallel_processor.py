import multiprocessing as mp
import concurrent.futures
import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any
import streamlit as st
from functools import partial
import time
import logging
import traceback

class ParallelProcessor:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.logger = logging.getLogger(__name__)
    
    def parallel_dataframe(self, df: pd.DataFrame, func: Callable, chunk_size: int = None) -> pd.DataFrame:
        progress_bar = st.progress(0)
        chunks = np.array_split(df, len(df) // chunk_size + 1)
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(func, chunk): i for i, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress_bar.progress((len(results) / len(chunks)))
                
        return pd.concat(results)
    
    def parallel_apply(self,
                      df: pd.DataFrame,
                      func: Callable,
                      column: str,
                      **kwargs) -> pd.Series:
        """Apply function to column in parallel"""
        partial_func = partial(func, **kwargs)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(partial_func, df[column]))
            
        return pd.Series(results, index=df.index)
    
    def parallel_groupby(self,
                        df: pd.DataFrame,
                        group_col: str,
                        agg_func: Dict[str, List[str]]) -> pd.DataFrame:
        """Parallel groupby operations"""
        groups = df.groupby(group_col)
        group_keys = list(groups.groups.keys())
        
        def process_group(key):
            group_df = groups.get_group(key)
            return group_df.agg(agg_func)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_group, group_keys))
            
        return pd.concat(results)
    
    def parallel_merge(self,
                      dfs: List[pd.DataFrame],
                      on: List[str],
                      how: str = 'inner') -> pd.DataFrame:
        """Merge multiple dataframes in parallel"""
        if len(dfs) < 2:
            return dfs[0] if dfs else pd.DataFrame()
            
        def merge_pair(df1, df2):
            return pd.merge(df1, df2, on=on, how=how)
        
        while len(dfs) > 1:
            pairs = [(dfs[i], dfs[i+1]) for i in range(0, len(dfs)-1, 2)]
            if len(dfs) % 2:
                pairs.append((dfs[-1],))
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                dfs = list(executor.map(lambda p: p[0] if len(p) == 1 else merge_pair(*p), pairs))
        
        return dfs[0]
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() / len(df) < 0.5:  # If column has low cardinality
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        return df
    
    def validate_inputs(self, df: pd.DataFrame, func: Callable) -> bool:
        """Validate inputs before processing"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        if not callable(func):
            raise TypeError("func must be callable")
        return True
    
    def show_parallel_processing_interface(self):
        """Show parallel processing interface in Streamlit"""
        st.subheader("⚡ Parallel Processing")
        
        if not st.session_state.uploaded_files:
            st.warning("Please upload some data files first!")
            return
        
        # Operation selection with descriptions
        operation_descriptions = {
            "Custom Function": "Apply custom processing to data chunks",
            "Group By": "Aggregate data using parallel group operations",
            "Merge": "Combine multiple datasets in parallel"
        }
        
        operation = st.selectbox(
            "Select Operation",
            list(operation_descriptions.keys()),
            help="Choose the type of parallel operation to perform"
        )
        st.info(operation_descriptions[operation])
        
        if operation == "Custom Function":
            self._show_custom_function_interface()
        elif operation == "Group By":
            self._show_groupby_interface()
        elif operation == "Merge":
            self._show_merge_interface()

    def _show_custom_function_interface(self):
        selected_file = st.selectbox(
            "Select dataset",
            list(st.session_state.uploaded_files.keys())
        )
        
        if selected_file:
            df = st.session_state.uploaded_files[selected_file]['data']
            st.write(f"Dataset shape: {df.shape}")
            
            custom_code = st.text_area(
                "Custom Processing Function",
                """def custom_function(chunk):
        # Your processing logic here
        # Example: chunk['new_col'] = chunk['existing_col'] * 2
        return chunk""",
                height=150
            )
            
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=1,
                value=min(1000, len(df)),
                help="Number of rows per chunk for parallel processing"
            )
            
            if st.button("Process", help="Start parallel processing"):
                with st.spinner("Processing in parallel..."):
                    try:
                        exec(custom_code)
                        result = self.parallel_dataframe(
                            df,
                            locals()['custom_function'],
                            chunk_size
                        )
                        self._show_results(result, "processing")
                    except Exception as e:
                        self._handle_error(e)

    def _show_groupby_interface(self):
        selected_file = st.selectbox(
            "Select dataset",
            list(st.session_state.uploaded_files.keys())
        )
        
        if selected_file:
            df = st.session_state.uploaded_files[selected_file]['data']
            
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox(
                    "Group By Column",
                    df.columns
                )
            
            with col2:
                agg_cols = st.multiselect(
                    "Columns to Aggregate",
                    df.select_dtypes(include=[np.number]).columns
                )
            
            agg_funcs = st.multiselect(
                "Aggregation Functions",
                ['mean', 'sum', 'count', 'min', 'max']
            )
            
            if agg_cols and agg_funcs and st.button("Process"):
                with st.spinner("Processing in parallel..."):
                    try:
                        agg_dict = {col: agg_funcs for col in agg_cols}
                        result = self.parallel_groupby(df, group_col, agg_dict)
                        self._show_results(result, "aggregation")
                    except Exception as e:
                        self._handle_error(e)

    def _show_merge_interface(self):
        selected_files = st.multiselect(
            "Select Datasets to Merge (2 or more)",
            list(st.session_state.uploaded_files.keys())
        )
        
        if len(selected_files) >= 2:
            dfs = [st.session_state.uploaded_files[f]['data'] for f in selected_files]
            common_columns = set.intersection(*[set(df.columns) for df in dfs])
            
            merge_cols = st.multiselect(
                "Merge Columns",
                list(common_columns),
                help="Select columns to join datasets on"
            )
            
            merge_type = st.selectbox(
                "Merge Type",
                ['inner', 'outer', 'left', 'right'],
                help="Choose how to combine the datasets"
            )
            
            if merge_cols and st.button("Merge Datasets"):
                with st.spinner("Merging in parallel..."):
                    try:
                        result = self.parallel_merge(dfs, merge_cols, merge_type)
                        self._show_results(result, "merge")
                    except Exception as e:
                        self._handle_error(e)

    def _show_results(self, result: pd.DataFrame, operation_type: str):
        processing_time = time.time() - st.session_state.get('start_time', time.time())
        st.success(f"{operation_type.title()} completed in {processing_time:.2f} seconds!")
        
        st.write(f"Result shape: {result.shape}")
        st.dataframe(result)
        
        csv = result.to_csv(index=False)
        st.download_button(
            label=f"Download {operation_type} results",
            data=csv,
            file_name=f"{operation_type}_results.csv",
            mime="text/csv"
        )

    def _handle_error(self, error: Exception):
        st.error(f"Error occurred: {str(error)}")
        self.logger.error(traceback.format_exc())
