import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import time
from datetime import datetime, timedelta
import threading
import queue
import asyncio
import websockets
import json

class RealTimeAnalyzer:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.analysis_functions = {}
        self.alert_callbacks = []
        self.websocket_server = None
        
    async def start_websocket_server(self):
        """Start WebSocket server for real-time data"""
        async def handler(websocket, path):
            try:
                async for message in websocket:
                    data = json.loads(message)
                    self.data_queue.put(data)
            except websockets.exceptions.ConnectionClosed:
                pass

        self.websocket_server = await websockets.serve(
            handler, "localhost", 8765
        )

    def add_analysis_function(self, name: str, func: Callable) -> None:
        """Add analysis function for real-time data"""
        self.analysis_functions[name] = func

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)

    def process_data(self, data: Dict) -> Dict:
        """Process incoming data with registered analysis functions"""
        results = {}
        for name, func in self.analysis_functions.items():
            try:
                results[name] = func(data)
            except Exception as e:
                st.error(f"Error in analysis function {name}: {str(e)}")
        return results

    def trigger_alerts(self, results: Dict) -> None:
        """Trigger alert callbacks if conditions are met"""
        for callback in self.alert_callbacks:
            try:
                callback(results)
            except Exception as e:
                st.error(f"Error in alert callback: {str(e)}")

    async def start_analysis(self):
        """Start real-time analysis"""
        if not self.websocket_server:
            await self.start_websocket_server()

        while not self.stop_flag.is_set():
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    results = self.process_data(data)
                    self.trigger_alerts(results)
                    
                    # Update Streamlit interface
                    if 'realtime_data' not in st.session_state:
                        st.session_state.realtime_data = []
                    st.session_state.realtime_data.append({
                        'timestamp': datetime.now(),
                        'data': data,
                        'results': results
                    })
                    
                    # Keep only last hour of data
                    cutoff = datetime.now() - timedelta(hours=1)
                    st.session_state.realtime_data = [
                        d for d in st.session_state.realtime_data 
                        if d['timestamp'] > cutoff
                    ]
                
                await asyncio.sleep(0.1)
            except Exception as e:
                st.error(f"Error in analysis loop: {str(e)}")
                await asyncio.sleep(1)

    def stop_analysis(self):
        """Stop real-time analysis"""
        self.stop_flag.set()
        if self.websocket_server:
            self.websocket_server.close()

    def show_realtime_dashboard(self):
        """Show real-time analysis dashboard in Streamlit"""
        st.subheader("📊 Real-time Analysis Dashboard")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Analysis"):
                asyncio.run(self.start_analysis())
        with col2:
            if st.button("Stop Analysis"):
                self.stop_analysis()

        # Display real-time data
        if 'realtime_data' in st.session_state:
            # Recent metrics
            st.subheader("Recent Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            if st.session_state.realtime_data:
                latest = st.session_state.realtime_data[-1]
                
                with metrics_col1:
                    st.metric(
                        "Data Points",
                        len(st.session_state.realtime_data)
                    )
                with metrics_col2:
                    st.metric(
                        "Last Update",
                        latest['timestamp'].strftime('%H:%M:%S')
                    )
                with metrics_col3:
                    st.metric(
                        "Analysis Functions",
                        len(self.analysis_functions)
                    )

            # Time series chart
            st.subheader("Real-time Data Visualization")
            chart_data = pd.DataFrame(
                [d['data'] for d in st.session_state.realtime_data]
            )
            st.line_chart(chart_data)

            # Analysis results
            st.subheader("Analysis Results")
            results_df = pd.DataFrame(
                [d['results'] for d in st.session_state.realtime_data]
            )
            st.dataframe(results_df)

            # Alert log
            if any('alert' in d['results'] for d in st.session_state.realtime_data):
                st.subheader("Alert Log")
                alerts = [
                    {
                        'timestamp': d['timestamp'],
                        'alert': d['results']['alert']
                    }
                    for d in st.session_state.realtime_data
                    if 'alert' in d['results']
                ]
                st.dataframe(pd.DataFrame(alerts))

    @staticmethod
    def sample_analysis_function(data: Dict) -> Dict:
        """Sample analysis function for demonstration"""
        return {
            'mean': np.mean(list(data.values())),
            'std': np.std(list(data.values())),
            'alert': 'High Variance' if np.std(list(data.values())) > 1 else None
        }

    @staticmethod
    def sample_alert_callback(results: Dict) -> None:
        """Sample alert callback for demonstration"""
        if results.get('alert'):
            st.warning(f"Alert: {results['alert']}")