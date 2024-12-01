import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import streamlit as st
from datetime import datetime, timedelta
import joblib
import os

class TimeSeriesPredictor:
    def __init__(self, model_path: str = "models/timeseries/"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.models = {
            'prophet': Prophet,
            'sarima': SARIMAX,
            'exponential': ExponentialSmoothing
        }
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    date_column: str,
                    target_column: str,
                    freq: str = 'D') -> pd.DataFrame:
        """Prepare data for time series analysis"""
        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Set date as index
        df = df.set_index(date_column)
        
        # Resample to regular frequency
        df = df.resample(freq)[target_column].mean().fillna(method='ffill')
        
        return df

    def train_prophet(self,
                     data: pd.DataFrame,
                     future_periods: int,
                     yearly_seasonality: bool = True,
                     weekly_seasonality: bool = True,
                     daily_seasonality: bool = True) -> Tuple[Prophet, pd.DataFrame]:
        """Train Facebook Prophet model"""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        # Initialize and train model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=future_periods)
        
        # Generate predictions
        forecast = model.predict(future)
        
        return model, forecast

    def train_sarima(self,
                    data: pd.DataFrame,
                    future_periods: int,
                    order: Tuple[int, int, int] = (1, 1, 1),
                    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)) -> Tuple[SARIMAX, pd.DataFrame]:
        """Train SARIMA model"""
        # Initialize and train model
        model = SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order
        )
        results = model.fit()
        
        # Generate predictions
        forecast = results.get_forecast(steps=future_periods)
        
        return results, forecast

    def train_exponential(self,
                         data: pd.DataFrame,
                         future_periods: int,
                         seasonal_periods: int = 7) -> Tuple[ExponentialSmoothing, pd.DataFrame]:
        """Train Exponential Smoothing model"""
        # Initialize and train model
        model = ExponentialSmoothing(
            data,
            seasonal_periods=seasonal_periods,
            seasonal='add'
        )
        results = model.fit()
        
        # Generate predictions
        forecast = results.forecast(future_periods)
        
        return results, forecast

    def evaluate_model(self,
                      actual: pd.Series,
                      predicted: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        return {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }

    def save_model(self,
                  model: object,
                  name: str,
                  metadata: Dict) -> None:
        """Save trained model"""
        model_data = {
            'model': model,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, f"{self.model_path}{name}.joblib")

    def load_model(self, name: str) -> Tuple[object, Dict]:
        """Load trained model"""
        model_data = joblib.load(f"{self.model_path}{name}.joblib")
        return model_data['model'], model_data['metadata']

    def visualize_predictions(self,
                            actual: pd.Series,
                            predicted: pd.Series,
                            forecast: pd.Series,
                            model_name: str) -> go.Figure:
        """Create visualization for time series predictions"""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add fitted values
        fig.add_trace(go.Scatter(
            x=predicted.index,
            y=predicted.values,
            mode='lines',
            name='Fitted',
            line=dict(color='green')
        ))
        
        # Add forecast with confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        if hasattr(forecast, 'conf_int'):
            lower = forecast.conf_int().iloc[:, 0]
            upper = forecast.conf_int().iloc[:, 1]
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=upper,
                fill=None,
                mode='lines',
                line_color='rgba(255,0,0,0.2)',
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=lower,
                fill='tonexty',
                mode='lines',
                line_color='rgba(255,0,0,0.2)',
                name='Lower Bound'
            ))
        
        fig.update_layout(
            title=f'Time Series Prediction ({model_name})',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig

    def show_prediction_interface(self):
        """Show time series prediction interface in Streamlit"""
        st.subheader("📈 Time Series Prediction")
        
        if not st.session_state.uploaded_files:
            st.warning("Please upload some data files first!")
            return
        
        selected_file = st.selectbox(
            "Select dataset for prediction:",
            list(st.session_state.uploaded_files.keys())
        )
        
        if selected_file:
            df = st.session_state.uploaded_files[selected_file]['data']
            
            # Configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_column = st.selectbox(
                    "Select date column",
                    df.columns
                )
                
            with col2:
                target_column = st.selectbox(
                    "Select target column",
                    df.select_dtypes(include=[np.number]).columns
                )
                
            with col3:
                model_type = st.selectbox(
                    "Select model type",
                    ['prophet', 'sarima', 'exponential']
                )
            
            # Advanced parameters
            with st.expander("Advanced Parameters"):
                if model_type == 'prophet':
                    yearly = st.checkbox("Yearly Seasonality", True)
                    weekly = st.checkbox("Weekly Seasonality", True)
                    daily = st.checkbox("Daily Seasonality", True)
                elif model_type == 'sarima':
                    p = st.slider("AR order (p)", 0, 5, 1)
                    d = st.slider("Difference order (d)", 0, 2, 1)
                    q = st.slider("MA order (q)", 0, 5, 1)
                elif model_type == 'exponential':
                    seasonal_periods = st.number_input(
                        "Seasonal Periods",
                        min_value=1,
                        value=7
                    )
            
            # Forecast parameters
            forecast_period = st.slider(
                "Forecast Period (days)",
                1, 365, 30
            )
            
            if st.button("Generate Forecast"):
                with st.spinner("Training model and generating forecast..."):
                    # Prepare data
                    data = self.prepare_data(
                        df.copy(),
                        date_column,
                        target_column
                    )
                    
                    # Split data
                    train_size = int(len(data) * 0.8)
                    train_data = data[:train_size]
                    test_data = data[train_size:]
                    
                    # Train model and generate forecast
                    if model_type == 'prophet':
                        model, forecast = self.train_prophet(
                            train_data,
                            forecast_period,
                            yearly,
                            weekly,
                            daily
                        )
                    elif model_type == 'sarima':
                        model, forecast = self.train_sarima(
                            train_data,
                            forecast_period,
                            order=(p, d, q)
                        )
                    else:
                        model, forecast = self.train_exponential(
                            train_data,
                            forecast_period,
                            seasonal_periods
                        )
                    
                    # Evaluate model
                    metrics = self.evaluate_model(test_data, forecast[:len(test_data)])
                    
                    # Display results
                    st.subheader("📊 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    
                    # Visualization
                    fig = self.visualize_predictions(
                        data,
                        forecast[:len(data)],
                        forecast[len(data):],
                        model_type
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model
                    if st.button("Save Model"):
                        model_name = st.text_input("Enter model name:")
                        if model_name:
                            self.save_model(
                                model,
                                model_name,
                                {
                                    'type': model_type,
                                    'target': target_column,
                                    'metrics': metrics
                                }
                            )
                            st.success("Model saved successfully!")