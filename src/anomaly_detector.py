import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Class for detecting anomalies in data using multiple methods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.methods = {
            'isolation_forest': self._isolation_forest_detect,
            'zscore': self._zscore_detect,
            'iqr': self._iqr_detect,
            'local_outlier_factor': self._lof_detect,
            'elliptic_envelope': self._elliptic_envelope_detect,
            'moving_average': self._moving_average_detect,
            'dbscan': self._dbscan_detect
        }

    def detect_anomalies(self,
                        df: pd.DataFrame,
                        method: str = 'isolation_forest',
                        columns: Optional[List[str]] = None,
                        threshold: float = 3.0,
                        **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect anomalies in the dataset using specified method
        
        Args:
            df: Input DataFrame
            method: Anomaly detection method
            columns: Columns to analyze
            threshold: Threshold for anomaly detection
            **kwargs: Additional parameters for specific methods
        
        Returns:
            Tuple of DataFrame with anomaly flags and statistics
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns

            if not columns:
                raise ValueError("No numeric columns available for analysis")

            # Handle missing values
            df_clean = df[columns].fillna(df[columns].mean())

            # Get detection method
            if method not in self.methods:
                raise ValueError(f"Unsupported method: {method}")
            
            detect_func = self.methods[method]
            
            # Detect anomalies
            result_df, stats = detect_func(
                df_clean, 
                threshold=threshold,
                **kwargs
            )

            # Add anomaly flags to original dataframe
            df_result = df.copy()
            df_result['is_anomaly'] = result_df['is_anomaly']
            df_result['anomaly_score'] = result_df['anomaly_score']

            return df_result, stats

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise

    def _isolation_forest_detect(self,
                               df: pd.DataFrame,
                               threshold: float = 0.1,
                               **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using Isolation Forest"""
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(df)
            
            # Initialize and fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=threshold,
                random_state=42,
                **kwargs
            )
            
            # Get anomaly scores (-1 for anomalies, 1 for normal points)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            anomaly_scores = iso_forest.score_samples(X_scaled)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'is_anomaly': anomaly_labels == -1,
                'anomaly_score': anomaly_scores
            })
            
            # Calculate statistics
            stats = self._calculate_stats(df, result_df)
            stats['method'] = 'isolation_forest'
            
            return result_df, stats

        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {str(e)}")
            raise

    def _zscore_detect(self,
                      df: pd.DataFrame,
                      threshold: float = 3.0,
                      **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using Z-score method"""
        try:
            # Calculate Z-scores
            z_scores = np.abs(self.scaler.fit_transform(df))
            
            # Mark anomalies where any column exceeds threshold
            anomaly_mask = (z_scores > threshold).any(axis=1)
            anomaly_scores = z_scores.max(axis=1)
            
            result_df = pd.DataFrame({
                'is_anomaly': anomaly_mask,
                'anomaly_score': anomaly_scores
            })
            
            stats = self._calculate_stats(df, result_df)
            stats['method'] = 'zscore'
            
            return result_df, stats

        except Exception as e:
            logger.error(f"Error in Z-score detection: {str(e)}")
            raise

    def _iqr_detect(self,
                    df: pd.DataFrame,
                    threshold: float = 1.5,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using IQR method"""
        try:
            anomaly_mask = pd.Series(False, index=df.index)
            anomaly_scores = pd.Series(0.0, index=df.index)
            
            for column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                column_anomalies = (df[column] < lower_bound) | (df[column] > upper_bound)
                column_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                
                anomaly_mask |= column_anomalies
                anomaly_scores = np.maximum(anomaly_scores, column_scores)
            
            result_df = pd.DataFrame({
                'is_anomaly': anomaly_mask,
                'anomaly_score': anomaly_scores
            })
            
            stats = self._calculate_stats(df, result_df)
            stats['method'] = 'iqr'
            
            return result_df, stats

        except Exception as e:
            logger.error(f"Error in IQR detection: {str(e)}")
            raise

    def _lof_detect(self,
                    df: pd.DataFrame,
                    threshold: float = -1.5,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using Local Outlier Factor"""
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(df)
            
            # Initialize and fit LOF
            lof = LocalOutlierFactor(
                contamination=abs(threshold),
                novelty=True,
                **kwargs
            )
            lof.fit(X_scaled)
            
            # Get anomaly scores
            anomaly_labels = lof.predict(X_scaled)
            anomaly_scores = -lof.score_samples(X_scaled)
            
            result_df = pd.DataFrame({
                'is_anomaly': anomaly_labels == -1,
                'anomaly_score': anomaly_scores
            })
            
            stats = self._calculate_stats(df, result_df)
            stats['method'] = 'local_outlier_factor'
            
            return result_df, stats

        except Exception as e:
            logger.error(f"Error in LOF detection: {str(e)}")
            raise

    def _elliptic_envelope_detect(self,
                                df: pd.DataFrame,
                                threshold: float = 0.1,
                                **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using Elliptic Envelope"""
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(df)
            
            # Initialize and fit Elliptic Envelope
            envelope = EllipticEnvelope(
                contamination=threshold,
                random_state=42,
                **kwargs
            )
            
            # Get anomaly scores
            anomaly_labels = envelope.fit_predict(X_scaled)
            anomaly_scores = -envelope.score_samples(X_scaled)
            
            result_df = pd.DataFrame({
                'is_anomaly': anomaly_labels == -1,
                'anomaly_score': anomaly_scores
            })
            
            stats = self._calculate_stats(df, result_df)
            stats['method'] = 'elliptic_envelope'
            
            return result_df, stats

        except Exception as e:
            logger.error(f"Error in Elliptic Envelope detection: {str(e)}")
            raise

    def _moving_average_detect(self,
                             df: pd.DataFrame,
                             threshold: float = 3.0,
                             window: int = 5,
                             **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using Moving Average"""
        try:
            anomaly_mask = pd.Series(False, index=df.index)
            anomaly_scores = pd.Series(0.0, index=df.index)
            
            for column in df.columns:
                # Calculate moving average and standard deviation
                ma = df[column].rolling(window=window).mean()
                mstd = df[column].rolling(window=window).std()
                
                # Calculate z-scores
                z_scores = np.abs((df[column] - ma) / mstd)
                
                # Mark anomalies
                column_anomalies = z_scores > threshold
                
                anomaly_mask |= column_anomalies
                anomaly_scores = np.maximum(anomaly_scores, z_scores)
            
            result_df = pd.DataFrame({
                'is_anomaly': anomaly_mask,
                'anomaly_score': anomaly_scores
            })
            
            stats = self._calculate_stats(df, result_df)
            stats['method'] = 'moving_average'
            
            return result_df, stats

        except Exception as e:
            logger.error(f"Error in Moving Average detection: {str(e)}")
            raise

    def _dbscan_detect(self,
                      df: pd.DataFrame,
                      eps: float = 0.5,
                      min_samples: int = 5,
                      **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using DBSCAN"""
        try:
            from sklearn.cluster import DBSCAN
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(df)
            
            # Initialize and fit DBSCAN
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                **kwargs
            )
            
            # Get cluster labels (-1 for outliers)
            cluster_labels = dbscan.fit_predict(X_scaled)
            
            # Calculate anomaly scores based on nearest neighbor distances
            distances = []
            for point in X_scaled:
                dist = np.min(np.linalg.norm(X_scaled - point, axis=1))
                distances.append(dist)
            
            result_df = pd.DataFrame({
                'is_anomaly': cluster_labels == -1,
                'anomaly_score': distances
            })
            
            stats = self._calculate_stats(df, result_df)
            stats['method'] = 'dbscan'
            
            return result_df, stats

        except Exception as e:
            logger.error(f"Error in DBSCAN detection: {str(e)}")
            raise

    def _calculate_stats(self,
                        df: pd.DataFrame,
                        result_df: pd.DataFrame) -> Dict:
        """Calculate anomaly detection statistics"""
        try:
            anomaly_indices = result_df[result_df['is_anomaly']].index
            normal_indices = result_df[~result_df['is_anomaly']].index
            
            stats = {
                'total_points': len(df),
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(df)) * 100,
                'timestamp': datetime.now().isoformat(),
                'column_stats': {}
            }
            
            for column in df.columns:
                column_stats = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'anomaly_mean': float(df.loc[anomaly_indices, column].mean()),
                    'normal_mean': float(df.loc[normal_indices, column].mean())
                }
                stats['column_stats'][column] = column_stats
            
            return stats

        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def visualize_anomalies(self,
                           df: pd.DataFrame,
                           anomaly_mask: pd.Series,
                           columns: List[str],
                           method: str) -> Dict[str, go.Figure]:
        """Create visualizations for anomaly detection results"""
        try:
            figures = {}
            
            # Time series plot if datetime column exists
            datetime_cols = df.select_dtypes(
                include=['datetime64']
            ).columns
            
            if len(datetime_cols) > 0:
                time_col = datetime_cols[0]
                for column in columns:
                    fig = go.Figure()
                    
                    # Normal points
                    fig.add_trace(go.Scatter(
                        x=df.loc[~anomaly_mask, time_col],
                        y=df.loc[~anomaly_mask, column],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=8)
                    ))
                    
                    # Anomaly points
                    fig.add_trace(go.Scatter(
                        x=df.loc[anomaly_mask, time_col],
                        y=df.loc[anomaly_mask, column],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=12, symbol='x')
                    ))
                    
                    fig.update_layout(
                        title=f"Anomalies in {column} ({method})",
                        xaxis_title=time_col,
                        yaxis_title=column
                    )
                    
                    figures[f'timeseries_{column}'] = fig
            
            # Distribution plots
            for column in columns:
                fig = ff.create_distplot(
                    [
                        df.loc[~anomaly_mask, column].values,
                        df.loc[anomaly_mask, column].values
                    ],
                    ['Normal', 'Anomaly'],
                    colors=['blue', 'red']
                )
                
                fig.update_layout(
                    title=f"Distribution of {column} values",
                    xaxis_title=column,
                    yaxis_title='Density'
                )
                
                figures[f'distribution_{column}'] = fig
            
            # Scatter plot if multiple
            if len(columns) >= 2:
                for i, col1 in enumerate(columns[:-1]):
                    for col2 in columns[i+1:]:
                        fig = go.Figure()
                        
                        # Normal points
                        fig.add_trace(go.Scatter(
                            x=df.loc[~anomaly_mask, col1],
                            y=df.loc[~anomaly_mask, col2],
                            mode='markers',
                            name='Normal',
                            marker=dict(color='blue', size=8)
                        ))
                        
                        # Anomaly points
                        fig.add_trace(go.Scatter(
                            x=df.loc[anomaly_mask, col1],
                            y=df.loc[anomaly_mask, col2],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(color='red', size=12, symbol='x')
                        ))
                        
                        fig.update_layout(
                            title=f"Anomalies: {col1} vs {col2}",
                            xaxis_title=col1,
                            yaxis_title=col2
                        )
                        
                        figures[f'scatter_{col1}_{col2}'] = fig
            
            # Create 3D scatter plot if 3 or more columns
            if len(columns) >= 3:
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=df.loc[~anomaly_mask, columns[0]],
                        y=df.loc[~anomaly_mask, columns[1]],
                        z=df.loc[~anomaly_mask, columns[2]],
                        mode='markers',
                        name='Normal',
                        marker=dict(
                            size=8,
                            color='blue',
                            opacity=0.6
                        )
                    ),
                    go.Scatter3d(
                        x=df.loc[anomaly_mask, columns[0]],
                        y=df.loc[anomaly_mask, columns[1]],
                        z=df.loc[anomaly_mask, columns[2]],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='x',
                            opacity=0.8
                        )
                    )
                ])
                
                fig.update_layout(
                    title=f"3D Visualization of Anomalies",
                    scene=dict(
                        xaxis_title=columns[0],
                        yaxis_title=columns[1],
                        zaxis_title=columns[2]
                    )
                )
                
                figures['3d_scatter'] = fig
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def analyze_anomalies(self,
                         df: pd.DataFrame,
                         anomaly_mask: pd.Series,
                         columns: List[str]) -> Dict:
        """Analyze detected anomalies"""
        try:
            analysis = {
                'summary': {
                    'total_points': len(df),
                    'anomaly_count': anomaly_mask.sum(),
                    'anomaly_percentage': (anomaly_mask.sum() / len(df)) * 100
                },
                'column_analysis': {},
                'patterns': [],
                'clusters': {}
            }
            
            # Analyze each column
            for column in columns:
                normal_values = df.loc[~anomaly_mask, column]
                anomaly_values = df.loc[anomaly_mask, column]
                
                col_analysis = {
                    'normal_stats': {
                        'mean': float(normal_values.mean()),
                        'std': float(normal_values.std()),
                        'min': float(normal_values.min()),
                        'max': float(normal_values.max()),
                        'q1': float(normal_values.quantile(0.25)),
                        'q3': float(normal_values.quantile(0.75))
                    },
                    'anomaly_stats': {
                        'mean': float(anomaly_values.mean()),
                        'std': float(anomaly_values.std()),
                        'min': float(anomaly_values.min()),
                        'max': float(anomaly_values.max()),
                        'q1': float(anomaly_values.quantile(0.25)),
                        'q3': float(anomaly_values.quantile(0.75))
                    }
                }
                
                # Detect patterns
                if len(anomaly_values) > 0:
                    if anomaly_values.mean() > normal_values.mean() + 2 * normal_values.std():
                        analysis['patterns'].append(f"High values in {column}")
                    elif anomaly_values.mean() < normal_values.mean() - 2 * normal_values.std():
                        analysis['patterns'].append(f"Low values in {column}")
                    
                    # Check for clusters
                    from sklearn.cluster import KMeans
                    if len(anomaly_values) >= 3:
                        kmeans = KMeans(n_clusters=min(3, len(anomaly_values)))
                        X = anomaly_values.values.reshape(-1, 1)
                        clusters = kmeans.fit_predict(X)
                        
                        analysis['clusters'][column] = {
                            f"cluster_{i}": {
                                'center': float(center),
                                'size': int((clusters == i).sum())
                            }
                            for i, center in enumerate(kmeans.cluster_centers_.flatten())
                        }
                
                analysis['column_analysis'][column] = col_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing anomalies: {str(e)}")
            raise

    def get_anomaly_explanation(self,
                              df: pd.DataFrame,
                              anomaly_mask: pd.Series,
                              anomaly_scores: pd.Series,
                              columns: List[str]) -> pd.DataFrame:
        """Generate explanations for detected anomalies"""
        try:
            explanations = []
            
            for idx in df[anomaly_mask].index:
                row = df.loc[idx]
                score = anomaly_scores[idx]
                
                # Find contributing columns
                contrib_cols = []
                for col in columns:
                    col_zscore = abs((row[col] - df[col].mean()) / df[col].std())
                    if col_zscore > 2:
                        contrib_cols.append({
                            'column': col,
                            'value': row[col],
                            'zscore': col_zscore,
                            'normal_range': f"{df[col].mean() - 2*df[col].std():.2f} to {df[col].mean() + 2*df[col].std():.2f}"
                        })
                
                explanation = {
                    'index': idx,
                    'anomaly_score': score,
                    'main_factors': sorted(contrib_cols, key=lambda x: x['zscore'], reverse=True),
                    'total_contributors': len(contrib_cols)
                }
                
                explanations.append(explanation)
            
            return pd.DataFrame(explanations)
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            raise

    def save_results(self,
                    df: pd.DataFrame,
                    anomaly_mask: pd.Series,
                    anomaly_scores: pd.Series,
                    stats: Dict,
                    method: str,
                    filename: str):
        """Save anomaly detection results"""
        try:
            results = {
                'data': df.to_dict(),
                'anomalies': {
                    'mask': anomaly_mask.to_list(),
                    'scores': anomaly_scores.to_list()
                },
                'stats': stats,
                'method': method,
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(results, f)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def load_results(self, filename: str) -> Dict:
        """Load saved anomaly detection results"""
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
            
            # Convert back to DataFrame
            results['data'] = pd.DataFrame.from_dict(results['data'])
            results['anomalies']['mask'] = pd.Series(results['anomalies']['mask'])
            results['anomalies']['scores'] = pd.Series(results['anomalies']['scores'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            raise