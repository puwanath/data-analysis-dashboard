import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class Visualizer:
    @staticmethod
    def create_chart(df: pd.DataFrame, 
                    chart_type: str,
                    x_column: str,
                    y_column: str,
                    color_column: Optional[str] = None,
                    title: str = "",
                    animation_frame: Optional[str] = None) -> go.Figure:
        """Create different types of charts based on input parameters"""
        
        if chart_type == "Bar":
            fig = px.bar(df, x=x_column, y=y_column, color=color_column,
                        title=title, animation_frame=animation_frame)
        elif chart_type == "Line":
            fig = px.line(df, x=x_column, y=y_column, color=color_column,
                         title=title, animation_frame=animation_frame)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_column, y=y_column, color=color_column,
                           title=title, animation_frame=animation_frame)
        elif chart_type == "Area":
            fig = px.area(df, x=x_column, y=y_column, color=color_column,
                         title=title, animation_frame=animation_frame)
        elif chart_type == "Box":
            fig = px.box(df, x=x_column, y=y_column, color=color_column,
                        title=title)
        elif chart_type == "Violin":
            fig = px.violin(df, x=x_column, y=y_column, color=color_column,
                          title=title)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_column, color=color_column,
                             title=title)
        elif chart_type == "Heatmap":
            pivot_table = pd.pivot_table(df, values=y_column, 
                                       index=x_column, 
                                       columns=color_column)
            fig = px.imshow(pivot_table, title=title)
        
        # Update layout for better appearance
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            margin=dict(t=100, l=0, r=0, b=0),
            title_font=dict(size=24),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        fig = px.imshow(corr_matrix,
                       title="Correlation Heatmap",
                       aspect="auto")
        
        fig.update_layout(
            title_x=0.5,
            margin=dict(t=100, l=0, r=0, b=0),
            title_font=dict(size=24)
        )
        
        return fig

    @staticmethod
    def create_distribution_plot(df: pd.DataFrame, 
                               column: str,
                               group_by: Optional[str] = None) -> go.Figure:
        """Create distribution plot for numeric columns"""
        if group_by:
            groups = df[group_by].unique()
            hist_data = [df[df[group_by] == group][column].dropna() for group in groups]
            group_labels = [str(group) for group in groups]
        else:
            hist_data = [df[column].dropna()]
            group_labels = [column]
            
        fig = ff.create_distplot(hist_data, group_labels, show_hist=True)
        
        fig.update_layout(
            title=f"Distribution Plot - {column}",
            title_x=0.5,
            margin=dict(t=100, l=0, r=0, b=0),
            title_font=dict(size=24)
        )
        
        return fig

    @staticmethod
    def create_dashboard(charts: List[go.Figure], 
                        layout: List[List[int]]) -> go.Figure:
        """Create a dashboard with multiple charts"""
        dashboard = go.Figure()
        
        # Calculate grid dimensions
        max_rows = len(layout)
        max_cols = max(len(row) for row in layout)
        
        # Calculate dimensions for each cell
        cell_height = 1.0 / max_rows
        
        for i, row in enumerate(layout):
            cell_width = 1.0 / len(row)
            for j, chart_index in enumerate(row):
                if chart_index < len(charts):
                    chart = charts[chart_index]
                    
                    # Add chart to dashboard
                    for trace in chart.data:
                        new_trace = trace.copy()
                        new_trace.xaxis = f'x{chart_index+1}'
                        new_trace.yaxis = f'y{chart_index+1}'
                        dashboard.add_trace(new_trace)
                    
                    # Update layout for this chart
                    dashboard.update_layout(**{
                        f'xaxis{chart_index+1}': dict(
                            domain=[j*cell_width, (j+1)*cell_width],
                            anchor=f'y{chart_index+1}'
                        ),
                        f'yaxis{chart_index+1}': dict(
                            domain=[i*cell_height, (i+1)*cell_height],
                            anchor=f'x{chart_index+1}'
                        )
                    })
        
        # Update overall layout
        dashboard.update_layout(
            height=300 * max_rows,
            showlegend=False,
            title="Data Analysis Dashboard",
            title_x=0.5
        )
        
        return dashboard