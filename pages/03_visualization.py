import streamlit as st
import pandas as pd
import numpy as np
from src.visualization import Visualizer
from src.data_processor import DataProcessor
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import logging

# Initialize components
logger = logging.getLogger(__name__)
visualizer = Visualizer()
data_processor = DataProcessor()

class ChartBuilder:
    """Class to handle chart building and customization"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame,
                    chart_type: str,
                    x_col: str,
                    y_col: str,
                    color_col: Optional[str] = None,
                    title: str = "",
                    **kwargs) -> go.Figure:
        """Create various types of charts"""
        try:
            if chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_col, color=color_col,
                            title=title, **kwargs)
            
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                           title=title, **kwargs)
            
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=title, **kwargs)
                
            elif chart_type == "Box":
                fig = px.box(df, x=x_col, y=y_col, color=color_col,
                           title=title, **kwargs)
                
            elif chart_type == "Violin":
                fig = px.violin(df, x=x_col, y=y_col, color=color_col,
                              title=title, **kwargs)
                
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_col,
                                 title=title, **kwargs)
                
            elif chart_type == "Heatmap":
                pivot_table = pd.pivot_table(df, values=y_col, 
                                           index=x_col, 
                                           columns=color_col)
                fig = px.imshow(pivot_table, title=title)
                
            elif chart_type == "3D Scatter":
                z_col = kwargs.get('z_column')
                if z_col:
                    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                                      color=color_col, title=title)
                else:
                    raise ValueError("Z column required for 3D Scatter plot")
                
            elif chart_type == "Bubble":
                size_col = kwargs.get('size_column')
                if size_col:
                    fig = px.scatter(df, x=x_col, y=y_col, size=size_col,
                                   color=color_col, title=title)
                else:
                    raise ValueError("Size column required for Bubble plot")
                    
            elif chart_type == "Area":
                fig = px.area(df, x=x_col, y=y_col, color=color_col,
                            title=title, **kwargs)
                
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            # Update layout
            fig.update_layout(
                template="plotly_white",
                title_x=0.5,
                margin=dict(t=100, l=0, r=0, b=0),
                title_font=dict(size=24),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    @staticmethod
    def customize_chart(fig: go.Figure, customization: Dict[str, Any]) -> go.Figure:
        """Apply customizations to chart"""
        try:
            if not fig:
                return None
                
            # Title customization
            if 'title_text' in customization:
                fig.update_layout(title_text=customization['title_text'])
            if 'title_font_size' in customization:
                fig.update_layout(title_font_size=customization['title_font_size'])
                
            # Axes customization
            if 'xaxis_title' in customization:
                fig.update_layout(xaxis_title=customization['xaxis_title'])
            if 'yaxis_title' in customization:
                fig.update_layout(yaxis_title=customization['yaxis_title'])
                
            # Colors - Using standard Plotly color sequences
            if 'colorscale' in customization:
                color_map = {
                    'Viridis': px.colors.sequential.Viridis,
                    'Plasma': px.colors.sequential.Plasma,
                    'Inferno': px.colors.sequential.Inferno,
                    'Magma': px.colors.sequential.Magma,
                    'RdBu': px.colors.diverging.RdBu
                }
                selected_colors = color_map.get(customization['colorscale'], px.colors.sequential.Viridis)
                fig.update_traces(marker_color=selected_colors[0])
                
            # Legend
            if 'show_legend' in customization:
                fig.update_layout(showlegend=customization['show_legend'])
                
            # Grid
            if 'show_grid' in customization:
                fig.update_layout(
                    xaxis=dict(showgrid=customization['show_grid']),
                    yaxis=dict(showgrid=customization['show_grid'])
                )
                
            return fig
            
        except Exception as e:
            logger.error(f"Error customizing chart: {str(e)}")
            st.error(f"Error customizing chart: {str(e)}")
            return fig

def show_chart_builder():
    """Show chart builder interface"""
    st.write("### 📊 Chart Builder")
    
    # Select dataset
    if not st.session_state.uploaded_files:
        st.warning("Please upload some data files first!")
        return
        
    selected_file = st.selectbox(
        "Select dataset:",
        list(st.session_state.uploaded_files.keys())
    )
    
    if selected_file:
        df = st.session_state.uploaded_files[selected_file]['data']
        
        # Chart type selection
        chart_types = [
            "Line", "Bar", "Scatter", "Box", "Violin", "Histogram",
            "Heatmap", "3D Scatter", "Bubble", "Area"
        ]
        chart_type = st.selectbox("Select chart type:", chart_types)
        
        # Column selection
        col1, col2 = st.columns(2)
        
        with col1:
            x_column = st.selectbox("Select X-axis column:", df.columns)
            
            y_columns = st.multiselect(
                "Select Y-axis column(s):",
                [col for col in df.columns if col != x_column]
            )
            
        with col2:
            color_column = st.selectbox(
                "Group/Color by (optional):",
                ["None"] + [col for col in df.columns 
                           if col not in [x_column] + y_columns]
            )
            
            if chart_type == "3D Scatter":
                z_column = st.selectbox(
                    "Select Z-axis column:",
                    [col for col in df.columns 
                     if col not in [x_column] + y_columns + [color_column]]
                )
            
            if chart_type == "Bubble":
                size_column = st.selectbox(
                    "Select size column:",
                    [col for col in df.columns 
                     if col not in [x_column] + y_columns + [color_column]]
                )
        
        # Chart customization
        with st.expander("Chart Customization"):
            custom_title = st.text_input("Chart Title", "")
            title_font_size = st.slider("Title Font Size", 12, 36, 24)
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis_title = st.text_input("X-axis Title", x_column)
                show_legend = st.checkbox("Show Legend", True)
            with col2:
                y_axis_title = st.text_input("Y-axis Title", ", ".join(y_columns))
                show_grid = st.checkbox("Show Grid", True)
            
            colorscale = st.selectbox(
                "Color Scale",
                ["Viridis", "Plasma", "Inferno", "Magma", "RdBu"]
            )
        
        # Create chart for each y-column
        for y_column in y_columns:
            try:
                # Prepare chart parameters
                params = {
                    'title': custom_title or f"{y_column} vs {x_column}"
                }
                
                if chart_type == "3D Scatter":
                    params['z_column'] = z_column
                elif chart_type == "Bubble":
                    params['size_column'] = size_column
                
                # Create chart
                fig = ChartBuilder.create_chart(
                    df, chart_type, x_column, y_column,
                    color_column if color_column != "None" else None,
                    **params
                )
                
                if fig:
                    # Apply customizations
                    fig = ChartBuilder.customize_chart(fig, {
                        'title_text': custom_title or f"{y_column} vs {x_column}",
                        'title_font_size': title_font_size,
                        'xaxis_title': x_axis_title,
                        'yaxis_title': y_axis_title,
                        'show_legend': show_legend,
                        'show_grid': show_grid,
                        'colorscale': colorscale.lower()
                    })
                    
                    # Display chart
                    # st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{y_column}_{datetime.now().timestamp()}")
                    
                    # Save to session state for dashboard
                    if 'visualizations' not in st.session_state:
                        st.session_state.visualizations = []
                    
                    if st.button(f"Add to Dashboard ({y_column})"):
                        st.session_state.visualizations.append({
                            'figure': fig,
                            'title': f"{y_column} vs {x_column}",
                            'timestamp': datetime.now().isoformat()
                        })
                        st.success("Chart added to dashboard!")
                
            except Exception as e:
                logger.error(f"Error creating chart for {y_column}: {str(e)}")
                st.error(f"Error creating chart for {y_column}: {str(e)}")

def show_dashboard():
    """Show saved visualizations dashboard"""
    st.write("### 📊 Visualization Dashboard")
    
    if 'visualizations' not in st.session_state or not st.session_state.visualizations:
        st.info("No visualizations saved yet. Create some charts and add them to the dashboard!")
        return
    
    # Dashboard layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display charts
        for i, viz in enumerate(st.session_state.visualizations):
            st.write(f"#### {viz['title']}")
            # st.plotly_chart(viz['figure'], use_container_width=True)
            st.plotly_chart(viz['figure'], use_container_width=True, key=f"dashboard_{i}_{viz['timestamp']}")
            
            # Remove button
            if st.button(f"Remove {viz['title']}", key=f"remove_{i}"):
                st.session_state.visualizations.pop(i)
                st.rerun()
    
    with col2:
        st.write("### Dashboard Controls")
        
        # Clear dashboard
        if st.button("Clear Dashboard"):
            st.session_state.visualizations = []
            st.rerun()
        
        # Export dashboard
        if st.button("Export Dashboard"):
            # TODO: Implement dashboard export functionality
            st.info("Dashboard export functionality coming soon!")

def show_page():
    """Main function to show visualization page"""
    st.write("## 📊 Data Visualization")
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Chart Builder", "Dashboard"])
    
    with tab1:
        show_chart_builder()
        
    with tab2:
        show_dashboard()

if __name__ == "__main__":
    show_page()