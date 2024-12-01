import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import jinja2
import pdfkit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
import json
import yaml
from pathlib import Path
import logging
import plotly.io as pio

class ReportGenerator:
    def __init__(self, template_dir: str = "templates/"):
        """Initialize Report Generator"""
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir))
        )
        self._init_templates()

    def _init_templates(self):
        """Initialize default report templates"""
        default_templates = {
            'executive_summary': """
            <h1>Executive Summary Report</h1>
            <h2>Overview</h2>
            <p>{{ overview }}</p>
            
            <h2>Key Metrics</h2>
            <ul>
            {% for metric in key_metrics %}
                <li>{{ metric.name }}: {{ metric.value }}</li>
            {% endfor %}
            </ul>
            
            <h2>Visualizations</h2>
            {% for viz in visualizations %}
                <img src="data:image/png;base64,{{ viz.image }}" alt="{{ viz.title }}">
                <p>{{ viz.description }}</p>
            {% endfor %}
            
            <h2>Recommendations</h2>
            <ul>
            {% for rec in recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
            """,
            
            'detailed_analysis': """
            <h1>Detailed Analysis Report</h1>
            
            <h2>Methodology</h2>
            <p>{{ methodology }}</p>
            
            <h2>Data Analysis</h2>
            {% for section in analysis_sections %}
                <h3>{{ section.title }}</h3>
                <p>{{ section.content }}</p>
                {% if section.table %}
                    {{ section.table }}
                {% endif %}
                {% if section.visualization %}
                    <img src="data:image/png;base64,{{ section.visualization }}" 
                         alt="{{ section.title }}">
                {% endif %}
            {% endfor %}
            
            <h2>Statistical Analysis</h2>
            {% for stat in statistics %}
                <h3>{{ stat.name }}</h3>
                <p>{{ stat.description }}</p>
                <pre>{{ stat.value }}</pre>
            {% endfor %}
            
            <h2>Conclusions</h2>
            <p>{{ conclusions }}</p>
            """,
            
            'dashboard': """
            <h1>{{ title }}</h1>
            <div class="dashboard">
                {% for widget in widgets %}
                    <div class="widget">
                        <h3>{{ widget.title }}</h3>
                        {% if widget.type == 'chart' %}
                            <img src="data:image/png;base64,{{ widget.content }}"
                                 alt="{{ widget.title }}">
                        {% elif widget.type == 'metric' %}
                            <div class="metric">
                                <span class="value">{{ widget.value }}</span>
                                <span class="label">{{ widget.label }}</span>
                            </div>
                        {% elif widget.type == 'table' %}
                            {{ widget.content }}
                        {% endif %}
                    </div>
                {% endfor %}
            </div>
            """
        }
        
        for name, content in default_templates.items():
            template_file = self.template_dir / f"{name}.html"
            if not template_file.exists():
                template_file.write_text(content)

    def generate_report(self,
                       template_name: str,
                       data: Dict,
                       output_format: str = 'html') -> bytes:
        """Generate report from template and data"""
        try:
            template = self.jinja_env.get_template(f"{template_name}.html")
            html_content = template.render(**data)
            
            if output_format == 'html':
                return html_content.encode()
            elif output_format == 'pdf':
                return pdfkit.from_string(html_content, False)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        except jinja2.TemplateNotFound:
            self.logger.error(f"Template '{template_name}' not found.")
            raise
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise


    def prepare_visualization(self, fig) -> str:
        """Convert visualization to base64 string"""
        try:
            if isinstance(fig, (plt.Figure, sns.FacetGrid)):
                # For matplotlib/seaborn figures
                from io import BytesIO
                buffer = BytesIO()
                fig.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode()
            else:
                # For plotly figures
                return base64.b64encode(pio.to_image(fig, format='png')).decode()
        except Exception as e:
            self.logger.error(f"Error preparing visualization: {str(e)}")
            return ""
        
    def create_executive_summary(self, 
                               df: pd.DataFrame,
                               analysis_results: Dict,
                               visualizations: List[Dict]) -> bytes:
        """Create executive summary report"""
        try:
            # Prepare data for template
            template_data = {
                'overview': analysis_results.get('overview', ''),
                'key_metrics': [
                    {'name': 'Total Records', 'value': len(df)},
                    {'name': 'Time Period', 'value': f"{df.index.min()} to {df.index.max()}"},
                    {'name': 'Data Quality', 'value': f"{(1 - df.isnull().mean().mean()) * 100:.1f}%"}
                ],
                'visualizations': [
                    {
                        'title': viz['title'],
                        'image': self.prepare_visualization(viz['figure']),
                        'description': viz['description']
                    }
                    for viz in visualizations
                ],
                'recommendations': analysis_results.get('recommendations', [])
            }
            
            return self.generate_report('executive_summary', template_data, 'pdf')
        except Exception as e:
            self.logger.error(f"Error creating executive summary: {str(e)}")
            raise

    def create_detailed_analysis(self,
                               df: pd.DataFrame,
                               analysis_sections: List[Dict],
                               statistics: List[Dict]) -> bytes:
        """Create detailed analysis report"""
        try:
            # Prepare methodology section
            methodology = """
            This analysis was performed using the following steps:
            1. Data cleaning and preprocessing
            2. Exploratory data analysis
            3. Statistical analysis
            4. Trend identification
            5. Pattern recognition
            """
            
            # Prepare template data
            template_data = {
                'methodology': methodology,
                'analysis_sections': [
                    {
                        'title': section['title'],
                        'content': section['content'],
                        'table': section.get('table', None),
                        'visualization': self.prepare_visualization(section['figure'])
                        if 'figure' in section else None
                    }
                    for section in analysis_sections
                ],
                'statistics': statistics,
                'conclusions': analysis_sections[-1]['content']
                if analysis_sections else ''
            }
            
            return self.generate_report('detailed_analysis', template_data, 'pdf')
        except Exception as e:
            self.logger.error(f"Error creating detailed analysis: {str(e)}")
            raise

    def create_dashboard_report(self,
                              title: str,
                              widgets: List[Dict]) -> bytes:
        """Create dashboard report"""
        try:
            template_data = {
                'title': title,
                'widgets': [
                    {
                        'title': widget['title'],
                        'type': widget['type'],
                        'content': (
                            self.prepare_visualization(widget['content'])
                            if widget['type'] == 'chart'
                            else widget['content']
                        ),
                        'value': widget.get('value'),
                        'label': widget.get('label')
                    }
                    for widget in widgets
                ]
            }
            
            return self.generate_report('dashboard', template_data, 'pdf')
        except Exception as e:
            self.logger.error(f"Error creating dashboard report: {str(e)}")
            raise

    def show_report_interface(self):
        """Show report generation interface in Streamlit"""
        st.subheader("📊 Report Generation")
        
        if not st.session_state.uploaded_files:
            st.warning("Please upload some data files first!")
            return
            
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Detailed Analysis", "Dashboard"]
        )
        
        # Data selection
        selected_file = st.selectbox(
            "Select Dataset",
            list(st.session_state.uploaded_files.keys())
        )
        
        if selected_file:
            df = st.session_state.uploaded_files[selected_file]['data']
            
            if report_type == "Executive Summary":
                st.write("### Executive Summary Configuration")
                
                # Overview
                overview = st.text_area(
                    "Overview",
                    "This report provides an analysis of the selected dataset..."
                )
                
                # Visualizations
                st.write("Select Visualizations")
                if 'visualizations' in st.session_state:
                    selected_viz = st.multiselect(
                        "Choose visualizations to include",
                        list(range(len(st.session_state.visualizations))),
                        format_func=lambda x: f"Visualization {x+1}"
                    )
                    viz_data = [st.session_state.visualizations[i] for i in selected_viz]
                else:
                    viz_data = []
                
                # Recommendations
                recommendations = []
                num_recommendations = st.number_input(
                    "Number of Recommendations",
                    min_value=1,
                    value=3
                )
                for i in range(num_recommendations):
                    rec = st.text_input(f"Recommendation {i+1}")
                    if rec:
                        recommendations.append(rec)
                
                if st.button("Generate Executive Summary"):
                    try:
                        analysis_results = {
                            'overview': overview,
                            'recommendations': recommendations
                        }
                        
                        report = self.create_executive_summary(
                            df,
                            analysis_results,
                            viz_data
                        )
                        
                        # Provide download link
                        b64 = base64.b64encode(report).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" ' \
                               'download="executive_summary.pdf">Download Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        
            elif report_type == "Detailed Analysis":
                st.write("### Detailed Analysis Configuration")
                
                # Analysis sections
                sections = []
                num_sections = st.number_input(
                    "Number of Analysis Sections",
                    min_value=1,
                    value=3
                )
                
                for i in range(num_sections):
                    with st.expander(f"Section {i+1}"):
                        title = st.text_input(f"Section {i+1} Title")
                        content = st.text_area(f"Section {i+1} Content")
                        
                        # Add visualization if available
                        if 'visualizations' in st.session_state:
                            viz_index = st.selectbox(
                                f"Add visualization to section {i+1}",
                                [-1] + list(range(len(st.session_state.visualizations))),
                                format_func=lambda x: "None" if x == -1 else f"Visualization {x+1}"
                            )
                            
                            if viz_index >= 0:
                                sections.append({
                                    'title': title,
                                    'content': content,
                                    'figure': st.session_state.visualizations[viz_index]['figure']
                                })
                            else:
                                sections.append({
                                    'title': title,
                                    'content': content
                                })
                        else:
                            sections.append({
                                'title': title,
                                'content': content
                            })
                
                # Statistical analysis
                stats = []
                include_stats = st.checkbox("Include Statistical Analysis")
                if include_stats:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    selected_cols = st.multiselect(
                        "Select columns for statistical analysis",
                        numeric_cols
                    )
                    
                    if selected_cols:
                        for col in selected_cols:
                            stats.append({
                                'name': col,
                                'description': f"Statistical summary of {col}",
                                'value': df[col].describe().to_string()
                            })
                
                if st.button("Generate Detailed Analysis"):
                    try:
                        report = self.create_detailed_analysis(
                            df,
                            sections,
                            stats
                        )
                        
                        # Provide download link
                        b64 = base64.b64encode(report).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" ' \
                               'download="detailed_analysis.pdf">Download Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        
            elif report_type == "Dashboard":
                st.write("### Dashboard Configuration")
                
                # Dashboard title
                title = st.text_input("Dashboard Title", "Data Analysis Dashboard")
                
                # Widgets
                widgets = []
                num_widgets = st.number_input(
                    "Number of Dashboard Widgets",
                    min_value=1,
                    value=4
                )
                
                for i in range(num_widgets):
                    with st.expander(f"Widget {i+1}"):
                        widget_title = st.text_input(f"Widget {i+1} Title")
                        widget_type = st.selectbox(
                            f"Widget {i+1} Type",
                            ["chart", "metric", "table"],
                            key=f"widget_type_{i}"
                        )
                        
                        if widget_type == "chart":
                            if 'visualizations' in st.session_state:
                                viz_index = st.selectbox(
                                    "Select visualization",
                                    range(len(st.session_state.visualizations)),
                                    format_func=lambda x: f"Visualization {x+1}",
                                    key=f"viz_select_{i}"
                                )
                                widgets.append({
                                    'title': widget_title,
                                    'type': 'chart',
                                    'content': st.session_state.visualizations[viz_index]['figure']
                                })
                        elif widget_type == "metric":
                            value = st.number_input(f"Metric Value", key=f"metric_value_{i}")
                            label = st.text_input(f"Metric Label", key=f"metric_label_{i}")
                            widgets.append({
                                'title': widget_title,
                                'type': 'metric',
                                'value': value,
                                'label': label
                            })
                        elif widget_type == "table":
                            cols = st.multiselect(
                                "Select columns for table",
                                df.columns,
                                key=f"table_cols_{i}"
                            )
                            if cols:
                                widgets.append({
                                    'title': widget_title,
                                    'type': 'table',
                                    'content': df[cols].head().to_html()
                                })
                
                if st.button("Generate Dashboard Report"):
                    try:
                        report = self.create_dashboard_report(
                            title,
                            widgets
                        )
                        
                        # Provide download link
                        b64 = base64.b64encode(report).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" ' \
                               'download="dashboard_report.pdf">Download Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")

    def save_template(self, name: str, content: str):
        """Save custom report template"""
        try:
            template_file = self.template_dir / f"{name}.html"
            template_file.write_text(content)
            return True
        except Exception as e:
            self.logger.error(f"Error saving template: {str(e)}")
            return False

    def list_templates(self) -> List[str]:
        """List available report templates"""
        return [f.stem for f in self.template_dir.glob("*.html")]