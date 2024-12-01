import streamlit as st
import pandas as pd
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from pathlib import Path
import logging
import plotly.graph_objects as go
import plotly.io as pio
import base64
import shutil
from src.visualization import Visualizer
from src.config import Config

logger = logging.getLogger(__name__)
config = Config()
visualizer = Visualizer()

class DashboardManager:
    def __init__(self, storage_path: str = "data/dashboards/"):
        """Initialize Dashboard Manager"""
        self.storage_path = storage_path
        self._init_storage()

    def _init_storage(self):
        """Initialize dashboard storage"""
        try:
            # Create storage directory
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Create subdirectories for different users
            os.makedirs(os.path.join(self.storage_path, 'shared'), exist_ok=True)
            
            logger.info("Dashboard storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing dashboard storage: {str(e)}")
            raise

    def save_dashboard(self, 
                      name: str,
                      config: Dict[str, Any],
                      user: str,
                      shared: bool = False) -> bool:
        """Save dashboard configuration"""
        try:
            dashboard_data = {
                'name': name,
                'config': config,
                'user': user,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'shared': shared,
                'version': '1.0'
            }
            
            # Determine save path
            save_dir = 'shared' if shared else user
            save_path = os.path.join(self.storage_path, save_dir)
            os.makedirs(save_path, exist_ok=True)
            
            # Save dashboard configuration
            filename = f"{name.lower().replace(' ', '_')}.json"
            with open(os.path.join(save_path, filename), 'w') as f:
                json.dump(dashboard_data, f, indent=4)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving dashboard: {str(e)}")
            return False

    def load_dashboard(self, name: str, user: str) -> Optional[Dict]:
        """Load dashboard configuration"""
        try:
            # Try loading from user's directory
            user_path = os.path.join(self.storage_path, user, f"{name}.json")
            shared_path = os.path.join(self.storage_path, 'shared', f"{name}.json")
            
            if os.path.exists(user_path):
                with open(user_path, 'r') as f:
                    return json.load(f)
            elif os.path.exists(shared_path):
                with open(shared_path, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading dashboard: {str(e)}")
            return None

    def list_dashboards(self, user: str) -> Dict[str, List[str]]:
        """List available dashboards"""
        try:
            dashboards = {
                'personal': [],
                'shared': []
            }
            
            # List personal dashboards
            user_path = os.path.join(self.storage_path, user)
            if os.path.exists(user_path):
                dashboards['personal'] = [
                    f.replace('.json', '')
                    for f in os.listdir(user_path)
                    if f.endswith('.json')
                ]
            
            # List shared dashboards
            shared_path = os.path.join(self.storage_path, 'shared')
            if os.path.exists(shared_path):
                dashboards['shared'] = [
                    f.replace('.json', '')
                    for f in os.listdir(shared_path)
                    if f.endswith('.json')
                ]
            
            return dashboards
            
        except Exception as e:
            logger.error(f"Error listing dashboards: {str(e)}")
            return {'personal': [], 'shared': []}

    def delete_dashboard(self, name: str, user: str) -> bool:
        """Delete a dashboard"""
        try:
            # Try deleting from user's directory first
            user_path = os.path.join(self.storage_path, user, f"{name}.json")
            shared_path = os.path.join(self.storage_path, 'shared', f"{name}.json")
            
            if os.path.exists(user_path):
                os.remove(user_path)
                return True
            elif os.path.exists(shared_path):
                # Check if user has permission to delete shared dashboard
                with open(shared_path, 'r') as f:
                    dashboard = json.load(f)
                    if dashboard['user'] == user:
                        os.remove(shared_path)
                        return True
                    else:
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting dashboard: {str(e)}")
            return False

    def export_dashboard(self, name: str, user: str, format: str = 'json') -> Optional[bytes]:
        """Export dashboard to file"""
        try:
            dashboard = self.load_dashboard(name, user)
            if not dashboard:
                return None
            
            if format == 'json':
                return json.dumps(dashboard, indent=4).encode()
            elif format == 'yaml':
                return yaml.dump(dashboard).encode()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting dashboard: {str(e)}")
            return None

    def import_dashboard(self, file_content: bytes, user: str) -> bool:
        """Import dashboard from file"""
        try:
            content = json.loads(file_content)
            
            # Validate dashboard structure
            required_keys = ['name', 'config', 'version']
            if not all(key in content for key in required_keys):
                raise ValueError("Invalid dashboard format")
            
            # Save imported dashboard
            return self.save_dashboard(
                content['name'],
                content['config'],
                user,
                shared=False
            )
            
        except Exception as e:
            logger.error(f"Error importing dashboard: {str(e)}")
            return False

    def create_snapshot(self, name: str, user: str) -> bool:
        """Create dashboard snapshot"""
        try:
            dashboard = self.load_dashboard(name, user)
            if not dashboard:
                return False
            
            # Create snapshots directory
            snapshots_dir = os.path.join(self.storage_path, user, 'snapshots', name)
            os.makedirs(snapshots_dir, exist_ok=True)
            
            # Save snapshot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_path = os.path.join(snapshots_dir, f"{timestamp}.json")
            
            with open(snapshot_path, 'w') as f:
                json.dump(dashboard, f, indent=4)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {str(e)}")
            return False

    def restore_snapshot(self, name: str, user: str, timestamp: str) -> bool:
        """Restore dashboard from snapshot"""
        try:
            snapshot_path = os.path.join(
                self.storage_path, user, 'snapshots', name, f"{timestamp}.json"
            )
            
            if not os.path.exists(snapshot_path):
                return False
            
            # Load and restore snapshot
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)
            
            return self.save_dashboard(
                snapshot['name'],
                snapshot['config'],
                user,
                shared=snapshot.get('shared', False)
            )
            
        except Exception as e:
            logger.error(f"Error restoring snapshot: {str(e)}")
            return False

    def show_dashboard_interface(self):
        """Show dashboard management interface in Streamlit"""
        st.subheader("📊 Dashboard Management")
        
        if 'user' not in st.session_state:
            st.warning("Please login first!")
            return
        
        user = st.session_state.user['username']
        
        # Dashboard operations
        operation = st.radio(
            "Select Operation",
            ["View Dashboards", "Create Dashboard", "Import/Export", "Snapshots"]
        )
        
        if operation == "View Dashboards":
            dashboards = self.list_dashboards(user)
            
            if dashboards['personal'] or dashboards['shared']:
                # Personal dashboards
                if dashboards['personal']:
                    st.write("### Personal Dashboards")
                    for dash in dashboards['personal']:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(dash)
                        with col2:
                            if st.button("Load", key=f"load_{dash}"):
                                config = self.load_dashboard(dash, user)
                                if config:
                                    st.session_state.current_dashboard = config
                                    st.success(f"Dashboard {dash} loaded!")
                                    st.rerun()
                        with col3:
                            if st.button("Delete", key=f"delete_{dash}"):
                                if self.delete_dashboard(dash, user):
                                    st.success(f"Dashboard {dash} deleted!")
                                    st.rerun()
                
                # Shared dashboards
                if dashboards['shared']:
                    st.write("### Shared Dashboards")
                    for dash in dashboards['shared']:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(dash)
                        with col2:
                            if st.button("Load", key=f"load_shared_{dash}"):
                                config = self.load_dashboard(dash, user)
                                if config:
                                    st.session_state.current_dashboard = config
                                    st.success(f"Dashboard {dash} loaded!")
                                    st.rerun()
            else:
                st.info("No dashboards available")
                
        elif operation == "Create Dashboard":
            st.write("### Create New Dashboard")
            
            name = st.text_input("Dashboard Name")
            shared = st.checkbox("Share Dashboard")
            
            if 'current_dashboard_config' in st.session_state and st.button("Save Dashboard"):
                if self.save_dashboard(
                    name,
                    st.session_state.current_dashboard_config,
                    user,
                    shared
                ):
                    st.success("Dashboard saved successfully!")
                    st.rerun()
                else:
                    st.error("Error saving dashboard")
                    
        elif operation == "Import/Export":
            st.write("### Import/Export Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Export Dashboard")
                dashboards = self.list_dashboards(user)
                all_dashboards = dashboards['personal'] + dashboards['shared']
                
                if all_dashboards:
                    dash_to_export = st.selectbox(
                        "Select Dashboard to Export",
                        all_dashboards
                    )
                    
                    export_format = st.selectbox(
                        "Export Format",
                        ["JSON", "YAML"]
                    )
                    
                    if st.button("Export"):
                        exported = self.export_dashboard(
                            dash_to_export,
                            user,
                            export_format.lower()
                        )
                        if exported:
                            b64 = base64.b64encode(exported).decode()
                            href = f'<a href="data:file/json;base64,{b64}" download="{dash_to_export}.{export_format.lower()}">Download Dashboard</a>'
                            st.markdown(href, unsafe_allow_html=True)
                else:
                    st.info("No dashboards available to export")
            
            with col2:
                st.write("#### Import Dashboard")
                uploaded_file = st.file_uploader(
                    "Choose dashboard file",
                    type=['json', 'yaml']
                )
                
                if uploaded_file and st.button("Import"):
                    content = uploaded_file.read()
                    if self.import_dashboard(content, user):
                        st.success("Dashboard imported successfully!")
                        st.rerun()
                    else:
                        st.error("Error importing dashboard")
                        
        elif operation == "Snapshots":
            st.write("### Dashboard Snapshots")
            
            dashboards = self.list_dashboards(user)
            all_dashboards = dashboards['personal'] + dashboards['shared']
            
            if all_dashboards:
                dashboard = st.selectbox(
                    "Select Dashboard",
                    all_dashboards
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Create Snapshot")
                    if st.button("Create Snapshot"):
                        if self.create_snapshot(dashboard, user):
                            st.success("Snapshot created successfully!")
                            st.rerun()
                        else:
                            st.error("Error creating snapshot")
                
                with col2:
                    st.write("#### Restore Snapshot")
                    snapshots_dir = os.path.join(
                        self.storage_path, user, 'snapshots', dashboard
                    )
                    
                    if os.path.exists(snapshots_dir):
                        snapshots = sorted(
                            [f.replace('.json', '') 
                             for f in os.listdir(snapshots_dir)
                             if f.endswith('.json')],
                            reverse=True
                        )
                        
                        if snapshots:
                            selected_snapshot = st.selectbox(
                                "Select Snapshot",
                                snapshots
                            )
                            
                            if st.button("Restore"):
                                if self.restore_snapshot(dashboard, user, selected_snapshot):
                                    st.success("Snapshot restored successfully!")
                                    st.rerun()
                                else:
                                    st.error("Error restoring snapshot")
                        else:
                            st.info("No snapshots available")
                    else:
                        st.info("No snapshots available")
            else:
                st.info("No dashboards available")

if __name__ == "__main__":
    dashboard_manager = DashboardManager()