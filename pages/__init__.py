import streamlit as st
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page registry
PAGES: Dict[str, Dict[str, Any]] = {
    "data_upload": {
        "title": "Data Upload",
        "icon": "📤",
        "access_level": ["admin", "analyst", "viewer"],
        "order": 1,
        "description": "Upload and manage data files"
    },
    "data_explorer": {
        "title": "Data Explorer",
        "icon": "🔍",
        "access_level": ["admin", "analyst", "viewer"],
        "order": 2,
        "description": "Explore and analyze data"
    },
    "visualization": {
        "title": "Visualization",
        "icon": "📊",
        "access_level": ["admin", "analyst", "viewer"],
        "order": 3,
        "description": "Create and customize visualizations"
    },
    "ai_analysis": {
        "title": "AI Analysis",
        "icon": "🤖",
        "access_level": ["admin", "analyst"],
        "order": 4,
        "description": "AI-powered data analysis"
    }
}

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_page = None
        st.session_state.uploaded_files = {}
        st.session_state.analysis_results = {}
        st.session_state.visualizations = []
        st.session_state.dashboard_config = {}
        st.session_state.settings = {}

def check_page_access(page_name: str, user_role: Optional[str] = None) -> bool:
    """Check if user has access to the page"""
    if page_name not in PAGES:
        logger.warning(f"Attempted to access undefined page: {page_name}")
        return False
    
    if not user_role:
        return False
        
    return user_role in PAGES[page_name]["access_level"]

def get_available_pages(user_role: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Get list of pages available to the user"""
    if not user_role:
        return {}
        
    return {
        name: page_info
        for name, page_info in PAGES.items()
        if user_role in page_info["access_level"]
    }

def load_page_module(page_name: str):
    """Dynamically load page module"""
    try:
        module_path = f"pages.{page_name}"
        module = __import__(module_path, fromlist=["show_page"])
        return module.show_page
    except ImportError as e:
        logger.error(f"Error loading page module {page_name}: {str(e)}")
        return None

def get_page_title(page_name: str) -> str:
    """Get page title"""
    return PAGES.get(page_name, {}).get("title", page_name.title())

def get_page_icon(page_name: str) -> str:
    """Get page icon"""
    return PAGES.get(page_name, {}).get("icon", "📄")

def show_page_error(message: str):
    """Show error message for page loading issues"""
    st.error(f"Error: {message}")
    st.write("Please try refreshing the page or contact support if the issue persists.")

# Page layout utilities
def set_page_layout():
    """Set page layout configuration"""
    st.set_page_config(
        page_title="Data Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def show_sidebar_navigation(user_role: Optional[str] = None):
    """Show sidebar navigation"""
    st.sidebar.title("Navigation")
    
    available_pages = get_available_pages(user_role)
    sorted_pages = sorted(
        available_pages.items(),
        key=lambda x: x[1]["order"]
    )
    
    for page_name, page_info in sorted_pages:
        if st.sidebar.button(
            f"{page_info['icon']} {page_info['title']}",
            help=page_info["description"]
        ):
            st.session_state.current_page = page_name
            st.rerun()

def show_breadcrumb(page_name: str):
    """Show page breadcrumb"""
    st.markdown(
        f"### {get_page_icon(page_name)} {get_page_title(page_name)}"
    )
    st.markdown("---")

class PageManager:
    """Page management class"""
    
    @staticmethod
    def initialize():
        """Initialize page manager"""
        init_session_state()
        set_page_layout()
    
    @staticmethod
    def show_page(page_name: str, user_role: Optional[str] = None):
        """Show specified page"""
        if not check_page_access(page_name, user_role):
            show_page_error("Access denied or invalid page")
            return
            
        show_page_module = load_page_module(page_name)
        if show_page_module:
            show_breadcrumb(page_name)
            show_page_module()
        else:
            show_page_error(f"Could not load page: {page_name}")
    
    @staticmethod
    def handle_navigation(user_role: Optional[str] = None):
        """Handle page navigation"""
        show_sidebar_navigation(user_role)
        
        if st.session_state.current_page:
            PageManager.show_page(
                st.session_state.current_page,
                user_role
            )
        else:
            # Show default page (data upload)
            PageManager.show_page("data_upload", user_role)

# Version information
__version__ = "1.0.0"
__author__ = "Puwanath Baibua"
__license__ = "MIT"