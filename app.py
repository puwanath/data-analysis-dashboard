import streamlit as st
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Data Analysis AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide the default menu
# hide_menu = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}
# </style>
# """
# st.markdown(hide_menu, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    st.title("📊 Data Analysis AI")

    # Main page content
    st.write("""
    ## Welcome to Data Analysis AI
    
    This tool helps you analyze and visualize your data using AI-powered insights.
    
    ### Features:
    - 📤 Data Upload and Management
    - 🔍 Data Explorer
    - 📊 Interactive Visualizations
    - 🤖 AI-Powered Analysis
    
    Get started by uploading your data files in the Data Upload section.
    """)
    
    # Display app info
    st.sidebar.info("""
    ### About Data Analysis AI
    Data Analysis AI helps you:
    - Upload and manage datasets
    - Explore data patterns
    - Create visualizations
    - Get AI-powered insights
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by 9kapong.dev")
    st.sidebar.markdown("Made with ❤️ using Streamlit")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again or contact support.")
