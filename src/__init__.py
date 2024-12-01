import logging
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Version information
__version__ = "1.0.0"
__author__ = "Puwanath Baibua"
__license__ = "MIT"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    """Configuration management class"""
    _instance = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load all configuration files"""
        config_dir = Path("config")
        
        try:
            # Load data sources config
            with open(config_dir / "data_sources.yaml") as f:
                self._config['data_sources'] = yaml.safe_load(f)

            # Load security config
            with open(config_dir / "security.json") as f:
                self._config['security'] = yaml.safe_load(f)

            # Load models config
            with open(config_dir / "models.yaml") as f:
                self._config['models'] = yaml.safe_load(f)

            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def get(self, key: str) -> Any:
        """Get configuration value"""
        return self._config.get(key)

class AppState:
    """Application state management class"""
    _instance = None
    _state = {
        'initialized': False,
        'user': None,
        'uploaded_files': {},
        'analysis_results': {},
        'visualizations': [],
        'dashboard_config': {},
        'settings': {}
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppState, cls).__new__(cls)
        return cls._instance

    def get(self, key: str) -> Any:
        """Get state value"""
        return self._state.get(key)

    def set(self, key: str, value: Any):
        """Set state value"""
        self._state[key] = value

    def clear(self):
        """Clear application state"""
        self._state = {
            'initialized': False,
            'user': None,
            'uploaded_files': {},
            'analysis_results': {},
            'visualizations': [],
            'dashboard_config': {},
            'settings': {}
        }

# Create required directories
def init_directories():
    """Initialize required directories"""
    directories = [
        'data',
        'logs',
        'cache',
        'models',
        'reports',
        'config',
        'translations',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory created/verified: {directory}")

# Initialize application components
def init_app():
    """Initialize application components"""
    try:
        # Create directories
        init_directories()
        
        # Load configuration
        config = Config()
        
        # Initialize application state
        app_state = AppState()
        
        # Set initialization flag
        app_state.set('initialized', True)
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        raise

# Export components
__all__ = [
    'Config',
    'AppState',
    'init_app',
    'logger'
]

# Constants
DEFAULT_LANGUAGE = 'en'
SUPPORTED_LANGUAGES = ['en', 'th', 'ja', 'zh']

SUPPORTED_FILE_TYPES = [
    'csv',
    'xlsx',
    'xls',
    'json',
    'parquet'
]

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

CACHE_TTL = 3600  # 1 hour

# Initialize application on module import
init_app()