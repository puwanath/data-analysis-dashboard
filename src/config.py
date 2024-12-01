import os
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional, List
import logging
from dotenv import load_dotenv
import streamlit as st

class Config:
    """Configuration management class"""
    _instance = None
    _config = {}
    _env_prefix = "APP_"

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration"""
        try:
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

            # Load environment variables
            load_dotenv()

            # Load all configurations
            self._load_all_configs()
            
            self.logger.info("Configuration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing configuration: {str(e)}")
            raise

    def _load_all_configs(self):
        """Load all configuration files"""
        config_dir = Path("config")

        try:
            # Load data sources config
            self._load_yaml_config(config_dir / "data_sources.yaml", "data_sources")

            # Load security config
            self._load_json_config(config_dir / "security.json", "security")

            # Load models config
            self._load_yaml_config(config_dir / "models.yaml", "models")

            # Load environment variables
            self._load_env_vars()

            # Validate configurations
            self._validate_config()
            
        except Exception as e:
            self.logger.error(f"Error loading configurations: {str(e)}")
            raise

    def _load_yaml_config(self, path: Path, key: str):
        """Load YAML configuration file"""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    self._config[key] = yaml.safe_load(f)
            else:
                self.logger.warning(f"Configuration file not found: {path}")
                self._config[key] = {}
                
        except Exception as e:
            self.logger.error(f"Error loading {path}: {str(e)}")
            self._config[key] = {}

    def _load_json_config(self, path: Path, key: str):
        """Load JSON configuration file"""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    self._config[key] = json.load(f)
            else:
                self.logger.warning(f"Configuration file not found: {path}")
                self._config[key] = {}
                
        except Exception as e:
            self.logger.error(f"Error loading {path}: {str(e)}")
            self._config[key] = {}

    def _load_env_vars(self):
        """Load environment variables with prefix"""
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix):].lower()
                env_config[config_key] = value
        self._config['env'] = env_config

    def _validate_config(self):
        """Validate configuration values"""
        required_configs = {
            'data_sources': ['storage', 'databases'],
            'security': ['auth', 'encryption'],
            'models': ['default_model']
        }

        for config_key, required_keys in required_configs.items():
            if config_key not in self._config:
                self.logger.error(f"Missing configuration section: {config_key}")
                continue

            for key in required_keys:
                if key not in self._config[config_key]:
                    self.logger.error(f"Missing required key in {config_key}: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            # Handle nested keys
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value.get(k)
                if value is None:
                    return default
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting config value for {key}: {str(e)}")
            return default

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config value for {key}: {str(e)}")
            return False

    def save(self) -> bool:
        """Save current configuration to files"""
        try:
            config_dir = Path("config")
            
            # Save data sources config
            with open(config_dir / "data_sources.yaml", 'w') as f:
                yaml.dump(self._config.get('data_sources', {}), f)
            
            # Save security config
            with open(config_dir / "security.json", 'w') as f:
                json.dump(self._config.get('security', {}), f, indent=4)
            
            # Save models config
            with open(config_dir / "models.yaml", 'w') as f:
                yaml.dump(self._config.get('models', {}), f)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configurations: {str(e)}")
            return False

    def reset(self) -> bool:
        """Reset configuration to default values"""
        try:
            self._config = {}
            self._load_all_configs()
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {str(e)}")
            return False

    def show_config_interface(self):
        """Show configuration management interface in Streamlit"""
        st.subheader("⚙️ Configuration Management")

        # Display current configuration
        if st.checkbox("Show Current Configuration"):
            st.json(self._config)

        # Configuration editor
        st.write("### Edit Configuration")
        
        config_type = st.selectbox(
            "Select Configuration Type",
            ["Data Sources", "Security", "Models"]
        )

        if config_type == "Data Sources":
            config_key = "data_sources"
            st.write("Edit Data Sources Configuration")
            
            # Database configuration
            st.write("#### Database Configuration")
            db_type = st.selectbox("Database Type", ["sqlite", "mysql", "postgresql"])
            
            if db_type == "sqlite":
                path = st.text_input(
                    "Database Path",
                    value=self.get(f"{config_key}.databases.sqlite.path", "data/app.db")
                )
                if st.button("Update SQLite Configuration"):
                    self.set(f"{config_key}.databases.sqlite.path", path)
                    self.save()
                    st.success("Configuration updated!")
                    
            elif db_type in ["mysql", "postgresql"]:
                col1, col2 = st.columns(2)
                with col1:
                    host = st.text_input(
                        "Host",
                        value=self.get(f"{config_key}.databases.{db_type}.host", "localhost")
                    )
                    username = st.text_input(
                        "Username",
                        value=self.get(f"{config_key}.databases.{db_type}.username", "")
                    )
                with col2:
                    port = st.number_input(
                        "Port",
                        value=self.get(f"{config_key}.databases.{db_type}.port", 3306 if db_type == "mysql" else 5432)
                    )
                    password = st.text_input(
                        "Password",
                        type="password",
                        value=self.get(f"{config_key}.databases.{db_type}.password", "")
                    )
                
                if st.button(f"Update {db_type.title()} Configuration"):
                    self.set(f"{config_key}.databases.{db_type}", {
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password
                    })
                    self.save()
                    st.success("Configuration updated!")

        elif config_type == "Security":
            config_key = "security"
            st.write("Edit Security Configuration")
            
            # JWT Configuration
            st.write("#### JWT Configuration")
            jwt_expiry = st.number_input(
                "Token Expiry (minutes)",
                value=self.get(f"{config_key}.auth.jwt.access_token_expire_minutes", 30)
            )
            
            # Password Policy
            st.write("#### Password Policy")
            min_length = st.number_input(
                "Minimum Password Length",
                value=self.get(f"{config_key}.auth.password_policy.min_length", 8)
            )
            require_uppercase = st.checkbox(
                "Require Uppercase",
                value=self.get(f"{config_key}.auth.password_policy.require_uppercase", True)
            )
            require_numbers = st.checkbox(
                "Require Numbers",
                value=self.get(f"{config_key}.auth.password_policy.require_numbers", True)
            )
            
            if st.button("Update Security Configuration"):
                self.set(f"{config_key}.auth.jwt.access_token_expire_minutes", jwt_expiry)
                self.set(f"{config_key}.auth.password_policy", {
                    "min_length": min_length,
                    "require_uppercase": require_uppercase,
                    "require_numbers": require_numbers
                })
                self.save()
                st.success("Configuration updated!")

        elif config_type == "Models":
            config_key = "models"
            st.write("Edit Models Configuration")
            
            # LLM Configuration
            st.write("#### LLM Configuration")
            default_model = st.selectbox(
                "Default Model",
                ["llama3.1", "mistral", "mixtral"],
                index=["llama3.1", "mistral", "mixtral"].index(
                    self.get(f"{config_key}.default_model", "llama3.1")
                )
            )
            
            if st.button("Update Models Configuration"):
                self.set(f"{config_key}.default_model", default_model)
                self.save()
                st.success("Configuration updated!")

        # Reset configuration
        st.write("### Reset Configuration")
        if st.button("Reset to Default"):
            if self.reset():
                st.success("Configuration reset to default values!")
            else:
                st.error("Error resetting configuration")

# Create global config instance
config = Config()