import streamlit as st
from typing import Dict, List, Optional
import hashlib
import jwt
import datetime
import os
import json
import logging
from cryptography.fernet import Fernet
import pandas as pd
import re

class SecurityManager:
    def __init__(self, config_path: str = "config/security.json"):
        """Initialize Security Manager"""
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self._init_security_config()
        self.fernet = Fernet(self._get_or_create_key())
        
    def _init_security_config(self):
        """Initialize security configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        if not os.path.exists(self.config_path):
            default_config = {
                'password_policy': {
                    'min_length': 8,
                    'require_uppercase': True,
                    'require_lowercase': True,
                    'require_numbers': True,
                    'require_special': True
                },
                'session': {
                    'timeout_minutes': 30,
                    'max_failed_attempts': 3,
                    'lockout_minutes': 15
                },
                'encryption': {
                    'sensitive_columns': ['email', 'phone', 'ssn', 'credit_card']
                },
                'audit': {
                    'enabled': True,
                    'log_file': 'logs/security_audit.log'
                }
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
                
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = "config/encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
            
    def check_password_strength(self, password: str) -> Dict[str, bool]:
        """Check password against security policy"""
        policy = self.config['password_policy']
        checks = {
            'length': len(password) >= policy['min_length'],
            'uppercase': bool(re.search(r'[A-Z]', password)) if policy['require_uppercase'] else True,
            'lowercase': bool(re.search(r'[a-z]', password)) if policy['require_lowercase'] else True,
            'numbers': bool(re.search(r'\d', password)) if policy['require_numbers'] else True,
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)) if policy['require_special'] else True
        }
        return checks
        
    def encrypt_sensitive_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encrypt sensitive columns in DataFrame"""
        sensitive_columns = self.config['encryption']['sensitive_columns']
        df_encrypted = df.copy()
        
        for col in df.columns:
            if col in sensitive_columns:
                df_encrypted[col] = df[col].apply(
                    lambda x: self.fernet.encrypt(str(x).encode()).decode()
                    if pd.notnull(x) else x
                )
                
        return df_encrypted
        
    def decrypt_sensitive_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decrypt sensitive columns in DataFrame"""
        sensitive_columns = self.config['encryption']['sensitive_columns']
        df_decrypted = df.copy()
        
        for col in df.columns:
            if col in sensitive_columns:
                df_decrypted[col] = df[col].apply(
                    lambda x: self.fernet.decrypt(str(x).encode()).decode()
                    if pd.notnull(x) else x
                )
                
        return df_decrypted
        
    def audit_log(self, action: str, user: str, details: str):
        """Log security-related actions"""
        if self.config['audit']['enabled']:
            timestamp = datetime.datetime.now().isoformat()
            log_entry = f"{timestamp} | {user} | {action} | {details}\n"
            
            os.makedirs(os.path.dirname(self.config['audit']['log_file']), exist_ok=True)
            with open(self.config['audit']['log_file'], 'a') as f:
                f.write(log_entry)
                
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>\'";]', '', text)
        return sanitized
        
    def mask_sensitive_data(self, text: str, pattern: str) -> str:
        """Mask sensitive data in text"""
        if pattern == 'email':
            return re.sub(r'(\w{2})\w+@', r'\1***@', text)
        elif pattern == 'phone':
            return re.sub(r'\d(?=\d{4})', '*', text)
        elif pattern == 'credit_card':
            return re.sub(r'\d(?=\d{4})', '*', text)
        return text
        
    def show_security_interface(self):
        """Show security management interface in Streamlit"""
        st.subheader("🔒 Security Management")
        
        # Security Dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Active Sessions",
                len(st.session_state.get('_active_sessions', []))
            )
            
        with col2:
            st.metric(
                "Failed Login Attempts",
                st.session_state.get('_failed_attempts', 0)
            )
            
        with col3:
            st.metric(
                "Encrypted Columns",
                len(self.config['encryption']['sensitive_columns'])
            )
            
        # Security Operations
        st.subheader("Security Operations")
        
        operation = st.selectbox(
            "Select Operation",
            ["View Audit Log", "Configure Security", "Manage Encryption"]
        )
        
        if operation == "View Audit Log":
            if os.path.exists(self.config['audit']['log_file']):
                with open(self.config['audit']['log_file'], 'r') as f:
                    logs = f.readlines()
                st.code("\n".join(logs[-50:]))  #
            elif operation == "Configure Security":
                st.subheader("Security Configuration")
                
                # Password Policy
                st.write("Password Policy")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_length = st.number_input(
                        "Minimum Password Length",
                        min_value=8,
                        value=self.config['password_policy']['min_length']
                    )
                    require_uppercase = st.checkbox(
                        "Require Uppercase",
                        value=self.config['password_policy']['require_uppercase']
                    )
                    require_lowercase = st.checkbox(
                        "Require Lowercase",
                        value=self.config['password_policy']['require_lowercase']
                    )
                    
                with col2:
                    require_numbers = st.checkbox(
                        "Require Numbers",
                        value=self.config['password_policy']['require_numbers']
                    )
                    require_special = st.checkbox(
                        "Require Special Characters",
                        value=self.config['password_policy']['require_special']
                    )
                
                # Session Settings
                st.write("Session Settings")
                timeout_minutes = st.number_input(
                    "Session Timeout (minutes)",
                    min_value=5,
                    value=self.config['session']['timeout_minutes']
                )
                max_attempts = st.number_input(
                    "Max Failed Login Attempts",
                    min_value=1,
                    value=self.config['session']['max_failed_attempts']
                )
                
                if st.button("Update Security Configuration"):
                    self.config['password_policy'].update({
                        'min_length': min_length,
                        'require_uppercase': require_uppercase,
                        'require_lowercase': require_lowercase,
                        'require_numbers': require_numbers,
                        'require_special': require_special
                    })
                    
                    self.config['session'].update({
                        'timeout_minutes': timeout_minutes,
                        'max_failed_attempts': max_attempts
                    })
                    
                    with open(self.config_path, 'w') as f:
                        json.dump(self.config, f, indent=4)
                        
                    st.success("Security configuration updated successfully!")
                    
            elif operation == "Manage Encryption":
                st.subheader("Encryption Management")
                
                # Sensitive Columns
                current_columns = self.config['encryption']['sensitive_columns']
                st.write("Current Sensitive Columns:", current_columns)
                
                new_column = st.text_input("Add New Sensitive Column")
                if new_column and st.button("Add Column"):
                    if new_column not in current_columns:
                        current_columns.append(new_column)
                        self.config['encryption']['sensitive_columns'] = current_columns
                        with open(self.config_path, 'w') as f:
                            json.dump(self.config, f, indent=4)
                        st.success(f"Added {new_column} to sensitive columns!")
                        st.rerun()
                        
                # Remove columns
                col_to_remove = st.selectbox(
                    "Select Column to Remove",
                    current_columns
                )
                if col_to_remove and st.button("Remove Column"):
                    current_columns.remove(col_to_remove)
                    self.config['encryption']['sensitive_columns'] = current_columns
                    with open(self.config_path, 'w') as f:
                        json.dump(self.config, f, indent=4)
                    st.success(f"Removed {col_to_remove} from sensitive columns!")
                    st.rerun()
                    
                # Test encryption
                st.write("Test Encryption")
                test_text = st.text_input("Enter text to encrypt")
                if test_text and st.button("Test Encryption"):
                    encrypted = self.fernet.encrypt(test_text.encode()).decode()
                    decrypted = self.fernet.decrypt(encrypted.encode()).decode()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Encrypted:", encrypted)
                    with col2:
                        st.write("Decrypted:", decrypted)

        def validate_session(self) -> bool:
            """Validate current session"""
            if 'last_activity' not in st.session_state:
                return False
                
            last_activity = st.session_state.last_activity
            timeout_minutes = self.config['session']['timeout_minutes']
            
            if datetime.datetime.now() - last_activity > datetime.timedelta(minutes=timeout_minutes):
                return False
                
            st.session_state.last_activity = datetime.datetime.now()
            return True

        def handle_failed_login(self):
            """Handle failed login attempt"""
            if '_failed_attempts' not in st.session_state:
                st.session_state._failed_attempts = 0
                
            st.session_state._failed_attempts += 1
            max_attempts = self.config['session']['max_failed_attempts']
            
            if st.session_state._failed_attempts >= max_attempts:
                st.session_state._account_locked = datetime.datetime.now()
                self.audit_log(
                    "ACCOUNT_LOCKED",
                    st.session_state.get('username', 'unknown'),
                    f"Account locked after {max_attempts} failed attempts"
                )

        def check_account_lockout(self) -> bool:
            """Check if account is locked"""
            if '_account_locked' not in st.session_state:
                return False
                
            lockout_minutes = self.config['session']['lockout_minutes']
            lockout_time = st.session_state._account_locked
            
            if datetime.datetime.now() - lockout_time > datetime.timedelta(minutes=lockout_minutes):
                del st.session_state._account_locked
                st.session_state._failed_attempts = 0
                return False
                
            return True