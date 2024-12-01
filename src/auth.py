import streamlit as st
import sqlite3
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import re
import yaml
import logging
from pathlib import Path
import json
import secrets
import base64

logger = logging.getLogger(__name__)

class AuthSystem:
    def __init__(self, db_path: str = "data/users.db", config_path: str = "config/security.json"):
        """Initialize Authentication System"""
        self.db_path = db_path
        self.config_path = config_path
        self._init_system()

    def _init_system(self):
        """Initialize the authentication system"""
        try:
            # Create data directory if not exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Load security configuration
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Initialize database
            self._init_db()
            
            # Set JWT secret
            self.jwt_secret = self.config['auth']['jwt']['secret_key']
            if not self.jwt_secret:
                self.jwt_secret = secrets.token_hex(32)
                self.config['auth']['jwt']['secret_key'] = self.jwt_secret
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            
            logger.info("Authentication system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing auth system: {str(e)}")
            raise

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                password_changed_at TIMESTAMP,
                previous_passwords TEXT
            )
        ''')
        
        # Create sessions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create activity log table
        c.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create default admin user if not exists
        c.execute("SELECT * FROM users WHERE username = ?", ("admin",))
        if not c.fetchone():
            hashed_password = self._hash_password("admin")
            c.execute("""
                INSERT INTO users (username, password, role, email)
                VALUES (?, ?, ?, ?)
            """, ("admin", hashed_password, "admin", "admin@example.com"))
        
        conn.commit()
        conn.close()

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(8)
        return f"{salt}${hashlib.sha256((password + salt).encode()).hexdigest()}"

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hashed value"""
        try:
            salt, hash_value = hashed.split('$')
            return hash_value == hashlib.sha256((password + salt).encode()).hexdigest()
        except:
            return False

    def _check_password_requirements(self, password: str) -> Tuple[bool, str]:
        """Check if password meets requirements"""
        policy = self.config['auth']['password_policy']
        
        if len(password) < policy['min_length']:
            return False, f"Password must be at least {policy['min_length']} characters long"
            
        if policy['require_uppercase'] and not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
            
        if policy['require_lowercase'] and not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
            
        if policy['require_numbers'] and not re.search(r'\d', password):
            return False, "Password must contain at least one number"
            
        if policy['require_special'] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
            
        return True, "Password meets requirements"

    def _generate_token(self, user_id: int, username: str, role: str) -> str:
        """Generate JWT token"""
        expiry = datetime.utcnow() + timedelta(
            minutes=self.config['auth']['jwt']['access_token_expire_minutes']
        )
        
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'exp': expiry
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def _verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
        except:
            return None

    def register(self, username: str, password: str, email: str, role: str = 'user') -> Tuple[bool, str]:
        """Register new user"""
        try:
            # Validate password
            is_valid, msg = self._check_password_requirements(password)
            if not is_valid:
                return False, msg
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if username exists
            c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
            if c.fetchone():
                return False, "Username already exists"
            
            # Check if email exists
            c.execute("SELECT 1 FROM users WHERE email = ?", (email,))
            if c.fetchone():
                return False, "Email already exists"
            
            # Create user
            hashed_password = self._hash_password(password)
            c.execute("""
                INSERT INTO users (username, password, email, role, password_changed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (username, hashed_password, email, role, datetime.utcnow()))
            
            user_id = c.lastrowid
            
            # Log activity
            c.execute("""
                INSERT INTO activity_log (user_id, action, details)
                VALUES (?, ?, ?)
            """, (user_id, "REGISTER", "User registration"))
            
            conn.commit()
            conn.close()
            
            return True, "Registration successful"
            
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False, "Registration failed"

    def login(self, username: str, password: str, ip_address: Optional[str] = None) -> Tuple[bool, str, Optional[Dict]]:
        """Handle user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Get user
            c.execute("""
                SELECT id, username, password, role, failed_attempts, locked_until, is_active
                FROM users WHERE username = ?
            """, (username,))
            user = c.fetchone()
            
            if not user:
                return False, "Invalid username or password", None
                
            user_id, username, hashed_pass, role, failed_attempts, locked_until, is_active = user
            
            # Check if account is active
            if not is_active:
                return False, "Account is deactivated", None
            
            # Check if account is locked
            if locked_until and datetime.fromisoformat(locked_until) > datetime.utcnow():
                return False, f"Account is locked until {locked_until}", None
            
            # Verify password
            if not self._verify_password(password, hashed_pass):
                # Increment failed attempts
                failed_attempts += 1
                if failed_attempts >= self.config['auth']['session']['max_failed_attempts']:
                    locked_until = datetime.utcnow() + timedelta(
                        minutes=self.config['auth']['session']['lockout_minutes']
                    )
                    c.execute("""
                        UPDATE users 
                        SET failed_attempts = ?, locked_until = ?
                        WHERE id = ?
                    """, (failed_attempts, locked_until, user_id))
                else:
                    c.execute("""
                        UPDATE users 
                        SET failed_attempts = ?
                        WHERE id = ?
                    """, (failed_attempts, user_id))
                
                conn.commit()
                return False, "Invalid username or password", None
            
            # Login successful - reset failed attempts and update last login
            token = self._generate_token(user_id, username, role)
            
            c.execute("""
                UPDATE users 
                SET failed_attempts = 0,
                    locked_until = NULL,
                    last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_id,))
            
            # Create session
            c.execute("""
                INSERT INTO sessions (user_id, token, expires_at)
                VALUES (?, ?, ?)
            """, (
                user_id,
                token,
                datetime.utcnow() + timedelta(
                    minutes=self.config['auth']['jwt']['access_token_expire_minutes']
                )
            ))
            
            # Log activity
            c.execute("""
                INSERT INTO activity_log (user_id, action, details, ip_address)
                VALUES (?, ?, ?, ?)
            """, (user_id, "LOGIN", "User login", ip_address))
            
            conn.commit()
            conn.close()
            
            return True, "Login successful", {
                'user_id': user_id,
                'username': username,
                'role': role,
                'token': token
            }
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return False, "Login failed", None

    def logout(self, token: str):
        """Handle user logout"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Deactivate session
            c.execute("""
                UPDATE sessions 
                SET is_active = 0
                WHERE token = ?
            """, (token,))
            
            # Get user_id from token
            token_data = self._verify_token(token)
            if token_data:
                # Log activity
                c.execute("""
                    INSERT INTO activity_log (user_id, action, details)
                    VALUES (?, ?, ?)
                """, (token_data['user_id'], "LOGOUT", "User logout"))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")

    def change_password(self, user_id: int, current_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        try:
            # Validate new password
            is_valid, msg = self._check_password_requirements(new_password)
            if not is_valid:
                return False, msg
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Get current password
            c.execute("SELECT password, previous_passwords FROM users WHERE id = ?", (user_id,))
            current_hash, prev_passwords = c.fetchone()
            
            # Verify current password
            if not self._verify_password(current_password, current_hash):
                return False, "Current password is incorrect"
            
            # Check password history
            if self.config['auth']['password_policy']['prevent_reuse']:
                prev_list = json.loads(prev_passwords) if prev_passwords else []
                for prev_hash in prev_list:
                    if self._verify_password(new_password, prev_hash):
                        return False, "Cannot reuse previous passwords"
            
            # Update password
            new_hash = self._hash_password(new_password)
            prev_list = json.loads(prev_passwords) if prev_passwords else []
            prev_list.append(current_hash)
            
            # Keep only recent passwords based on config
            max_previous = self.config['auth']['password_policy']['previous_passwords_count']
            if len(prev_list) > max_previous:
                prev_list = prev_list[-max_previous:]
            
            c.execute("""
                UPDATE users 
                SET password = ?,
                    previous_passwords = ?,
                    password_changed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_hash, json.dumps(prev_list), user_id))
            
            # Log activity
            c.execute("""
                INSERT INTO activity_log (user_id, action, details)
                VALUES (?, ?, ?)
            """, (user_id, "CHANGE_PASSWORD", "Password changed"))
            
            conn.commit()
            conn.close()
            
            return True, "Password changed successfully"
            
        except Exception as e:
            logger.error(f"Password change error: {str(e)}")
            return False, "Password change failed"

    def reset_password(self, email: str) -> Tuple[bool, str]:
        """Reset user password"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Get user
            c.execute("SELECT id, username FROM users WHERE email = ?", (email,))
            user = c.fetchone()
            
            if not user:
                return False, "Email not found"
            
            user_id, username = user
            
            # Generate temporary password
            temp_password = secrets.token_urlsafe(12)
            hashed_password = self._hash_password(temp_password)
            
            c.execute("""
                UPDATE users 
                SET password = ?,
                    password_changed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (hashed_password, user_id))
            
            # Log activity
            c.execute("""
                INSERT INTO activity_log (user_id, action, details)
                VALUES (?, ?, ?)
            """, (user_id, "RESET_PASSWORD", "Password reset"))
            
            conn.commit()
            conn.close()
            
            # TODO: Send email with temporary password
            logger.info(f"Temporary password for {username}: {temp_password}")
            
            return True, "Password reset successful"
            
        except Exception as e:
            logger.error(f"Password reset error: {str(e)}")
            return False, "Password reset failed"
        

    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """Get user information"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT id, username, email, role, created_at, last_login,
                       is_active, failed_attempts, locked_until, password_changed_at
                FROM users WHERE id = ?
            """, (user_id,))
            
            user = c.fetchone()
            if user:
                return {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'role': user[3],
                    'created_at': user[4],
                    'last_login': user[5],
                    'is_active': user[6],
                    'failed_attempts': user[7],
                    'locked_until': user[8],
                    'password_changed_at': user[9]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting user info: {str(e)}")
            return None
        finally:
            conn.close()

    def update_user(self, user_id: int, data: Dict) -> Tuple[bool, str]:
        """Update user information"""
        try:
            allowed_fields = {'email', 'role', 'is_active'}
            update_fields = {k: v for k, v in data.items() if k in allowed_fields}
            
            if not update_fields:
                return False, "No valid fields to update"
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Build update query
            query = "UPDATE users SET " + ", ".join(f"{k} = ?" for k in update_fields)
            query += " WHERE id = ?"
            
            c.execute(query, list(update_fields.values()) + [user_id])
            
            # Log activity
            c.execute("""
                INSERT INTO activity_log (user_id, action, details)
                VALUES (?, ?, ?)
            """, (user_id, "UPDATE_USER", f"Updated fields: {', '.join(update_fields.keys())}"))
            
            conn.commit()
            conn.close()
            
            return True, "User updated successfully"
            
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return False, "Update failed"

    def delete_user(self, user_id: int) -> Tuple[bool, str]:
        """Delete user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT role FROM users WHERE id = ?", (user_id,))
            user = c.fetchone()
            
            if not user:
                return False, "User not found"
            
            if user[0] == 'admin':
                return False, "Cannot delete admin user"
            
            # Delete user's sessions
            c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            
            # Log activity before deletion
            c.execute("""
                INSERT INTO activity_log (user_id, action, details)
                VALUES (?, ?, ?)
            """, (user_id, "DELETE_USER", "User account deleted"))
            
            # Delete user
            c.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            conn.commit()
            conn.close()
            
            return True, "User deleted successfully"
            
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False, "Deletion failed"

    def get_active_sessions(self, user_id: int) -> List[Dict]:
        """Get user's active sessions"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT id, token, created_at, expires_at
                FROM sessions
                WHERE user_id = ? AND is_active = 1 AND expires_at > CURRENT_TIMESTAMP
            """, (user_id,))
            
            sessions = []
            for session in c.fetchall():
                sessions.append({
                    'id': session[0],
                    'token': session[1],
                    'created_at': session[2],
                    'expires_at': session[3]
                })
                
            conn.close()
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting sessions: {str(e)}")
            return []

    def revoke_session(self, session_id: int) -> bool:
        """Revoke a specific session"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                UPDATE sessions 
                SET is_active = 0
                WHERE id = ?
            """, (session_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking session: {str(e)}")
            return False

    def get_user_activity(self, user_id: int, limit: int = 100) -> List[Dict]:
        """Get user activity history"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT action, details, ip_address, timestamp
                FROM activity_log
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            activities = []
            for activity in c.fetchall():
                activities.append({
                    'action': activity[0],
                    'details': activity[1],
                    'ip_address': activity[2],
                    'timestamp': activity[3]
                })
                
            conn.close()
            return activities
            
        except Exception as e:
            logger.error(f"Error getting user activity: {str(e)}")
            return []

    def verify_security_questions(self, user_id: int, answers: Dict[str, str]) -> bool:
        """Verify security question answers"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT security_questions
                FROM users
                WHERE id = ?
            """, (user_id,))
            
            stored_questions = c.fetchone()
            if not stored_questions or not stored_questions[0]:
                return False
            
            stored_data = json.loads(stored_questions[0])
            for question, stored_answer in stored_data.items():
                if question in answers:
                    hashed_answer = self._hash_password(answers[question])
                    if not self._verify_password(answers[question], stored_answer):
                        return False
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying security questions: {str(e)}")
            return False
        finally:
            conn.close()

    def require_password_change(self, user_id: int) -> bool:
        """Force user to change password on next login"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                UPDATE users
                SET password_changed_at = NULL
                WHERE id = ?
            """, (user_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error requiring password change: {str(e)}")
            return False

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                UPDATE sessions
                SET is_active = 0
                WHERE expires_at < CURRENT_TIMESTAMP
                AND is_active = 1
            """)
            
            affected = c.rowcount
            conn.commit()
            conn.close()
            
            return affected
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {str(e)}")
            return 0

    def show_login_form(self):
        """Display login form in Streamlit"""
        st.subheader("🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Login")
            with col2:
                if st.form_submit_button("Forgot Password?"):
                    st.session_state.show_reset = True
            
            if submit:
                success, message, user_data = self.login(
                    username, 
                    password,
                    st.session_state.get('client_ip')
                )
                
                if success:
                    st.session_state.user = user_data
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # Password reset form
        if st.session_state.get('show_reset'):
            with st.form("reset_form"):
                st.subheader("Reset Password")
                email = st.text_input("Email")
                if st.form_submit_button("Reset Password"):
                    success, message = self.reset_password(email)
                    if success:
                        st.success(message)
                        st.session_state.show_reset = False
                    else:
                        st.error(message)

    def show_register_form(self):
        """Display registration form in Streamlit"""
        st.subheader("📝 Register")
        
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Register"):
                if password != confirm_password:
                    st.error("Passwords don't match!")
                else:
                    success, message = self.register(username, password, email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

if __name__ == "__main__":
    auth = AuthSystem()