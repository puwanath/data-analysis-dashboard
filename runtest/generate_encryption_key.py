from cryptography.fernet import Fernet
import base64
import os
from pathlib import Path

def generate_key():
    """Generate a new encryption key"""
    # Generate a new key
    key = Fernet.generate_key()
    
    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Save the key to file
    key_path = config_dir / "encryption.key"
    with open(key_path, 'wb') as key_file:
        key_file.write(key)
    
    # Set appropriate file permissions (for Unix-like systems)
    if os.name == 'posix':
        os.chmod(key_path, 0o600)
    
    return key

if __name__ == "__main__":
    key = generate_key()
    print("Encryption key generated successfully!")
    print("Key path: config/encryption.key")
    print("Warning: Keep this key secure and backup safely")