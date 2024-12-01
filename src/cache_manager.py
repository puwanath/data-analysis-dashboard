import streamlit as st
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
import pickle
import json
import hashlib
import os
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
import shutil
import gzip
import base64

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str = "cache/", ttl: int = 3600):
        """
        Initialize Cache Manager
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self._init_cache()

    def _init_cache(self):
        """Initialize cache directory and cleanup old files"""
        try:
            # Create cache directory if not exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Cleanup old cache files
            self.cleanup()
            
            logger.info("Cache system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing cache: {str(e)}")
            raise

    def _get_cache_key(self, key_data: Any) -> str:
        """Generate unique cache key"""
        try:
            if isinstance(key_data, pd.DataFrame):
                # For DataFrames, use shape and column names
                key_str = f"{key_data.shape}_{list(key_data.columns)}"
                
            elif isinstance(key_data, dict):
                # For dictionaries, convert to sorted string
                key_str = json.dumps(key_data, sort_keys=True)
                
            elif isinstance(key_data, (list, tuple, set)):
                # For sequences, convert to string
                key_str = str(sorted(key_data))
                
            else:
                key_str = str(key_data)
            
            # Generate MD5 hash
            return hashlib.md5(key_str.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            return None

    def _get_cache_path(self, key: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{key}.gz")

    def get(self, key_data: Any) -> Optional[Any]:
        """Retrieve data from cache"""
        try:
            key = self._get_cache_key(key_data)
            if not key:
                return None
                
            cache_path = self._get_cache_path(key)
            
            # Check if cache exists
            if not os.path.exists(cache_path):
                return None
            
            # Check if cache is expired
            if time.time() - os.path.getmtime(cache_path) > self.ttl:
                os.remove(cache_path)
                return None
            
            # Load cached data
            with gzip.open(cache_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def set(self, key_data: Any, value: Any, compress_level: int = 6) -> bool:
        """Store data in cache"""
        try:
            key = self._get_cache_key(key_data)
            if not key:
                return False
                
            cache_path = self._get_cache_path(key)
            
            # Store data with compression
            with gzip.open(cache_path, 'wb', compresslevel=compress_level) as f:
                pickle.dump(value, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")
            return False

    def delete(self, key_data: Any) -> bool:
        """Delete specific cache entry"""
        try:
            key = self._get_cache_key(key_data)
            if not key:
                return False
                
            cache_path = self._get_cache_path(key)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting cache: {str(e)}")
            return False

    def cleanup(self, max_age: Optional[int] = None) -> int:
        """Cleanup expired cache files"""
        try:
            cleaned = 0
            current_time = time.time()
            max_age = max_age or self.ttl
            
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                
                # Remove if expired
                if current_time - os.path.getmtime(file_path) > max_age:
                    os.remove(file_path)
                    cleaned += 1
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")
            return 0

    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False

    def get_cache_info(self) -> Dict:
        """Get cache statistics and information"""
        try:
            files = os.listdir(self.cache_dir)
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in files
            )
            
            cache_info = {
                'total_entries': len(files),
                'total_size_mb': total_size / (1024 * 1024),
                'oldest_entry': None,
                'newest_entry': None,
                'ttl': self.ttl
            }
            
            if files:
                timestamps = [
                    os.path.getmtime(os.path.join(self.cache_dir, f))
                    for f in files
                ]
                cache_info.update({
                    'oldest_entry': datetime.fromtimestamp(min(timestamps)),
                    'newest_entry': datetime.fromtimestamp(max(timestamps))
                })
            
            return cache_info
            
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {}

    def optimize(self, min_size_mb: float = 1.0) -> int:
        """Optimize cache by recompressing large files"""
        try:
            optimized = 0
            min_size = min_size_mb * 1024 * 1024  # Convert to bytes
            
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                
                # Check if file is large enough to optimize
                if os.path.getsize(file_path) > min_size:
                    try:
                        # Load data
                        with gzip.open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Recompress with maximum compression
                        with gzip.open(file_path, 'wb', compresslevel=9) as f:
                            pickle.dump(data, f)
                        
                        optimized += 1
                        
                    except Exception as e:
                        logger.warning(f"Error optimizing {filename}: {str(e)}")
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {str(e)}")
            return 0

    def show_cache_interface(self):
        """Show cache management interface in Streamlit"""
        st.subheader("📦 Cache Management")
        
        # Display cache statistics
        cache_info = self.get_cache_info()
        
        if cache_info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Cache Entries",
                    cache_info['total_entries']
                )
            
            with col2:
                st.metric(
                    "Cache Size",
                    f"{cache_info['total_size_mb']:.2f} MB"
                )
            
            with col3:
                if cache_info['newest_entry']:
                    st.metric(
                        "Last Updated",
                        cache_info['newest_entry'].strftime('%Y-%m-%d %H:%M:%S')
                    )
        
        # Cache operations
        st.write("### Cache Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache"):
                if self.clear():
                    st.success("Cache cleared successfully!")
                    st.rerun()
                else:
                    st.error("Error clearing cache")
        
        with col2:
            if st.button("Optimize Cache"):
                optimized = self.optimize()
                if optimized > 0:
                    st.success(f"Optimized {optimized} cache entries!")
                else:
                    st.info("No entries required optimization")

def cache_data(ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_mgr = CacheManager(ttl=ttl)
            
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            
            # Try to get from cache
            result = cache_mgr.get(key_data)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_mgr.set(key_data, result)
            return result
            
        return wrapper
    return decorator