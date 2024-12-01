import streamlit as st
import pandas as pd
import sqlalchemy
import pymongo
import redis
import elasticsearch
from typing import Dict, List, Optional, Any
import json
import yaml
import requests
from datetime import datetime
import logging

class DataSourceIntegration:
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """Initialize Data Source Integration"""
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.connections = {}
        self.connection_pools = {}
        self._load_config()
        self._init_connection_pools()

    def _init_connection_pools(self):
        """Initialize connection pools for different data sources"""
        for db_type, config in self.config['databases'].items():
            if db_type in ['mysql', 'postgresql']:
                self.connection_pools[db_type] = sqlalchemy.pool.QueuePool(
                    lambda: self.connect_sql(db_type, config),
                    max_overflow=10,
                    pool_size=5
                )
        
    def _load_config(self):
        """Load data source configurations"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = self._create_default_config()

    def check_connections(self) -> Dict[str, bool]:
        """Check health of all configured connections"""
        status = {}
        
        for db_type, config in self.config['databases'].items():
            try:
                if db_type in ['mysql', 'postgresql']:
                    engine = self.connect_sql(db_type, config)
                    self.query_sql(engine, "SELECT 1")
                    status[db_type] = True
                elif db_type == 'mongodb':
                    client = self.connect_mongodb(config)
                    client.server_info()
                    status[db_type] = True
                elif db_type == 'elasticsearch':
                    es = self.connect_elasticsearch(config)
                    status[db_type] = es.ping()
            except Exception:
                status[db_type] = False
                
        return status
    
    def validate_data(self, data: pd.DataFrame, rules: Dict) -> Dict[str, List]:
        """Validate data against specified rules"""
        validation_results = {
            'errors': [],
            'warnings': []
        }
        
        for column, rule in rules.items():
            if column not in data.columns:
                validation_results['errors'].append(f"Column {column} not found")
                continue
                
            if 'type' in rule:
                if not all(isinstance(x, rule['type']) for x in data[column]):
                    validation_results['errors'].append(
                        f"Invalid type in column {column}"
                    )
                    
            if 'range' in rule:
                min_val, max_val = rule['range']
                if not all(min_val <= x <= max_val for x in data[column]):
                    validation_results['warnings'].append(
                        f"Values out of range in column {column}"
                    )
                    
        return validation_results
    
    def transform_data(self, data: pd.DataFrame, transformations: Dict) -> pd.DataFrame:
        """Apply transformations to data"""
        df = data.copy()
        
        for column, transforms in transformations.items():
            for transform in transforms:
                if transform == 'uppercase':
                    df[column] = df[column].str.upper()
                elif transform == 'lowercase':
                    df[column] = df[column].str.lower()
                elif transform == 'strip':
                    df[column] = df[column].str.strip()
                elif isinstance(transform, dict):
                    if 'replace' in transform:
                        df[column] = df[column].replace(transform['replace'])
                    elif 'date_format' in transform:
                        df[column] = pd.to_datetime(
                            df[column], 
                            format=transform['date_format']
                        )
                        
        return df
    
    def export_data(self, data: pd.DataFrame, format: str, path: str) -> bool:
        """Export data to various formats"""
        try:
            if format == 'csv':
                data.to_csv(path, index=False)
            elif format == 'excel':
                data.to_excel(path, index=False)
            elif format == 'json':
                data.to_json(path, orient='records')
            elif format == 'parquet':
                data.to_parquet(path, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Export error: {str(e)}")
            return False
            
    def _create_default_config(self) -> Dict:
        """Create default configuration file"""
        default_config = {
            'databases': {
                'mysql': {
                    'host': 'localhost',
                    'port': 3306,
                    'username': '',
                    'password': '',
                    'database': ''
                },
                'postgresql': {
                    'host': 'localhost',
                    'port': 5432,
                    'username': '',
                    'password': '',
                    'database': ''
                },
                'mongodb': {
                    'host': 'localhost',
                    'port': 27017,
                    'username': '',
                    'password': '',
                    'database': ''
                }
            },
            'apis': {
                'rest': {
                    'base_url': '',
                    'api_key': '',
                    'headers': {}
                },
                'graphql': {
                    'endpoint': '',
                    'headers': {}
                }
            },
            'cache': {
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'password': ''
                }
            },
            'search': {
                'elasticsearch': {
                    'host': 'localhost',
                    'port': 9200,
                    'username': '',
                    'password': ''
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        return default_config
    
    def connect_sql(self, db_type: str, config: Dict) -> sqlalchemy.engine.Engine:
        """Connect to SQL database"""
        try:
            if db_type == 'mysql':
                url = f"mysql+pymysql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
            elif db_type == 'postgresql':
                url = f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
            engine = sqlalchemy.create_engine(url)
            return engine
            
        except Exception as e:
            self.logger.error(f"SQL connection error: {str(e)}")
            raise
            
    def connect_mongodb(self, config: Dict) -> pymongo.MongoClient:
        """Connect to MongoDB"""
        try:
            client = pymongo.MongoClient(
                host=config['host'],
                port=config['port'],
                username=config['username'],
                password=config['password']
            )
            return client
            
        except Exception as e:
            self.logger.error(f"MongoDB connection error: {str(e)}")
            raise
            
    def connect_elasticsearch(self, config: Dict) -> elasticsearch.Elasticsearch:
        """Connect to Elasticsearch"""
        try:
            es = elasticsearch.Elasticsearch(
                [{'host': config['host'], 'port': config['port']}],
                http_auth=(config['username'], config['password'])
            )
            return es
            
        except Exception as e:
            self.logger.error(f"Elasticsearch connection error: {str(e)}")
            raise
            
    def connect_redis(self, config: Dict) -> redis.Redis:
        """Connect to Redis"""
        try:
            r = redis.Redis(
                host=config['host'],
                port=config['port'],
                password=config['password']
            )
            return r
            
        except Exception as e:
            self.logger.error(f"Redis connection error: {str(e)}")
            raise
            
    def query_sql(self, 
                  engine: sqlalchemy.engine.Engine,
                  query: str,
                  params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute SQL query"""
        try:
            return pd.read_sql(query, engine, params=params)
        except Exception as e:
            self.logger.error(f"SQL query error: {str(e)}")
            raise
            
    def query_mongodb(self,
                     client: pymongo.MongoClient,
                     database: str,
                     collection: str,
                     query: Dict) -> List[Dict]:
        """Query MongoDB collection"""
        try:
            db = client[database]
            coll = db[collection]
            return list(coll.find(query))
        except Exception as e:
            self.logger.error(f"MongoDB query error: {str(e)}")
            raise
            
    def search_elasticsearch(self,
                           es: elasticsearch.Elasticsearch,
                           index: str,
                           query: Dict) -> List[Dict]:
        """Search Elasticsearch index"""
        try:
            response = es.search(index=index, body=query)
            return response['hits']['hits']
        except Exception as e:
            self.logger.error(f"Elasticsearch query error: {str(e)}")
            raise
            
    def cache_data(self,
                   r: redis.Redis,
                   key: str,
                   data: Any,
                   expiry: int = 3600) -> bool:
        """Cache data in Redis"""
        try:
            return r.setex(key, expiry, json.dumps(data))
        except Exception as e:
            self.logger.error(f"Redis cache error: {str(e)}")
            raise
            
    def get_cached_data(self, r: redis.Redis, key: str) -> Optional[Any]:
        """Retrieve cached data from Redis"""
        try:
            data = r.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Redis retrieval error: {str(e)}")
            raise
            
    def api_request(self,
                    method: str,
                    url: str,
                    headers: Optional[Dict] = None,
                    params: Optional[Dict] = None,
                    data: Optional[Dict] = None) -> requests.Response:
        """Make API request"""
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=data
            )
            response.raise_for_status()
            return response
        except Exception as e:
            self.logger.error(f"API request error: {str(e)}")
            raise

    def show_data_source_interface(self):
        """Show data source management interface in Streamlit"""
        st.subheader("🔌 Data Source Integration")
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "Database Connections",
            "API Configuration",
            "Cache Settings"
        ])

        with tab1:
            st.subheader("Database Connections")
            
            # Database type selection
            db_type = st.selectbox(
                "Select Database Type",
                ["MySQL", "PostgreSQL", "MongoDB", "Elasticsearch"]
            )
            
            # Connection settings
            with st.form(f"{db_type.lower()}_connection"):
                st.write(f"{db_type} Connection Settings")
                
                host = st.text_input("Host", 
                    value=self.config['databases'][db_type.lower()]['host'])
                port = st.number_input("Port", 
                    value=self.config['databases'][db_type.lower()]['port'])
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                database = st.text_input("Database Name")
                
                test_connection = st.form_submit_button("Test Connection")
                
                if test_connection:
                    try:
                        if db_type.lower() in ['mysql', 'postgresql']:
                            engine = self.connect_sql(db_type.lower(), {
                                'host': host,
                                'port': port,
                                'username': username,
                                'password': password,
                                'database': database
                            })
                            # Test query
                            self.query_sql(engine, "SELECT 1")
                            st.success("Connection successful!")
                            
                        elif db_type.lower() == 'mongodb':
                            client = self.connect_mongodb({
                                'host': host,
                                'port': port,
                                'username': username,
                                'password': password
                            })
                            client.server_info()
                            st.success("Connection successful!")
                            
                        elif db_type.lower() == 'elasticsearch':
                            es = self.connect_elasticsearch({
                                'host': host,
                                'port': port,
                                'username': username,
                                'password': password
                            })
                            if es.ping():
                                st.success("Connection successful!")
                            else:
                                st.error("Connection failed!")
                                
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")
                        
            # Query interface
            if db_type.lower() in ['mysql', 'postgresql']:
                with st.expander("SQL Query"):
                    query = st.text_area("Enter SQL Query")
                    if st.button("Execute Query"):
                        try:
                            engine = self.connect_sql(db_type.lower(), {
                                'host': host,
                                'port': port,
                                'username': username,
                                'password': password,
                                'database': database
                            })
                            result = self.query_sql(engine, query)
                            st.dataframe(result)
                            
                        except Exception as e:
                            st.error(f"Query failed: {str(e)}")
                            
            elif db_type.lower() == 'mongodb':
                with st.expander("MongoDB Query"):
                    collection = st.text_input("Collection Name")
                    query = st.text_area("Enter Query (JSON)")
                    
                    if st.button("Execute Query"):
                        try:
                            client = self.connect_mongodb({
                                'host': host,
                                'port': port,
                                'username': username,
                                'password': password
                            })
                            result = self.query_mongodb(
                                client,
                                database,
                                collection,
                                json.loads(query)
                            )
                            st.json(result)
                            
                        except Exception as e:
                            st.error(f"Query failed: {str(e)}")
            
        with tab2:
            st.subheader("API Configuration")
            
            api_type = st.selectbox(
                "Select API Type",
                ["REST", "GraphQL"]
            )
            
            with st.form(f"{api_type.lower()}_config"):
                st.write(f"{api_type} API Configuration")
                
                if api_type == "REST":
                    base_url = st.text_input(
                        "Base URL",
                        value=self.config['apis']['rest']['base_url']
                    )
                    api_key = st.text_input(
                        "API Key",
                        type="password",
                        value=self.config['apis']['rest']['api_key']
                    )
                    
                    # Headers
                    st.write("Headers")
                    num_headers = st.number_input("Number of Headers", min_value=0, value=1)
                    headers = {}
                    
                    for i in range(num_headers):
                        col1, col2 = st.columns(2)
                        with col1:
                            key = st.text_input(f"Header {i+1} Key")
                        with col2:
                            value = st.text_input(f"Header {i+1} Value")
                        if key and value:
                            headers[key] = value
                            
                    # Test endpoint
                    test_endpoint = st.text_input("Test Endpoint")
                    test_method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
                    
                    if st.form_submit_button("Test API"):
                        try:
                            response = self.api_request(
                                test_method,
                                f"{base_url}{test_endpoint}",
                                headers=headers
                            )
                            st.success("API test successful!")
                            st.json(response.json())
                            
                        except Exception as e:
                            st.error(f"API test failed: {str(e)}")
                            
                elif api_type == "GraphQL":
                    endpoint = st.text_input(
                        "GraphQL Endpoint",
                        value=self.config['apis']['graphql']['endpoint']
                    )
                    
                    # Headers
                    st.write("Headers")
                    num_headers = st.number_input("Number of Headers", min_value=0, value=1)
                    headers = {}
                    
                    for i in range(num_headers):
                        col1, col2 = st.columns(2)
                        with col1:
                            key = st.text_input(f"Header {i+1} Key")
                        with col2:
                            value = st.text_input(f"Header {i+1} Value")
                        if key and value:
                            headers[key] = value
                            
                    # Test query
                    test_query = st.text_area("Test GraphQL Query")
                    
                    if st.form_submit_button("Test Query"):
                        try:
                            response = self.api_request(
                                "POST",
                                endpoint,
                                headers=headers,
                                data={'query': test_query}
                            )
                            st.success("Query successful!")
                            st.json(response.json())
                            
                        except Exception as e:
                            st.error(f"Query failed: {str(e)}")
                            
        with tab3:
            st.subheader("Cache Settings")
            
            # Redis configuration
            with st.form("redis_config"):
                st.write("Redis Configuration")
                
                redis_host = st.text_input(
                    "Redis Host",
                    value=self.config['cache']['redis']['host']
                )
                redis_port = st.number_input(
                    "Redis Port",
                    value=self.config['cache']['redis']['port']
                )
                redis_password = st.text_input(
                    "Redis Password",
                    type="password"
                )
                
                if st.form_submit_button("Test Redis Connection"):
                    try:
                        redis_client = self.connect_redis({
                            'host': redis_host,
                            'port': redis_port,
                            'password': redis_password
                        })
                        redis_client.ping()
                        st.success("Redis connection successful!")
                        
                    except Exception as e:
                        st.error(f"Redis connection failed: {str(e)}")
                        
            # Cache management
            with st.expander("Cache Management"):
                # Test cache operations
                key = st.text_input("Cache Key")
                value = st.text_area("Cache Value")
                expiry = st.number_input("Expiry (seconds)", value=3600)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Set Cache"):
                        try:
                            redis_client = self.connect_redis({
                                'host': redis_host,
                                'port': redis_port,
                                'password': redis_password
                            })
                            self.cache_data(redis_client, key, value, expiry)
                            st.success("Data cached successfully!")
                            
                        except Exception as e:
                            st.error(f"Cache operation failed: {str(e)}")
                            
                with col2:
                    if st.button("Get Cache"):
                        try:
                            redis_client = self.connect_redis({
                                'host': redis_host,
                                'port': redis_port,
                                'password': redis_password
                            })
                            cached_value = self.get_cached_data(redis_client, key)
                            if cached_value:
                                st.json(cached_value)
                            else:
                                st.info("No cached data found")
                                
                        except Exception as e:
                            st.error(f"Cache operation failed: {str(e)}")
                            
                with col3:
                    if st.button("Clear Cache"):
                        try:
                            redis_client = self.connect_redis({
                                'host': redis_host,
                                'port': redis_port,
                                'password': redis_password
                            })
                            redis_client.delete(key)
                            st.success("Cache cleared successfully!")
                            
                        except Exception as e:
                            st.error(f"Cache operation failed: {str(e)}")

    def save_config(self):
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)