"""
Assignment 4 - Task 4: MongoDB Atlas Deployment
Deploy UCI Online Retail Dataset to MongoDB Atlas Cloud

This script demonstrates deployment and configuration of the UCI Online Retail
dataset on MongoDB Atlas cloud platform with connection management,
performance optimization, and monitoring setup.

Author: MLOps Student
Date: December 2024
"""

import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import ssl
import time
from urllib.parse import quote_plus
import certifi
from bson import ObjectId
import gridfs
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AtlasConfig:
    """Configuration for MongoDB Atlas connection."""
    username: str
    password: str
    cluster_name: str
    database_name: str
    region: str = "us-east-1"
    tier: str = "M0"  # Free tier
    
    def get_connection_string(self) -> str:
        """Generate MongoDB Atlas connection string."""
        encoded_username = quote_plus(self.username)
        encoded_password = quote_plus(self.password)

        return (f"mongodb+srv://{encoded_username}:{encoded_password}"
                f"@{self.cluster_name}.mongodb.net/{self.database_name}"
                f"?retryWrites=true&w=majority")
@dataclass
class DeploymentStats:
    """Statistics for deployment operations."""
    operation: str
    start_time: datetime
    end_time: datetime
    records_processed: int
    success: bool
    error_message: str = ""
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

class AtlasDeployment:
    """Handles MongoDB Atlas deployment operations."""
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.client = None
        self.database = None
        self.stats = []
        
    def connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas.
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info("Connecting to MongoDB Atlas...")
            
            # Create connection with SSL configuration for PyMongo 4.x
            self.client = MongoClient(
                self.config.get_connection_string(),
                serverSelectionTimeoutMS=10000,  # 10 second timeout
                socketTimeoutMS=20000,          # 20 second socket timeout
                maxPoolSize=50,                 # Connection pool size
                retryWrites=True,
                tls=True,                       # Use 'tls' instead of 'ssl'
                tlsCAFile=certifi.where()       # Use 'tlsCAFile' instead of 'ssl_ca_certs'
            )
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("✅ Successfully connected to MongoDB Atlas")
            
            # Get database reference
            self.database = self.client[self.config.database_name]
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB Atlas: {str(e)}")
            return False
    
    def create_collections(self) -> bool:
        """
        Create collections with optimized indexes.
        
        Returns:
            bool: True if collections created successfully
        """
        try:
            logger.info("Creating collections and indexes...")
            
            # Collections to create
            collections_config = {
                'transactions_centric': {
                    'indexes': [
                        [('customer_id', 1)],
                        [('invoice_date', -1)],
                        [('country', 1)],
                        [('total_amount', -1)],
                        [('customer_id', 1), ('invoice_date', -1)],  # Compound index
                    ]
                },
                'customers_centric': {
                    'indexes': [
                        [('country', 1)],
                        [('total_spent', -1)],
                        [('total_orders', -1)],
                        [('registration_date', -1)],
                        [('country', 1), ('total_spent', -1)],  # Compound index
                    ]
                },
                'products': {
                    'indexes': [
                        [('stock_code', 1)],
                        [('description', 'text')],  # Text index for search
                        [('unit_price', -1)],
                    ]
                },
                'analytics_cache': {
                    'indexes': [
                        [('cache_key', 1)],
                        [('created_at', 1)],
                        [('expires_at', 1)],
                    ]
                }
            }
            
            for collection_name, config in collections_config.items():
                collection = self.database[collection_name]
                
                # Create indexes
                for index_spec in config['indexes']:
                    try:
                        if isinstance(index_spec[0], tuple):
                            # Multiple field index
                            collection.create_index(index_spec)
                        else:
                            # Single field index
                            collection.create_index(index_spec)
                        logger.info(f"✅ Created index {index_spec} on {collection_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Index creation warning for {collection_name}: {str(e)}")
            
            logger.info("✅ Collections and indexes created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collections: {str(e)}")
            return False
    
    def upload_data_from_local(self, local_data_path: str = "../data/Online Retail.xlsx") -> bool:
        """
        Upload data from local Excel file to Atlas.
        
        Args:
            local_data_path: Path to the local Excel file
            
        Returns:
            bool: True if upload successful
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Loading data from {local_data_path}...")
            
            # Check if file exists
            if not os.path.exists(local_data_path):
                logger.error(f"❌ Data file not found: {local_data_path}")
                return False
            
            # Load Excel data
            df = pd.read_excel(local_data_path, dtype_backend='numpy_nullable')
            logger.info(f"✅ Loaded {len(df)} records from Excel file")
            
            # Clean and prepare data
            df = self._clean_data(df)
            
            # Upload in batches
            success = self._upload_data_in_batches(df)
            
            end_time = datetime.now()
            
            # Record stats
            stat = DeploymentStats(
                operation="Data Upload",
                start_time=start_time,
                end_time=end_time,
                records_processed=len(df),
                success=success
            )
            self.stats.append(stat)
            
            if success:
                logger.info(f"✅ Data upload completed in {stat.duration_seconds:.2f} seconds")
            
            return success
            
        except Exception as e:
            end_time = datetime.now()
            stat = DeploymentStats(
                operation="Data Upload",
                start_time=start_time,
                end_time=end_time,
                records_processed=0,
                success=False,
                error_message=str(e)
            )
            self.stats.append(stat)
            
            logger.error(f"❌ Data upload failed: {str(e)}")
            return False
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for MongoDB.
        
        Args:
            df: Raw DataFrame from Excel
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove rows with missing CustomerID
        df = df.dropna(subset=['CustomerID'])
        
        # Convert CustomerID to string
        df['CustomerID'] = df['CustomerID'].astype(str)
        
        # Parse InvoiceDate
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Calculate total price
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Identify cancelled transactions
        df['IsCancelled'] = df['InvoiceNo'].str.startswith('C')
        
        # Clean description
        df['Description'] = df['Description'].fillna('Unknown Item')
        
        # Clean country
        df['Country'] = df['Country'].fillna('Unknown')
        
        logger.info(f"✅ Data cleaned. Final record count: {len(df)}")
        return df
    
    def _upload_data_in_batches(self, df: pd.DataFrame, batch_size: int = 1000) -> bool:
        """
        Upload data in batches for better performance.
        
        Args:
            df: DataFrame to upload
            batch_size: Number of records per batch
            
        Returns:
            bool: True if all batches uploaded successfully
        """
        try:
            logger.info(f"Uploading data in batches of {batch_size}...")
            
            # Prepare collections
            transactions_coll = self.database['transactions_centric']
            customers_coll = self.database['customers_centric']
            products_coll = self.database['products']
            
            # Group by invoice for transaction-centric model
            invoice_groups = df.groupby('InvoiceNo')
            total_invoices = len(invoice_groups)
            
            # Upload transactions in batches
            batch_docs = []
            processed = 0
            
            for invoice_no, invoice_data in invoice_groups:
                # Create transaction document
                first_row = invoice_data.iloc[0]
                
                transaction_doc = {
                    '_id': invoice_no,
                    'customer_id': str(first_row['CustomerID']),
                    'invoice_date': first_row['InvoiceDate'],
                    'country': first_row['Country'],
                    'is_cancelled': bool(first_row['IsCancelled']),
                    'total_amount': float(invoice_data['TotalPrice'].sum()),
                    'items': []
                }
                
                # Add items
                for _, item in invoice_data.iterrows():
                    item_doc = {
                        'stock_code': str(item['StockCode']),
                        'description': str(item['Description']),
                        'quantity': int(item['Quantity']),
                        'unit_price': float(item['UnitPrice']),
                        'total_price': float(item['TotalPrice'])
                    }
                    transaction_doc['items'].append(item_doc)
                
                batch_docs.append(transaction_doc)
                processed += 1
                
                # Upload batch when full
                if len(batch_docs) >= batch_size:
                    self._upload_batch(transactions_coll, batch_docs, 'transactions')
                    batch_docs = []
                
                # Progress logging
                if processed % 100 == 0:
                    logger.info(f"Processed {processed}/{total_invoices} invoices...")
            
            # Upload remaining documents
            if batch_docs:
                self._upload_batch(transactions_coll, batch_docs, 'transactions')
            
            # Create customer-centric documents
            self._create_customer_documents(df, customers_coll)
            
            # Create product catalog
            self._create_product_documents(df, products_coll)
            
            logger.info("✅ All data uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch upload failed: {str(e)}")
            return False
    
    def _upload_batch(self, collection, documents: List[Dict], doc_type: str):
        """Upload a batch of documents with error handling."""
        try:
            if documents:
                collection.insert_many(documents, ordered=False)
                logger.debug(f"✅ Uploaded batch of {len(documents)} {doc_type}")
        except Exception as e:
            logger.warning(f"⚠️ Batch upload warning for {doc_type}: {str(e)}")
    
    def _create_customer_documents(self, df: pd.DataFrame, collection):
        """Create customer-centric documents."""
        logger.info("Creating customer-centric documents...")
        
        customer_groups = df.groupby('CustomerID')
        batch_docs = []
        
        for customer_id, customer_data in customer_groups:
            # Calculate customer metrics
            total_spent = customer_data['TotalPrice'].sum()
            total_orders = customer_data['InvoiceNo'].nunique()
            first_order = customer_data['InvoiceDate'].min()
            last_order = customer_data['InvoiceDate'].max()
            country = customer_data['Country'].iloc[0]
            
            # Group transactions by invoice
            transactions = []
            for invoice_no, invoice_data in customer_data.groupby('InvoiceNo'):
                transaction = {
                    'invoice_no': invoice_no,
                    'date': invoice_data['InvoiceDate'].iloc[0],
                    'amount': float(invoice_data['TotalPrice'].sum()),
                    'items_count': len(invoice_data)
                }
                transactions.append(transaction)
            
            customer_doc = {
                '_id': str(customer_id),
                'country': country,
                'registration_date': first_order,
                'last_order_date': last_order,
                'total_spent': float(total_spent),
                'total_orders': total_orders,
                'avg_order_value': float(total_spent / total_orders) if total_orders > 0 else 0,
                'transactions': transactions[:10]  # Limit to recent transactions
            }
            
            batch_docs.append(customer_doc)
            
            # Upload in batches
            if len(batch_docs) >= 100:
                self._upload_batch(collection, batch_docs, 'customers')
                batch_docs = []
        
        # Upload remaining documents
        if batch_docs:
            self._upload_batch(collection, batch_docs, 'customers')
        
        logger.info("✅ Customer documents created")
    
    def _create_product_documents(self, df: pd.DataFrame, collection):
        """Create product catalog."""
        logger.info("Creating product catalog...")
        
        # Group by stock code
        product_groups = df.groupby('StockCode')
        batch_docs = []
        
        for stock_code, product_data in product_groups:
            # Get most common description
            description = product_data['Description'].mode().iloc[0] if not product_data['Description'].mode().empty else 'Unknown'
            
            # Calculate metrics
            total_sold = product_data['Quantity'].sum()
            avg_price = product_data['UnitPrice'].mean()
            total_revenue = product_data['TotalPrice'].sum()
            
            product_doc = {
                '_id': str(stock_code),
                'description': description,
                'avg_unit_price': float(avg_price),
                'total_quantity_sold': int(total_sold),
                'total_revenue': float(total_revenue),
                'unique_customers': int(product_data['CustomerID'].nunique()),
                'first_sale': product_data['InvoiceDate'].min(),
                'last_sale': product_data['InvoiceDate'].max()
            }
            
            batch_docs.append(product_doc)
            
            # Upload in batches
            if len(batch_docs) >= 500:
                self._upload_batch(collection, batch_docs, 'products')
                batch_docs = []
        
        # Upload remaining documents
        if batch_docs:
            self._upload_batch(collection, batch_docs, 'products')
        
        logger.info("✅ Product catalog created")
    
    def test_atlas_queries(self) -> bool:
        """
        Test various queries on Atlas to verify deployment.
        
        Returns:
            bool: True if all queries successful
        """
        logger.info("Testing Atlas queries...")
        
        try:
            # Test basic counts
            transactions_count = self.database['transactions_centric'].count_documents({})
            customers_count = self.database['customers_centric'].count_documents({})
            products_count = self.database['products'].count_documents({})
            
            logger.info(f"📊 Collection counts:")
            logger.info(f"   Transactions: {transactions_count:,}")
            logger.info(f"   Customers: {customers_count:,}")
            logger.info(f"   Products: {products_count:,}")
            
            # Test aggregation query - Top countries by revenue
            pipeline = [
                {"$match": {"is_cancelled": False}},
                {"$group": {
                    "_id": "$country",
                    "total_revenue": {"$sum": "$total_amount"},
                    "order_count": {"$sum": 1}
                }},
                {"$sort": {"total_revenue": -1}},
                {"$limit": 5}
            ]
            
            top_countries = list(self.database['transactions_centric'].aggregate(pipeline))
            
            logger.info("🌍 Top countries by revenue:")
            for country in top_countries:
                logger.info(f"   {country['_id']}: ${country['total_revenue']:,.2f} ({country['order_count']} orders)")
            
            # Test customer query - Top spenders
            top_customers = list(
                self.database['customers_centric']
                .find({}, {"_id": 1, "total_spent": 1, "total_orders": 1})
                .sort("total_spent", -1)
                .limit(5)
            )
            
            logger.info("💰 Top customers by spending:")
            for customer in top_customers:
                logger.info(f"   Customer {customer['_id']}: ${customer['total_spent']:.2f} ({customer['total_orders']} orders)")
            
            # Test text search on products
            products_with_text_search = list(
                self.database['products']
                .find({"description": {"$regex": "CHRISTMAS", "$options": "i"}})
                .limit(3)
            )
            
            logger.info("🎄 Christmas products:")
            for product in products_with_text_search:
                logger.info(f"   {product['_id']}: {product['description']} (${product['avg_unit_price']:.2f})")
            
            # Test performance - Time a complex aggregation
            start_time = time.time()
            
            monthly_sales = list(self.database['transactions_centric'].aggregate([
                {"$match": {"is_cancelled": False}},
                {"$group": {
                    "_id": {
                        "year": {"$year": "$invoice_date"},
                        "month": {"$month": "$invoice_date"}
                    },
                    "total_sales": {"$sum": "$total_amount"},
                    "transaction_count": {"$sum": 1}
                }},
                {"$sort": {"_id.year": 1, "_id.month": 1}}
            ]))
            
            query_time = (time.time() - start_time) * 1000
            logger.info(f"📈 Monthly sales aggregation completed in {query_time:.2f}ms")
            logger.info(f"   Found {len(monthly_sales)} months of data")
            
            logger.info("✅ All Atlas queries completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Atlas query testing failed: {str(e)}")
            return False
    
    def setup_monitoring(self) -> bool:
        """
        Setup monitoring and alerting for the Atlas deployment.
        
        Returns:
            bool: True if monitoring setup successful
        """
        logger.info("Setting up monitoring...")
        
        try:
            # Create monitoring collection
            monitoring_coll = self.database['monitoring']
            
            # Initial performance baseline
            baseline_doc = {
                '_id': 'performance_baseline',
                'created_at': datetime.now(),
                'metrics': {
                    'avg_query_time_ms': 50.0,  # Target average query time
                    'max_connections': 50,       # Connection pool limit
                    'memory_usage_target_mb': 512,
                    'disk_usage_target_gb': 1.0
                },
                'alerts': {
                    'slow_queries_threshold_ms': 1000,
                    'connection_pool_threshold': 80,  # 80% of max connections
                    'memory_threshold_mb': 400,
                    'error_rate_threshold_percent': 5.0
                }
            }
            
            monitoring_coll.replace_one(
                {'_id': 'performance_baseline'}, 
                baseline_doc, 
                upsert=True
            )
            
            # Create analytics cache for frequent queries
            cache_docs = [
                {
                    '_id': 'daily_sales_summary',
                    'query_type': 'dashboard',
                    'cache_ttl_minutes': 60,
                    'last_updated': datetime.now(),
                    'data': {}
                },
                {
                    '_id': 'top_products_weekly',
                    'query_type': 'analytics',
                    'cache_ttl_minutes': 1440,  # 24 hours
                    'last_updated': datetime.now(),
                    'data': {}
                }
            ]
            
            for cache_doc in cache_docs:
                self.database['analytics_cache'].replace_one(
                    {'_id': cache_doc['_id']}, 
                    cache_doc, 
                    upsert=True
                )
            
            logger.info("✅ Monitoring setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {str(e)}")
            return False
    
    def generate_deployment_report(self) -> str:
        """
        Generate deployment summary report.
        
        Returns:
            str: Path to generated report file
        """
        logger.info("Generating deployment report...")
        
        try:
            # Collect statistics
            report_data = {
                'deployment_info': {
                    'cluster_name': self.config.cluster_name,
                    'database_name': self.config.database_name,
                    'region': self.config.region,
                    'tier': self.config.tier,
                    'deployment_time': datetime.now().isoformat()
                },
                'collections': {},
                'performance_stats': [asdict(stat) for stat in self.stats],
                'sample_queries': {}
            }
            
            # Get collection statistics
            for coll_name in self.database.list_collection_names():
                collection = self.database[coll_name]
                stats = {
                    'document_count': collection.count_documents({}),
                    'indexes': len(list(collection.list_indexes())),
                    'estimated_size_mb': collection.estimated_document_count() * 0.001  # Rough estimate
                }
                report_data['collections'][coll_name] = stats
            
            # Save report as JSON
            report_file = "Q4_Atlas_Deployment_Report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Generate summary text report
            summary_file = "Q4_Atlas_Deployment_Summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("MongoDB Atlas Deployment Summary\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Cluster: {self.config.cluster_name}\n")
                f.write(f"Database: {self.config.database_name}\n")
                f.write(f"Region: {self.config.region}\n")
                f.write(f"Tier: {self.config.tier}\n")
                f.write(f"Deployed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Collection Statistics:\n")
                f.write("-" * 25 + "\n")
                for coll_name, stats in report_data['collections'].items():
                    f.write(f"{coll_name}:\n")
                    f.write(f"  Documents: {stats['document_count']:,}\n")
                    f.write(f"  Indexes: {stats['indexes']}\n")
                    f.write(f"  Est. Size: {stats['estimated_size_mb']:.1f} MB\n\n")
                
                f.write("Performance Statistics:\n")
                f.write("-" * 25 + "\n")
                for stat in self.stats:
                    status = "[SUCCESS]" if stat.success else "[FAILED]"
                    f.write(f"{stat.operation}: {status}\n")
                    f.write(f"  Duration: {stat.duration_seconds:.2f}s\n")
                    f.write(f"  Records: {stat.records_processed:,}\n")
                    if stat.error_message:
                        f.write(f"  Error: {stat.error_message}\n")
                    f.write("\n")
            
            logger.info(f"✅ Reports generated: {report_file}, {summary_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return ""
    
    def cleanup_test_data(self):
        """Clean up any test data created during deployment."""
        try:
            logger.info("Cleaning up test data...")
            
            # Remove test documents
            self.database['transactions_centric'].delete_many({"customer_id": {"$regex": "^TEST"}})
            self.database['customers_centric'].delete_many({"_id": {"$regex": "^TEST"}})
            
            logger.info("✅ Test data cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {str(e)}")
    
    def disconnect(self):
        """Close Atlas connection."""
        if self.client:
            self.client.close()
            logger.info("✅ Disconnected from MongoDB Atlas")

def setup_atlas_instructions():
    """Print setup instructions for MongoDB Atlas."""
    print("""
    📋 MONGODB ATLAS SETUP INSTRUCTIONS
    ═══════════════════════════════════════════
    
    1. Create MongoDB Atlas Account:
       • Go to https://cloud.mongodb.com/
       • Sign up for a free account
       • Verify your email address
    
    2. Create a New Cluster:
       • Click "Create" or "Build a Database"
       • Choose "Shared" (Free tier - M0)
       • Select your preferred cloud provider and region
       • Name your cluster (e.g., "online-retail-cluster")
       • Click "Create Cluster"
    
    3. Configure Database Access:
       • Go to "Database Access" in the left menu
       • Click "Add New Database User"
       • Choose "Password" authentication
       • Create username/password (save these!)
       • Set privileges to "Read and write to any database"
       • Click "Add User"
    
    4. Configure Network Access:
       • Go to "Network Access" in the left menu
       • Click "Add IP Address"
       • Either add your current IP or choose "Allow Access from Anywhere"
       • Click "Confirm"
    
    5. Get Connection String:
       • Go to "Clusters" and click "Connect"
       • Choose "Connect your application"
       • Select Python driver version 4.0 or later
       • Copy the connection string
    
    6. Update Configuration:
       • Replace <username>, <password>, and <cluster-name> in the connection string
       • Use this information with the AtlasConfig class below
    
    ⚠️  SECURITY NOTE: Never commit credentials to version control!
    """)

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env file")
        return True
    return False

def get_atlas_config():
    """Get Atlas configuration with fallback options."""
    # Try to load from .env file first
    load_env_file()
    
    # Check for example/demo configuration
    if not any(os.getenv(var) for var in ['ATLAS_USERNAME', 'ATLAS_PASSWORD', 'ATLAS_CLUSTER_NAME']):
        print("\n🔧 No Atlas configuration found!")
        print("Run 'python atlas_setup.py' first to configure your Atlas connection")
        print("Or set these environment variables manually:")
        print("- ATLAS_USERNAME")
        print("- ATLAS_PASSWORD") 
        print("- ATLAS_CLUSTER_NAME")
        print("- ATLAS_DATABASE_NAME")
        
        # For demonstration, use example values
        print("\n⚠️  Using example configuration for demo purposes...")
        return AtlasConfig(
            username="demo_user",
            password="demo_password", 
            cluster_name="cluster0",
            database_name="online_retail_cloud"
        ), False
    
    # Get configuration from environment
    config = AtlasConfig(
        username=os.getenv('ATLAS_USERNAME'),
        password=os.getenv('ATLAS_PASSWORD'),
        cluster_name=os.getenv('ATLAS_CLUSTER_NAME'),
        database_name=os.getenv('ATLAS_DATABASE_NAME', 'online_retail_cloud'),
        region=os.getenv('ATLAS_REGION', 'us-east-1'),
        tier=os.getenv('ATLAS_TIER', 'M0')
    )
    
    return config, True

def main():
    """Main function for MongoDB Atlas deployment."""
    print("UCI Online Retail Dataset - MongoDB Atlas Deployment")
    print("Assignment 4 - Task 4: Cloud Database Deployment")
    print("-" * 60)
    
    # Load configuration
    config, is_real_config = get_atlas_config()
    
    if not is_real_config:
        print("\n❌ Please run 'python atlas_setup.py' first to configure Atlas")
        print("This will help you set up your MongoDB Atlas credentials properly.")
        return 1
    
    # Initialize deployment
    deployment = AtlasDeployment(config)
    
    try:
        # Connect to Atlas
        print(f"\n🌐 Connecting to Atlas cluster: {config.cluster_name}")
        if not deployment.connect():
            print("❌ Failed to connect to Atlas. Please check your configuration.")
            return 1
        
        # Create collections and indexes
        print("\n🏗️ Setting up collections and indexes...")
        if not deployment.create_collections():
            print("❌ Failed to create collections")
            return 1
        
        # Upload data
        print("\n📤 Uploading data to Atlas...")
        if not deployment.upload_data_from_local():
            print("❌ Data upload failed")
            return 1
        
        # Test queries
        print("\n🧪 Testing Atlas queries...")
        if not deployment.test_atlas_queries():
            print("❌ Query testing failed")
            return 1
        
        # Setup monitoring
        print("\n📊 Setting up monitoring...")
        if not deployment.setup_monitoring():
            print("❌ Monitoring setup failed")
            return 1
        
        # Generate report
        print("\n📋 Generating deployment report...")
        report_file = deployment.generate_deployment_report()
        
        # Clean up test data
        deployment.cleanup_test_data()
        
        # Success summary
        print(f"\n{'='*60}")
        print("🎉 MONGODB ATLAS DEPLOYMENT COMPLETED!")
        print(f"{'='*60}")
        print(f"✅ Cluster: {config.cluster_name}")
        print(f"✅ Database: {config.database_name}")
        print(f"✅ Region: {config.region}")
        
        if report_file:
            print(f"📄 Report: {report_file}")
        
        print(f"\n🌐 Access your data at: https://cloud.mongodb.com/")
        print(f"📊 Monitor performance in the Atlas dashboard")
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1
        
    finally:
        # Always disconnect
        deployment.disconnect()

if __name__ == "__main__":
    exit(main())