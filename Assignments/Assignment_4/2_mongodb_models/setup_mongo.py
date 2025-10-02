"""
Assignment 4 - Task 2: MongoDB Implementation
Document-oriented Database with Transaction-Centric and Customer-Centric Approaches

This module implements both transaction-centric and customer-centric MongoDB models
using PyMongo with connection pooling and comprehensive error handling.

Author: MLOps Student
Date: December 2024
"""

import pandas as pd
import pymongo
from pymongo import MongoClient, errors
from pymongo.collection import Collection
from pymongo.database import Database
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConnectionConfig:
    """MongoDB connection configuration."""
    host: str = "localhost"
    port: int = 27017
    database_name: str = "online_retail_db"
    username: Optional[str] = None
    password: Optional[str] = None
    connection_timeout: int = 5000
    max_pool_size: int = 100
    min_pool_size: int = 10

class MongoDBSetup:
    """
    Manages MongoDB database setup with both transaction-centric and customer-centric approaches.
    Implements connection pooling, error handling, and data migration from Excel.
    """
    
    def __init__(self, config: ConnectionConfig = None):
        """
        Initialize MongoDB setup with configuration.
        
        Args:
            config: MongoDB connection configuration
        """
        self.config = config or ConnectionConfig()
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.data_file = Path("../data/Online Retail.xlsx")
        
        # Collection names
        self.transaction_collection = "transactions_centric"
        self.customer_collection = "customers_centric"
        self.metadata_collection = "setup_metadata"
        
    def establish_connection(self) -> bool:
        """
        Establish MongoDB connection with connection pooling and error handling.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Build connection string
            if self.config.username and self.config.password:
                connection_string = (
                    f"mongodb://{self.config.username}:{self.config.password}"
                    f"@{self.config.host}:{self.config.port}/{self.config.database_name}"
                )
            else:
                connection_string = f"mongodb://{self.config.host}:{self.config.port}"
            
            # Create client with connection pooling
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=self.config.connection_timeout,
                maxPoolSize=self.config.max_pool_size,
                minPoolSize=self.config.min_pool_size,
                retryWrites=True,
                w='majority'
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database
            self.db = self.client[self.config.database_name]
            
            logger.info(f"Successfully connected to MongoDB at {self.config.host}:{self.config.port}")
            logger.info(f"Database: {self.config.database_name}")
            logger.info(f"Connection pool: {self.config.min_pool_size}-{self.config.max_pool_size}")
            
            return True
            
        except errors.ServerSelectionTimeoutError:
            logger.error("Failed to connect to MongoDB server - timeout")
            return False
        except errors.ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            return False
    
    def close_connection(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def load_and_clean_data(self, limit_records: int = None) -> pd.DataFrame:
        """
        Load and clean data from Excel file.
        
        Args:
            limit_records: Maximum number of records to load
            
        Returns:
            Cleaned DataFrame
        """
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        logger.info(f"Loading data from {self.data_file}...")
        
        # Load Excel file
        df = pd.read_excel(self.data_file)
        
        # Clean data
        logger.info("Cleaning and preprocessing data...")
        
        # Remove rows with missing CustomerID
        initial_count = len(df)
        df = df.dropna(subset=['CustomerID'])
        logger.info(f"Removed {initial_count - len(df)} rows with missing CustomerID")
        
        # Convert data types
        df['CustomerID'] = df['CustomerID'].astype(str).str.replace('.0', '')
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        df['StockCode'] = df['StockCode'].astype(str)
        df['Description'] = df['Description'].fillna('Unknown Product')
        
        # Convert dates
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Remove invalid prices
        df = df[df['UnitPrice'] > 0]
        
        # Add calculated fields
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df['IsCancellation'] = df['InvoiceNo'].str.startswith('C')
        
        # Limit records if specified
        if limit_records:
            df = df.head(limit_records)
            logger.info(f"Limited dataset to {limit_records} records")
        
        logger.info(f"Data cleaning completed. Final dataset: {len(df)} records")
        return df
    
    def create_indexes(self) -> None:
        """Create indexes for both collections to optimize query performance."""
        logger.info("Creating database indexes...")
        
        try:
            # Indexes for transaction-centric collection
            transaction_coll = self.db[self.transaction_collection]
            transaction_indexes = [
                [("invoice_no", 1)],  # Primary key
                [("customer_id", 1)],  # Customer queries
                [("invoice_date", -1)],  # Date-based queries
                [("country", 1)],  # Geographic queries
                [("total_amount", -1)],  # Value-based queries
                [("items.stock_code", 1)],  # Product queries in embedded items
                [("customer_id", 1), ("invoice_date", -1)],  # Compound index
            ]
            
            for index in transaction_indexes:
                transaction_coll.create_index(index)
            
            # Indexes for customer-centric collection  
            customer_coll = self.db[self.customer_collection]
            customer_indexes = [
                [("customer_id", 1)],  # Primary key
                [("country", 1)],  # Geographic queries
                [("total_spent", -1)],  # Customer value queries
                [("total_orders", -1)],  # Activity queries
                [("registration_date", -1)],  # Registration date
                [("transactions.invoice_no", 1)],  # Transaction lookups
                [("country", 1), ("total_spent", -1)],  # Compound index
            ]
            
            for index in customer_indexes:
                customer_coll.create_index(index)
            
            logger.info("Indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            raise
    
    def implement_transaction_centric_model(self, df: pd.DataFrame) -> int:
        """
        Implement transaction-centric MongoDB model.
        Each document represents a transaction with embedded items.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Number of documents inserted
        """
        logger.info("Implementing transaction-centric model...")
        
        collection = self.db[self.transaction_collection]
        
        # Drop existing collection
        collection.drop()
        
        # Group data by invoice number
        grouped = df.groupby('InvoiceNo')
        
        documents = []
        
        for invoice_no, group in tqdm(grouped, desc="Processing transactions"):
            # Get transaction header info
            first_row = group.iloc[0]
            
            # Build items array
            items = []
            for _, item_row in group.iterrows():
                item_doc = {
                    "stock_code": item_row['StockCode'],
                    "description": item_row['Description'],
                    "quantity": int(item_row['Quantity']),
                    "unit_price": float(item_row['UnitPrice']),
                    "line_total": float(item_row['TotalPrice'])
                }
                items.append(item_doc)
            
            # Build transaction document
            transaction_doc = {
                "_id": invoice_no,
                "invoice_no": invoice_no,
                "customer_id": first_row['CustomerID'],
                "invoice_date": first_row['InvoiceDate'],
                "country": first_row['Country'],
                "is_cancellation": bool(first_row['IsCancellation']),
                "total_amount": float(group['TotalPrice'].sum()),
                "total_items": int(group['Quantity'].sum()),
                "item_count": len(items),
                "items": items,
                "created_at": datetime.now(timezone.utc),
                "model_type": "transaction_centric"
            }
            
            documents.append(transaction_doc)
            
            # Batch insert to avoid memory issues
            if len(documents) >= 1000:
                try:
                    collection.insert_many(documents, ordered=False)
                    documents = []
                except errors.BulkWriteError as e:
                    logger.warning(f"Bulk write error (continuing): {len(e.details['writeErrors'])} errors")
                    documents = []
        
        # Insert remaining documents
        if documents:
            try:
                collection.insert_many(documents, ordered=False)
            except errors.BulkWriteError as e:
                logger.warning(f"Bulk write error (final): {len(e.details['writeErrors'])} errors")
        
        # Get final count
        total_count = collection.count_documents({})
        logger.info(f"Transaction-centric model: {total_count} documents inserted")
        
        return total_count
    
    def implement_customer_centric_model(self, df: pd.DataFrame) -> int:
        """
        Implement customer-centric MongoDB model.
        Each document represents a customer with embedded transactions.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Number of documents inserted
        """
        logger.info("Implementing customer-centric model...")
        
        collection = self.db[self.customer_collection]
        
        # Drop existing collection
        collection.drop()
        
        # Group data by customer
        grouped = df.groupby('CustomerID')
        
        documents = []
        
        for customer_id, customer_group in tqdm(grouped, desc="Processing customers"):
            # Get customer info
            first_row = customer_group.iloc[0]
            
            # Group customer transactions
            transactions = []
            transaction_groups = customer_group.groupby('InvoiceNo')
            
            for invoice_no, transaction_group in transaction_groups:
                # Build items for this transaction
                items = []
                for _, item_row in transaction_group.iterrows():
                    item_doc = {
                        "stock_code": item_row['StockCode'],
                        "description": item_row['Description'],
                        "quantity": int(item_row['Quantity']),
                        "unit_price": float(item_row['UnitPrice']),
                        "line_total": float(item_row['TotalPrice'])
                    }
                    items.append(item_doc)
                
                # Build transaction document
                transaction_doc = {
                    "invoice_no": invoice_no,
                    "date": transaction_group.iloc[0]['InvoiceDate'],
                    "country": transaction_group.iloc[0]['Country'],
                    "is_cancellation": bool(transaction_group.iloc[0]['IsCancellation']),
                    "total": float(transaction_group['TotalPrice'].sum()),
                    "item_count": len(items),
                    "items": items
                }
                
                transactions.append(transaction_doc)
            
            # Calculate customer statistics
            total_spent = customer_group[~customer_group['IsCancellation']]['TotalPrice'].sum()
            total_orders = len([t for t in transactions if not t['is_cancellation']])
            
            # Build customer document
            customer_doc = {
                "_id": customer_id,
                "customer_id": customer_id,
                "country": first_row['Country'],
                "registration_date": customer_group['InvoiceDate'].min(),
                "last_order_date": customer_group['InvoiceDate'].max(),
                "total_spent": float(total_spent),
                "total_orders": total_orders,
                "total_transactions": len(transactions),
                "avg_order_value": float(total_spent / total_orders) if total_orders > 0 else 0,
                "transactions": transactions,
                "created_at": datetime.now(timezone.utc),
                "model_type": "customer_centric"
            }
            
            documents.append(customer_doc)
            
            # Batch insert
            if len(documents) >= 100:  # Smaller batches for customer docs (they're larger)
                try:
                    collection.insert_many(documents, ordered=False)
                    documents = []
                except errors.BulkWriteError as e:
                    logger.warning(f"Bulk write error (continuing): {len(e.details['writeErrors'])} errors")
                    documents = []
        
        # Insert remaining documents
        if documents:
            try:
                collection.insert_many(documents, ordered=False)
            except errors.BulkWriteError as e:
                logger.warning(f"Bulk write error (final): {len(e.details['writeErrors'])} errors")
        
        # Get final count
        total_count = collection.count_documents({})
        logger.info(f"Customer-centric model: {total_count} documents inserted")
        
        return total_count
    
    def demonstrate_queries(self) -> None:
        """Demonstrate various queries on both MongoDB models."""
        logger.info("Demonstrating MongoDB queries...")
        
        print("\n" + "="*60)
        print("MONGODB QUERY DEMONSTRATIONS")
        print("="*60)
        
        # Transaction-centric queries
        print("\n🔸 TRANSACTION-CENTRIC MODEL QUERIES:")
        print("-" * 50)
        
        trans_coll = self.db[self.transaction_collection]
        
        # Query 1: Largest transactions
        print("\n1. Top 5 Largest Transactions:")
        largest_transactions = trans_coll.find(
            {"is_cancellation": False}
        ).sort("total_amount", -1).limit(5)
        
        for trans in largest_transactions:
            print(f"Invoice {trans['invoice_no']}: ${trans['total_amount']:.2f} "
                  f"({trans['item_count']} items) - {trans['country']}")
        
        # Query 2: Transactions by country
        print("\n2. Transaction Count by Country:")
        pipeline = [
            {"$match": {"is_cancellation": False}},
            {"$group": {
                "_id": "$country",
                "transaction_count": {"$sum": 1},
                "total_revenue": {"$sum": "$total_amount"}
            }},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 5}
        ]
        
        for result in trans_coll.aggregate(pipeline):
            print(f"{result['_id']:<20} | {result['transaction_count']:>5} transactions | "
                  f"${result['total_revenue']:>10.2f}")
        
        # Customer-centric queries
        print("\n🔸 CUSTOMER-CENTRIC MODEL QUERIES:")
        print("-" * 50)
        
        cust_coll = self.db[self.customer_collection]
        
        # Query 3: Top customers by spending
        print("\n3. Top 5 Customers by Total Spending:")
        top_customers = cust_coll.find().sort("total_spent", -1).limit(5)
        
        for customer in top_customers:
            print(f"Customer {customer['customer_id']}: ${customer['total_spent']:.2f} "
                  f"({customer['total_orders']} orders) - {customer['country']}")
        
        # Query 4: Customer activity analysis
        print("\n4. Customer Activity Analysis:")
        pipeline = [
            {"$group": {
                "_id": None,
                "total_customers": {"$sum": 1},
                "avg_spent_per_customer": {"$avg": "$total_spent"},
                "avg_orders_per_customer": {"$avg": "$total_orders"},
                "max_spent": {"$max": "$total_spent"},
                "min_spent": {"$min": "$total_spent"}
            }}
        ]
        
        for result in cust_coll.aggregate(pipeline):
            print(f"Total Customers: {result['total_customers']:,}")
            print(f"Average Spent: ${result['avg_spent_per_customer']:.2f}")
            print(f"Average Orders: {result['avg_orders_per_customer']:.1f}")
            print(f"Spending Range: ${result['min_spent']:.2f} - ${result['max_spent']:.2f}")
        
        # Query 5: Product popularity across both models
        print("\n5. Most Popular Products (from transaction model):")
        pipeline = [
            {"$unwind": "$items"},
            {"$group": {
                "_id": {
                    "stock_code": "$items.stock_code",
                    "description": "$items.description"
                },
                "total_quantity": {"$sum": "$items.quantity"},
                "times_ordered": {"$sum": 1}
            }},
            {"$sort": {"total_quantity": -1}},
            {"$limit": 5}
        ]
        
        for result in trans_coll.aggregate(pipeline):
            product = result['_id']
            print(f"{product['description'][:40]:<40} | "
                  f"Qty: {result['total_quantity']:>5} | "
                  f"Orders: {result['times_ordered']:>4}")
    
    def save_setup_metadata(self, results: Dict[str, Any]) -> None:
        """Save setup metadata for future reference."""
        metadata_coll = self.db[self.metadata_collection]
        
        metadata_doc = {
            "setup_date": datetime.now(timezone.utc),
            "setup_results": results,
            "collections": {
                "transaction_centric": self.transaction_collection,
                "customer_centric": self.customer_collection
            },
            "indexes_created": True,
            "connection_config": {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database_name,
                "pool_size": f"{self.config.min_pool_size}-{self.config.max_pool_size}"
            }
        }
        
        metadata_coll.insert_one(metadata_doc)
        logger.info("Setup metadata saved")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        stats = {}
        
        try:
            # Database-level stats
            db_stats = self.db.command("dbStats")
            stats['database'] = {
                'name': self.config.database_name,
                'collections': db_stats.get('collections', 0),
                'data_size_mb': round(db_stats.get('dataSize', 0) / (1024*1024), 2),
                'index_size_mb': round(db_stats.get('indexSize', 0) / (1024*1024), 2)
            }
            
            # Collection-specific stats
            stats['collections'] = {}
            
            for collection_name in [self.transaction_collection, self.customer_collection]:
                if collection_name in self.db.list_collection_names():
                    coll_stats = self.db.command("collStats", collection_name)
                    stats['collections'][collection_name] = {
                        'document_count': coll_stats.get('count', 0),
                        'avg_document_size': coll_stats.get('avgObjSize', 0),
                        'total_size_mb': round(coll_stats.get('size', 0) / (1024*1024), 2),
                        'index_count': coll_stats.get('nindexes', 0)
                    }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            stats['error'] = str(e)
        
        return stats
    
    def run_complete_setup(self, limit_records: int = 10000) -> Dict[str, Any]:
        """
        Run the complete MongoDB setup process.
        
        Args:
            limit_records: Maximum number of records to process
            
        Returns:
            Dictionary containing setup results
        """
        start_time = datetime.now()
        
        try:
            # Establish connection
            if not self.establish_connection():
                raise ConnectionError("Failed to establish MongoDB connection")
            
            # Load and clean data
            df = self.load_and_clean_data(limit_records)
            
            # Create indexes
            self.create_indexes()
            
            # Implement transaction-centric model
            transaction_count = self.implement_transaction_centric_model(df)
            
            # Implement customer-centric model  
            customer_count = self.implement_customer_centric_model(df)
            
            # Demonstrate queries
            self.demonstrate_queries()
            
            # Get database statistics
            db_stats = self.get_database_stats()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'execution_time_seconds': execution_time,
                'records_processed': len(df),
                'documents_created': {
                    'transaction_centric': transaction_count,
                    'customer_centric': customer_count
                },
                'database_stats': db_stats,
                'connection_info': {
                    'host': self.config.host,
                    'port': self.config.port,
                    'database': self.config.database_name
                },
                'completion_time': end_time.isoformat()
            }
            
            # Save metadata
            self.save_setup_metadata(results)
            
            print(f"\n{'='*60}")
            print("MONGODB SETUP COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print(f"Records Processed: {len(df):,}")
            print(f"Transaction Documents: {transaction_count:,}")
            print(f"Customer Documents: {customer_count:,}")
            print(f"Database: {self.config.host}:{self.config.port}/{self.config.database_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"MongoDB setup failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }
        
        finally:
            self.close_connection()

def main():
    """Main function to run MongoDB setup."""
    print("UCI Online Retail Dataset - MongoDB Setup")
    print("Assignment 4 - Task 2: Document-Oriented Database Implementation")
    print("Transaction-Centric & Customer-Centric Models with PyMongo")
    print("-" * 60)
    
    # Configuration
    config = ConnectionConfig(
        host="localhost",
        port=27017,
        database_name="online_retail_db",
        max_pool_size=50,
        min_pool_size=5
    )
    
    # Initialize and run MongoDB setup
    mongo_setup = MongoDBSetup(config)
    
    # Run with 10,000 records
    results = mongo_setup.run_complete_setup(limit_records=10000)
    
    if results['success']:
        print(f"\n MongoDB setup completed successfully!")
        print(f" Database: {results['connection_info']['database']}")
        print(f" Transaction Documents: {results['documents_created']['transaction_centric']:,}")
        print(f" Customer Documents: {results['documents_created']['customer_centric']:,}")
    else:
        print(f"\n Setup failed: {results['error']}")
        print("\n Troubleshooting tips:")
        print("1. Ensure MongoDB is running: mongod")
        print("2. Check connection details in the configuration")
        print("3. Verify MongoDB service is accessible on localhost:27017")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())