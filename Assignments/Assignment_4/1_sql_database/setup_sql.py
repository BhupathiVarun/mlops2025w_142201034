"""
Assignment 4 - Task 1: SQL Database Implementation
2nd Normalized Relational Database Design

This module creates a SQLite database with proper 2NF design for the UCI Online Retail dataset.
It implements normalized tables with relationships and loads data from the Excel file.

Author: MLOps Student
Date: December 2024
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLDatabaseSetup:
    """
    Manages the creation and population of a normalized SQLite database
    for the UCI Online Retail dataset following 2NF principles.
    """
    
    def __init__(self, db_path: str = "online_retail.db"):
        """
        Initialize the database setup.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.data_file = Path("../data/Online Retail.xlsx")
        
    def create_database_schema(self) -> None:
        """
        Creates the normalized database schema following 2NF principles.
        
        Tables created:
        - customers: Customer information
        - products: Product catalog
        - transactions: Transaction headers
        - transaction_items: Transaction line items
        """
        logger.info("Creating database schema...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Drop existing tables if they exist
            cursor.execute("DROP TABLE IF EXISTS transaction_items")
            cursor.execute("DROP TABLE IF EXISTS transactions") 
            cursor.execute("DROP TABLE IF EXISTS products")
            cursor.execute("DROP TABLE IF EXISTS customers")
            
            # Create customers table
            cursor.execute("""
                CREATE TABLE customers (
                    customer_id TEXT PRIMARY KEY,
                    country TEXT NOT NULL,
                    first_transaction_date DATE,
                    total_orders INTEGER DEFAULT 0,
                    total_spent DECIMAL(10,2) DEFAULT 0.00,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create products table  
            cursor.execute("""
                CREATE TABLE products (
                    stock_code TEXT PRIMARY KEY,
                    description TEXT,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create transactions table
            cursor.execute("""
                CREATE TABLE transactions (
                    invoice_no TEXT PRIMARY KEY,
                    customer_id TEXT,
                    invoice_date TIMESTAMP NOT NULL,
                    country TEXT NOT NULL,
                    total_amount DECIMAL(10,2) DEFAULT 0.00,
                    total_items INTEGER DEFAULT 0,
                    is_cancelled BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)
            
            # Create transaction_items table (normalized line items)
            cursor.execute("""
                CREATE TABLE transaction_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_no TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price DECIMAL(10,4) NOT NULL,
                    line_total DECIMAL(10,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (invoice_no) REFERENCES transactions(invoice_no),
                    FOREIGN KEY (stock_code) REFERENCES products(stock_code),
                    UNIQUE(invoice_no, stock_code)
                )
            """)
            
            # Create indexes for performance
            self._create_indexes(cursor)
            
            # Create triggers for data integrity
            self._create_triggers(cursor)
            
            conn.commit()
            logger.info("Database schema created successfully!")
    
    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create indexes for improved query performance."""
        indexes = [
            "CREATE INDEX idx_customers_country ON customers(country)",
            "CREATE INDEX idx_transactions_customer_id ON transactions(customer_id)",
            "CREATE INDEX idx_transactions_date ON transactions(invoice_date)",
            "CREATE INDEX idx_transactions_country ON transactions(country)",
            "CREATE INDEX idx_transaction_items_invoice ON transaction_items(invoice_no)",
            "CREATE INDEX idx_transaction_items_product ON transaction_items(stock_code)",
            "CREATE INDEX idx_products_description ON products(description)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        logger.info("Indexes created successfully!")
    
    def _create_triggers(self, cursor: sqlite3.Cursor) -> None:
        """Create triggers for maintaining data integrity and calculated fields."""
        
        # Trigger to update transaction totals when items are added
        cursor.execute("""
            CREATE TRIGGER update_transaction_totals 
            AFTER INSERT ON transaction_items
            BEGIN
                UPDATE transactions 
                SET total_amount = (
                    SELECT COALESCE(SUM(quantity * unit_price), 0)
                    FROM transaction_items 
                    WHERE invoice_no = NEW.invoice_no
                ),
                total_items = (
                    SELECT COALESCE(SUM(quantity), 0)
                    FROM transaction_items 
                    WHERE invoice_no = NEW.invoice_no
                )
                WHERE invoice_no = NEW.invoice_no;
            END
        """)
        
        # Trigger to update customer statistics
        cursor.execute("""
            CREATE TRIGGER update_customer_stats
            AFTER INSERT ON transactions
            BEGIN
                UPDATE customers
                SET total_orders = (
                    SELECT COUNT(*)
                    FROM transactions
                    WHERE customer_id = NEW.customer_id
                ),
                total_spent = (
                    SELECT COALESCE(SUM(total_amount), 0)
                    FROM transactions
                    WHERE customer_id = NEW.customer_id
                        AND is_cancelled = FALSE
                ),
                updated_at = CURRENT_TIMESTAMP
                WHERE customer_id = NEW.customer_id;
            END
        """)
        
        logger.info("Triggers created successfully!")
    
    def load_data_from_excel(self, limit_records: int = None) -> pd.DataFrame:
        """
        Load data from the Excel file.
        
        Args:
            limit_records: Maximum number of records to load (None for all)
            
        Returns:
            DataFrame containing the loaded data
        """
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        logger.info(f"Loading data from {self.data_file}...")
        
        # Load the Excel file
        df = pd.read_excel(self.data_file)
        
        # Clean the data
        df = self._clean_data(df)
        
        # Limit records if specified
        if limit_records:
            df = df.head(limit_records)
            logger.info(f"Limited dataset to {limit_records} records")
        
        logger.info(f"Loaded {len(df)} records from Excel file")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw DataFrame from Excel
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Remove rows with missing CustomerID (can't process without customer info)
        initial_count = len(df)
        df = df.dropna(subset=['CustomerID'])
        logger.info(f"Removed {initial_count - len(df)} rows with missing CustomerID")
        
        # Convert data types
        df['CustomerID'] = df['CustomerID'].astype(str).str.replace('.0', '')
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        df['StockCode'] = df['StockCode'].astype(str)
        
        # Handle missing descriptions
        df['Description'] = df['Description'].fillna('Unknown Product')
        
        # Convert invoice date to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Remove rows with zero or negative unit prices (invalid transactions)
        df = df[df['UnitPrice'] > 0]
        
        # Add derived fields
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df['IsCancellation'] = df['InvoiceNo'].str.startswith('C')
        
        logger.info(f"Data cleaning completed. Final dataset: {len(df)} records")
        return df
    
    def populate_database(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Populate the normalized database with data from the DataFrame.
        
        Args:
            df: Cleaned DataFrame to insert
            
        Returns:
            Dictionary with insert counts for each table
        """
        logger.info("Populating database...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Track insertion counts
            counts = {
                'customers': 0,
                'products': 0,
                'transactions': 0,
                'transaction_items': 0
            }
            
            # Insert customers
            counts['customers'] = self._insert_customers(cursor, df)
            
            # Insert products  
            counts['products'] = self._insert_products(cursor, df)
            
            # Insert transactions
            counts['transactions'] = self._insert_transactions(cursor, df)
            
            # Insert transaction items
            counts['transaction_items'] = self._insert_transaction_items(cursor, df)
            
            conn.commit()
            
        logger.info("Database population completed!")
        return counts
    
    def _insert_customers(self, cursor: sqlite3.Cursor, df: pd.DataFrame) -> int:
        """Insert unique customers."""
        logger.info("Inserting customers...")
        
        # Get unique customers with their first transaction date
        customers_df = df.groupby('CustomerID').agg({
            'Country': 'first',
            'InvoiceDate': 'min'
        }).reset_index()
        
        customers_data = [
            (
                row['CustomerID'],
                row['Country'], 
                row['InvoiceDate'].strftime('%Y-%m-%d %H:%M:%S')
            )
            for _, row in customers_df.iterrows()
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO customers 
            (customer_id, country, first_transaction_date)
            VALUES (?, ?, ?)
        """, customers_data)
        
        logger.info(f"Inserted {len(customers_data)} customers")
        return len(customers_data)
    
    def _insert_products(self, cursor: sqlite3.Cursor, df: pd.DataFrame) -> int:
        """Insert unique products."""
        logger.info("Inserting products...")
        
        # Get unique products
        products_df = df.groupby('StockCode').agg({
            'Description': 'first'
        }).reset_index()
        
        # Simple category assignment based on description keywords
        def assign_category(description: str) -> str:
            desc_lower = description.lower()
            if any(word in desc_lower for word in ['heart', 'love', 'valentine']):
                return 'Decorative'
            elif any(word in desc_lower for word in ['bag', 'basket', 'box']):
                return 'Storage'
            elif any(word in desc_lower for word in ['light', 'candle', 'holder']):
                return 'Lighting'
            elif any(word in desc_lower for word in ['christmas', 'xmas', 'santa']):
                return 'Seasonal'
            elif any(word in desc_lower for word in ['vintage', 'retro', 'classic']):
                return 'Vintage'
            else:
                return 'General'
        
        products_df['Category'] = products_df['Description'].apply(assign_category)
        
        products_data = [
            (row['StockCode'], row['Description'], row['Category'])
            for _, row in products_df.iterrows()
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO products 
            (stock_code, description, category)
            VALUES (?, ?, ?)
        """, products_data)
        
        logger.info(f"Inserted {len(products_data)} products")
        return len(products_data)
    
    def _insert_transactions(self, cursor: sqlite3.Cursor, df: pd.DataFrame) -> int:
        """Insert unique transactions."""
        logger.info("Inserting transactions...")
        
        # Get transaction headers
        transactions_df = df.groupby('InvoiceNo').agg({
            'CustomerID': 'first',
            'InvoiceDate': 'first',
            'Country': 'first',
            'IsCancellation': 'first'
        }).reset_index()
        
        transactions_data = [
            (
                row['InvoiceNo'],
                row['CustomerID'],
                row['InvoiceDate'].strftime('%Y-%m-%d %H:%M:%S'),
                row['Country'],
                row['IsCancellation']
            )
            for _, row in transactions_df.iterrows()
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO transactions 
            (invoice_no, customer_id, invoice_date, country, is_cancelled)
            VALUES (?, ?, ?, ?, ?)
        """, transactions_data)
        
        logger.info(f"Inserted {len(transactions_data)} transactions")
        return len(transactions_data)
    
    def _insert_transaction_items(self, cursor: sqlite3.Cursor, df: pd.DataFrame) -> int:
        """Insert transaction line items."""
        logger.info("Inserting transaction items...")
        
        items_data = [
            (row['InvoiceNo'], row['StockCode'], row['Quantity'], row['UnitPrice'])
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing items")
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO transaction_items 
            (invoice_no, stock_code, quantity, unit_price)
            VALUES (?, ?, ?, ?)
        """, items_data)
        
        logger.info(f"Inserted {len(items_data)} transaction items")
        return len(items_data)
    
    def generate_sample_queries(self) -> None:
        """Generate and execute sample queries to demonstrate functionality."""
        logger.info("Running sample queries...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            print("\n" + "="*60)
            print("SAMPLE SQL QUERIES AND RESULTS")
            print("="*60)
            
            # Query 1: Top 10 customers by total spent
            print("\n1. Top 10 Customers by Total Spent:")
            print("-" * 40)
            cursor.execute("""
                SELECT c.customer_id, c.country, c.total_spent, c.total_orders
                FROM customers c
                WHERE c.total_spent > 0
                ORDER BY c.total_spent DESC
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                print(f"Customer {row[0]}: ${row[2]:.2f} ({row[3]} orders) - {row[1]}")
            
            # Query 2: Most popular products
            print("\n2. Top 10 Most Popular Products:")
            print("-" * 40)
            cursor.execute("""
                SELECT p.stock_code, p.description, p.category,
                       SUM(ti.quantity) as total_sold,
                       COUNT(DISTINCT ti.invoice_no) as times_ordered
                FROM products p
                JOIN transaction_items ti ON p.stock_code = ti.stock_code
                GROUP BY p.stock_code, p.description, p.category
                ORDER BY total_sold DESC
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                print(f"{row[1][:50]:<50} | Sold: {row[3]} | Orders: {row[4]}")
            
            # Query 3: Sales by country
            print("\n3. Sales by Country (Top 10):")
            print("-" * 40)
            cursor.execute("""
                SELECT t.country, 
                       COUNT(DISTINCT t.invoice_no) as total_transactions,
                       COUNT(DISTINCT t.customer_id) as unique_customers,
                       SUM(t.total_amount) as total_revenue
                FROM transactions t
                WHERE t.is_cancelled = FALSE
                GROUP BY t.country
                ORDER BY total_revenue DESC
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                print(f"{row[0]:<20} | ${row[3]:>10.2f} | {row[2]:>5} customers | {row[1]:>6} orders")
            
            # Query 4: Database statistics
            print("\n4. Database Statistics:")
            print("-" * 40)
            
            stats_queries = [
                ("Total Customers", "SELECT COUNT(*) FROM customers"),
                ("Total Products", "SELECT COUNT(*) FROM products"),
                ("Total Transactions", "SELECT COUNT(*) FROM transactions"),
                ("Total Items Sold", "SELECT SUM(quantity) FROM transaction_items"),
                ("Average Order Value", """
                    SELECT AVG(total_amount) FROM transactions 
                    WHERE is_cancelled = FALSE AND total_amount > 0
                """),
            ]
            
            for stat_name, query in stats_queries:
                cursor.execute(query)
                result = cursor.fetchone()[0]
                if stat_name == "Average Order Value":
                    print(f"{stat_name}: ${result:.2f}")
                else:
                    print(f"{stat_name}: {result:,}")
    
    def run_complete_setup(self, limit_records: int = 10000) -> Dict[str, Any]:
        """
        Run the complete database setup process.
        
        Args:
            limit_records: Maximum number of records to process
            
        Returns:
            Dictionary containing setup results and statistics
        """
        start_time = datetime.now()
        
        try:
            # Create database schema
            self.create_database_schema()
            
            # Load data from Excel
            df = self.load_data_from_excel(limit_records)
            
            # Populate database
            insert_counts = self.populate_database(df)
            
            # Generate sample queries
            self.generate_sample_queries()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'execution_time_seconds': execution_time,
                'records_processed': len(df),
                'insert_counts': insert_counts,
                'database_file': os.path.abspath(self.db_path),
                'completion_time': end_time.isoformat()
            }
            
            print(f"\n{'='*60}")
            print("DATABASE SETUP COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print(f"Records Processed: {len(df):,}")
            print(f"Database File: {results['database_file']}")
            print(f"Insert Counts: {insert_counts}")
            
            return results
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }

def main():
    """Main function to run the SQL database setup."""
    print("UCI Online Retail Dataset - SQL Database Setup")
    print("Assignment 4 - Task 1: 2nd Normal Form Implementation")
    print("-" * 60)
    
    # Initialize and run database setup
    db_setup = SQLDatabaseSetup()
    
    # Run with 10,000 records (well above the minimum requirement of 1,000)
    results = db_setup.run_complete_setup(limit_records=10000)
    
    if results['success']:
        print(f"\n SQL Database setup completed successfully!")
        print(f" Database file created at: {results['database_file']}")
    else:
        print(f"\ Setup failed: {results['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())