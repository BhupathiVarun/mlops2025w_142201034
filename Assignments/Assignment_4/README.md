# Assignment 4: Database Comparison - SQL vs MongoDB

This project implements a comprehensive comparison between SQL (SQLite) and MongoDB database approaches using the UCI Online Retail Dataset.

## 📁 Project Structure

```
Assignment_4/
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   └── Online Retail.xlsx
├── 1_sql_database/
│   ├── online_retail.db
│   └── setup_sql.py
├── 2_mongodb_models/
│   └── setup_mongo.py
├── 3_performance_comparison/
│   ├── crud_comparison.py
│   └── Q3_Report.pdf
└── 4_atlas_deployment/
    └── deploy_to_atlas.py
```

## 📊 Dataset Information

**UCI Online Retail Dataset**
- **Source**: https://archive.ics.uci.edu/dataset/352/online+retail
- **Size**: 540,000+ transactions
- **Period**: December 2010 - December 2011
- **Domain**: UK-based online retailer

### Dataset Schema:
- **InvoiceNo**: 6-digit transaction identifier (prefix 'C' = cancellations)
- **StockCode**: 5-digit product identifier
- **Description**: Product name/description
- **Quantity**: Number of items per transaction
- **InvoiceDate**: Transaction timestamp
- **UnitPrice**: Product price in Sterling
- **CustomerID**: 5-digit customer identifier
- **Country**: Customer's country of residence

## 🎯 Assignment Tasks

### Task 1: SQL Database Design (5 Marks)
- Design 2nd Normal Form (2NF) relational database
- Insert minimum 1000 records
- Implement proper relationships and constraints

### Task 2: MongoDB Implementation (10 Marks)
- **Transaction-centric approach** (4 marks): Documents organized by transactions
- **Customer-centric approach** (4 marks): Documents organized by customers
- **PyMongo with connection pooling** (2 marks): Error handling and optimization

### Task 3: Performance Comparison (5 Marks)
- CRUD operations benchmark
- Performance analysis with evidence
- Detailed comparison report

### Task 4: MongoDB Atlas Deployment (5 Marks)
- Cloud deployment with specific cluster configuration
- Choose either transaction-centric or customer-centric approach

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- UV package manager
- MongoDB (local installation or Atlas account)
- Git

### Installation Steps

1. **Clone and Navigate**:
   ```bash
   cd Assignment_4
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

3. **Activate Virtual Environment**:
   ```bash
   uv shell
   ```

## 🗄️ Database Implementations

### 1. SQL Database (SQLite)
**Design**: 2nd Normal Form with separate tables:
- **customers**: Customer information
- **products**: Product catalog
- **transactions**: Transaction headers
- **transaction_items**: Transaction line items

**Run**:
```bash
cd 1_sql_database
python setup_sql.py
```

### 2. MongoDB Implementation
**Two Approaches**:

**A. Transaction-Centric**:
```json
{
  "_id": "536365",
  "invoice_date": "2010-12-01T08:26:00",
  "customer_id": "17850",
  "country": "United Kingdom",
  "items": [
    {
      "stock_code": "85123A",
      "description": "WHITE HANGING HEART T-LIGHT HOLDER",
      "quantity": 6,
      "unit_price": 2.55
    }
  ],
  "total_amount": 15.3
}
```

**B. Customer-Centric**:
```json
{
  "_id": "17850",
  "country": "United Kingdom",
  "registration_date": "2010-12-01",
  "transactions": [
    {
      "invoice_no": "536365",
      "date": "2010-12-01T08:26:00",
      "items": [...],
      "total": 15.3
    }
  ],
  "total_spent": 1234.56,
  "total_orders": 5
}
```

**Run**:
```bash
cd 2_mongodb_models
python setup_mongo.py
```

### 3. Performance Comparison
**Benchmarked Operations**:
- **Create**: Insert new records
- **Read**: Query by various criteria
- **Update**: Modify existing data
- **Delete**: Remove records

**Run**:
```bash
cd 3_performance_comparison
python crud_comparison.py
```

### 4. MongoDB Atlas Deployment
**Cloud Configuration**:
- **Cluster**: M0 Sandbox (Free tier)
- **Region**: Closest to your location
- **MongoDB Version**: 6.0+
- **Approach**: Transaction-centric (optimized for e-commerce queries)

**Run**:
```bash
cd 4_atlas_deployment
python deploy_to_atlas.py
```

## 🔧 MongoDB Setup Guide

### Local MongoDB Installation

**Windows**:
1. Download MongoDB Community Server
2. Install with default settings
3. Add to PATH: `C:\Program Files\MongoDB\Server\6.0\bin`
4. Start service: `net start MongoDB`

**MacOS** (using Homebrew):
```bash
brew tap mongodb/brew
brew install mongodb-community@6.0
brew services start mongodb/brew/mongodb-community@6.0
```

**Ubuntu/Debian**:
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
```

### MongoDB Atlas Setup

1. **Create Account**: Sign up at https://www.mongodb.com/atlas
2. **Create Cluster**: 
   - Choose M0 Sandbox (free)
   - Select your preferred region
   - Name your cluster
3. **Network Access**: Add your IP address
4. **Database User**: Create username/password
5. **Get Connection String**: Copy for use in code

### Connection String Format:
```
mongodb+srv://<username>:<password>@<cluster-url>/<database>?retryWrites=true&w=majority
```

## 📈 Performance Results Summary

### Expected Performance Characteristics:

**SQL (SQLite) Strengths**:
- ACID compliance
- Complex queries with JOINs
- Structured data integrity
- Mature ecosystem

**MongoDB Strengths**:
- Horizontal scaling
- Flexible schema
- Fast document retrieval
- JSON/BSON native support

**Use Case Recommendations**:
- **SQL**: Complex reporting, financial transactions, strict consistency
- **MongoDB**: Real-time analytics, content management, rapid prototyping

## 📝 Files Explanation

### Core Implementation Files:

1. **`1_sql_database/setup_sql.py`**:
   - Creates normalized SQLite database
   - Implements 2NF design with proper relationships
   - Bulk loads data from Excel file
   - Creates indexes for performance

2. **`2_mongodb_models/setup_mongo.py`**:
   - Implements both transaction-centric and customer-centric models
   - Uses PyMongo with connection pooling
   - Includes error handling and retry logic
   - Creates appropriate indexes

3. **`3_performance_comparison/crud_comparison.py`**:
   - Benchmarks CRUD operations on both databases
   - Measures execution time and memory usage
   - Generates performance metrics and charts
   - Creates detailed comparison report

4. **`4_atlas_deployment/deploy_to_atlas.py`**:
   - Connects to MongoDB Atlas cloud
   - Migrates data to cloud cluster
   - Configures security and access controls
   - Tests cloud deployment

### Configuration Files:

- **`pyproject.toml`**: Project dependencies and metadata
- **`uv.lock`**: Locked dependency versions
- **`.gitignore`**: Excludes database files and sensitive data

## 🚀 Quick Start Commands

```bash
# Setup everything
uv sync && uv shell

# Run SQL implementation
python 1_sql_database/setup_sql.py

# Run MongoDB implementation  
python 2_mongodb_models/setup_mongo.py

# Run performance comparison
python 3_performance_comparison/crud_comparison.py

# Deploy to Atlas (requires credentials)
python 4_atlas_deployment/deploy_to_atlas.py
```

## 📊 Expected Output

1. **SQLite database** with normalized tables and 1000+ records
2. **MongoDB collections** with both document models
3. **Performance report** (PDF) with benchmarks and analysis
4. **Cloud deployment** confirmation with Atlas cluster

## 🔍 Troubleshooting

### Common Issues:

1. **MongoDB Connection Error**:
   - Ensure MongoDB service is running
   - Check connection string format
   - Verify network access for Atlas

2. **Excel File Not Found**:
   - Ensure `Online Retail.xlsx` is in `data/` folder
   - Check file permissions

3. **UV Command Not Found**:
   - Install UV: `pip install uv`
   - Or use pip directly for dependencies

4. **Memory Issues with Large Dataset**:
   - Process data in chunks
   - Increase available RAM
   - Use pagination for queries

## 🎓 Learning Objectives

By completing this assignment, you will understand:

- **Database Design**: Normalization vs. denormalization trade-offs
- **NoSQL vs SQL**: When to use each approach
- **Performance Optimization**: Indexing and query optimization
- **Cloud Deployment**: Modern database hosting solutions
- **Real-world Application**: E-commerce data modeling patterns

## 📚 References

- [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [SQLite Documentation](https://sqlite.org/docs.html)
- [PyMongo Tutorial](https://pymongo.readthedocs.io/)
- [UV Package Manager](https://docs.astral.sh/uv/)

---

**Note**: This is an educational project demonstrating database design patterns and performance comparison techniques. All implementations include proper error handling and follow best practices for production-ready code.
