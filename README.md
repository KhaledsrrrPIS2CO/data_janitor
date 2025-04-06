DATA JANITOR
A pragmatic data cleaning utility that turns messy datasets into analysis-ready tables through automated pandas operations.

Technical Stack
Backend: Python + pandas for heavy-duty data transformation

Frontend: Minimalist file upload/download interface

Data Flow: Raw data in → Cleaning pipeline → Processed data out

Architecture: Stateless processing for scalability

Core Capabilities
Missing Data Handler: Intelligent imputation based on statistical properties

Type Conversion Engine: Automatically fixes common data type inconsistencies

Duplicate Detection: Identifies and resolves duplicate entries using configurable rules

Outlier Management: Statistical detection and handling of anomalous values

Column Standardization: Enforces naming conventions and structural consistency

Quick Setup
bash
git clone https://github.com/yourusername/data-janitor.git
cd data-janitor
pip install -r requirements.txt
python app.py
Usage
python
# Core usage pattern
from data_janitor import clean_dataset

df = clean_dataset('messy_data.csv', operations=['normalize', 'impute_missing', 'fix_types'])
df.to_csv('clean_data.csv')
Implementation Notes
This project deliberately leverages pandas' advanced capabilities including:

Method chaining for efficient data transformation pipelines

Vectorized operations for performance optimization

Custom accessor methods for domain-specific cleaning operations

Project Background
Data Janitor was developed to explore pandas' full potential beyond basic operations. The project serves as both a practical utility and a deep dive into pandas' internals, designed to strengthen fundamental data engineering skills critical for ML engineering.

Roadmap
Implement custom cleaning extension functions

Add ML-based anomaly detection for complex datasets

Build API for integration with data processing workflows

Develop specialized cleaning models for domain-specific data

License
MIT