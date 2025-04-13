# DATA JANITOR

A pragmatic data cleaning utility that turns messy datasets into analysis-ready tables through automated pandas operations.

![Data Janitor](https://img.shields.io/badge/Data%20Janitor-v0.1.0-blue)

## Overview

Data Janitor automates common data cleaning tasks to help you prepare datasets for analysis or machine learning. It provides both a user-friendly web interface and a Python API for programmatic use.

### Key Features

- **Missing Data Handler:** Intelligent imputation based on statistical properties
- **Type Conversion Engine:** Automatically fixes common data type inconsistencies
- **Duplicate Detection:** Identifies and resolves duplicate entries using configurable rules
- **Outlier Management:** Statistical detection and handling of anomalous values
- **Column Standardization:** Enforces naming conventions and structural consistency
- **Interactive UI:** Visual data profiling and cleaning operations through a Streamlit interface
- **Data Export:** Save cleaned data in various formats (CSV, Excel, JSON)
- **Pipeline Export:** Export your cleaning steps as a reusable Python script

### Technical Stack

- **Backend:** Python + pandas for heavy-duty data transformation
- **Frontend:** Streamlit for interactive data visualization and cleaning
- **Data Flow:** Raw data → Cleaning pipeline → Processed data
- **Architecture:** Stateless processing for scalability

## Installation

### Requirements

- Python 3.8+
- pandas, NumPy, scikit-learn, Streamlit, and other dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-janitor.git
   cd data-janitor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

1. Start the Streamlit app:
   ```bash
   python app.py
   ```
   
2. Open your browser at `http://localhost:8501`

3. Follow the interface to:
   - Upload your dataset
   - Explore data profiles and visualizations
   - Apply cleaning operations 
   - Export your cleaned data

### Python API

```python
# Import the library
from data_janitor import DataJanitor, clean_dataset

# Quick usage with default cleaning operations
cleaned_df = clean_dataset('messy_data.csv', operations=['normalize', 'impute_missing', 'fix_types'])
cleaned_df.to_csv('clean_data.csv')

# Or use the detailed API for more control
janitor = DataJanitor()
janitor.load_data('messy_data.csv')
janitor.normalize_column_names()
janitor.fix_data_types()
janitor.handle_missing_values({'column1': 'median', 'column2': 'most_frequent'})
janitor.remove_duplicates()
janitor.save_data('cleaned_data.csv')
```

## Cleaning Operations

### Column Name Normalization

Converts column names to lowercase and replaces spaces with underscores for consistent naming conventions.

### Data Type Fixing

Intelligently converts columns to appropriate data types:
- Strings that contain numbers → numeric types
- Date-like strings → datetime objects
- True/False strings → boolean types

### Missing Value Handling

Multiple strategies for imputing missing values:
- Numeric columns: mean, median, zero, or custom value
- Categorical columns: most frequent value, specified value
- Option to drop rows or columns with missing values

### Duplicate Removal

Identifies and removes duplicate rows based on all or specific columns.

### Outlier Detection and Removal

Detects and handles outliers using statistical methods:
- IQR (Interquartile Range) method
- Z-score method
- Configurable thresholds

### Format Standardization

Standardizes formats for:
- Dates: Converts various date formats to a consistent format
- Phone numbers: Standardizes to a consistent pattern
- Currency: Removes symbols and standardizes numeric representation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Background

Data Janitor was developed to explore pandas' full potential beyond basic operations. The project serves as both a practical utility and a deep dive into pandas' internals, designed to strengthen fundamental data engineering skills critical for ML engineering.

## Roadmap

- Implement custom cleaning extension functions
- Add ML-based anomaly detection for complex datasets
- Build API for integration with data processing workflows
- Develop specialized cleaning models for domain-specific data