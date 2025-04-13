import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.impute import SimpleImputer
from datetime import datetime
import re

class DataJanitor:
    """
    Main class for data cleaning operations.
    Provides methods to transform messy datasets into analysis-ready tables.
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        
    def load_data(self, file_path: str) -> 'DataJanitor':
        """Load data from various file formats."""
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'csv':
            self.df = pd.read_csv(file_path)
        elif file_ext in ['xls', 'xlsx']:
            self.df = pd.read_excel(file_path)
        elif file_ext == 'json':
            self.df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return self
    
    def normalize_column_names(self) -> 'DataJanitor':
        """Standardize column names: lowercase, replace spaces with underscores."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
            
        self.df.columns = [re.sub(r'[^\w\s]', '', col).lower().replace(' ', '_') for col in self.df.columns]
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataJanitor':
        """Remove duplicate rows based on all or specified columns."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
            
        self.df = self.df.drop_duplicates(subset=subset)
        return self
    
    def fix_data_types(self) -> 'DataJanitor':
        """Infer and convert columns to appropriate data types."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
        
        # Try to convert object columns to numeric
        for col in self.df.select_dtypes(include=['object']).columns:
            # Try numeric conversion
            try:
                numeric_vals = pd.to_numeric(self.df[col])
                self.df[col] = numeric_vals
            except (ValueError, TypeError):
                # Try datetime conversion
                try:
                    date_vals = pd.to_datetime(self.df[col])
                    if not date_vals.isna().all():  # Only convert if at least some values parsed correctly
                        self.df[col] = date_vals
                except (ValueError, TypeError):
                    # Keep as string/object
                    pass
        
        return self
    
    def handle_missing_values(self, strategy: Dict[str, str] = None) -> 'DataJanitor':
        """
        Handle missing values using specified strategies.
        
        Args:
            strategy: Dict mapping column names to imputation strategies:
                      'mean', 'median', 'most_frequent', 'constant', or 'drop'
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded")
            
        if strategy is None:
            strategy = {}
            
        # Default strategies by data type
        for col in self.df.columns:
            col_strategy = strategy.get(col, None)
            
            # If no strategy specified, choose based on data type
            if col_strategy is None:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    col_strategy = 'median'
                else:
                    col_strategy = 'most_frequent'
            
            if col_strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
            elif col_strategy in ['mean', 'median', 'most_frequent']:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    imputer = SimpleImputer(strategy=col_strategy)
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                elif col_strategy == 'most_frequent':
                    imputer = SimpleImputer(strategy='most_frequent')
                    self.df[col] = imputer.fit_transform(self.df[[col]])
            elif col_strategy == 'constant':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(0)
                else:
                    self.df[col] = self.df[col].fillna('unknown')
        
        return self
    
    def remove_outliers(self, columns: List[str] = None, method: str = 'iqr', threshold: float = 1.5) -> 'DataJanitor':
        """
        Detect and remove outliers in numerical columns.
        
        Args:
            columns: List of column names to check for outliers
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded")
            
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns.tolist()
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
            elif method == 'zscore':
                mean = self.df[col].mean()
                std = self.df[col].std()
                
                if std != 0:  # Avoid division by zero
                    z_scores = np.abs((self.df[col] - mean) / std)
                    self.df = self.df[z_scores <= threshold]
        
        return self
    
    def standardize_formats(self, format_dict: Dict[str, str]) -> 'DataJanitor':
        """
        Standardize formats for specified columns.
        
        Args:
            format_dict: Dict mapping column names to format types ('date', 'phone', 'currency', etc.)
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded")
            
        for col, format_type in format_dict.items():
            if col not in self.df.columns:
                continue
                
            if format_type == 'date':
                try:
                    self.df[col] = pd.to_datetime(self.df[col]).dt.strftime('%Y-%m-%d')
                except:
                    pass
            elif format_type == 'phone':
                # Simple phone number standardization (US format)
                self.df[col] = self.df[col].astype(str).apply(
                    lambda x: re.sub(r'[^\d]', '', x)[-10:] if re.sub(r'[^\d]', '', x) else x
                )
                # Format as XXX-XXX-XXXX
                self.df[col] = self.df[col].apply(
                    lambda x: f"{x[:3]}-{x[3:6]}-{x[6:]}" if len(x) == 10 else x
                )
            elif format_type == 'currency':
                # Remove currency symbols and convert to float
                self.df[col] = self.df[col].astype(str).apply(
                    lambda x: re.sub(r'[^\d.]', '', x) if re.search(r'[\d.]', x) else x
                )
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
        return self
    
    def get_clean_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
            
        return self.df.copy()
    
    def save_data(self, file_path: str) -> None:
        """Save the cleaned data to a file."""
        if self.df is None:
            raise ValueError("No DataFrame loaded")
            
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'csv':
            self.df.to_csv(file_path, index=False)
        elif file_ext in ['xls', 'xlsx']:
            self.df.to_excel(file_path, index=False)
        elif file_ext == 'json':
            self.df.to_json(file_path, orient='records')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


def clean_dataset(file_path: str, operations: List[str] = None, **kwargs) -> pd.DataFrame:
    """
    High-level function to clean a dataset with specified operations.
    
    Args:
        file_path: Path to the input file
        operations: List of cleaning operations to perform
        kwargs: Additional arguments for specific operations
        
    Returns:
        Cleaned pandas DataFrame
    """
    janitor = DataJanitor()
    janitor.load_data(file_path)
    
    if operations is None:
        operations = ['normalize', 'impute_missing', 'fix_types', 'remove_duplicates']
    
    for op in operations:
        if op == 'normalize':
            janitor.normalize_column_names()
        elif op == 'impute_missing':
            strategy = kwargs.get('missing_strategy', None)
            janitor.handle_missing_values(strategy)
        elif op == 'fix_types':
            janitor.fix_data_types()
        elif op == 'remove_duplicates':
            subset = kwargs.get('duplicate_subset', None)
            janitor.remove_duplicates(subset)
        elif op == 'remove_outliers':
            columns = kwargs.get('outlier_columns', None)
            method = kwargs.get('outlier_method', 'iqr')
            threshold = kwargs.get('outlier_threshold', 1.5)
            janitor.remove_outliers(columns, method, threshold)
        elif op == 'standardize_formats':
            format_dict = kwargs.get('format_dict', {})
            janitor.standardize_formats(format_dict)
    
    return janitor.get_clean_data()
