import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

def profile_dataset(df: pd.DataFrame) -> Dict:
    """
    Generate a basic profile of the dataset, including:
    - Basic stats: row count, column count
    - Missing value analysis
    - Data type summary
    - Sample values
    
    Args:
        df: Pandas DataFrame to profile
        
    Returns:
        Dictionary with profiling results
    """
    # Basic stats
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "columns": {}
    }
    
    # Profile each column
    for col in df.columns:
        col_profile = {
            "dtype": str(df[col].dtype),
            "missing_count": df[col].isna().sum(),
            "missing_percentage": round(df[col].isna().mean() * 100, 2),
            "unique_count": df[col].nunique(),
        }
        
        # Add stats specific to data type
        if pd.api.types.is_numeric_dtype(df[col]):
            col_profile.update({
                "min": df[col].min() if not df[col].isna().all() else None,
                "max": df[col].max() if not df[col].isna().all() else None,
                "mean": df[col].mean() if not df[col].isna().all() else None,
                "median": df[col].median() if not df[col].isna().all() else None,
            })
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_profile.update({
                "min": df[col].min() if not df[col].isna().all() else None,
                "max": df[col].max() if not df[col].isna().all() else None,
            })
        else:
            # For object/string columns, get most common values
            if not df[col].isna().all():
                value_counts = df[col].value_counts().head(5).to_dict()
                col_profile["top_values"] = value_counts
            
        # Get sample values
        col_profile["sample_values"] = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()
        
        profile["columns"][col] = col_profile
    
    return profile

def validate_constraints(df: pd.DataFrame, constraints: Dict) -> Dict:
    """
    Validate the dataset against a set of constraints.
    
    Args:
        df: Pandas DataFrame to validate
        constraints: Dictionary of constraints like:
            {
                "column_name": {
                    "type": "numeric|date|string|etc.",
                    "required": True|False,
                    "unique": True|False,
                    "min": value,
                    "max": value,
                    "regex": pattern,
                    "allowed_values": [...]
                }
            }
            
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": []
    }
    
    for col, constraint in constraints.items():
        # Check if column exists
        if col not in df.columns:
            results["valid"] = False
            results["errors"].append(f"Column '{col}' not found in dataset")
            continue
        
        # Check required (no nulls)
        if constraint.get("required", False) and df[col].isna().any():
            results["valid"] = False
            results["errors"].append(f"Column '{col}' has missing values but is required")
        
        # Check unique
        if constraint.get("unique", False) and df[col].nunique() < len(df[col].dropna()):
            results["valid"] = False
            results["errors"].append(f"Column '{col}' has duplicate values but requires unique values")
        
        # Type checks
        col_type = constraint.get("type", None)
        if col_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            results["valid"] = False
            results["errors"].append(f"Column '{col}' should be numeric but is {df[col].dtype}")
        elif col_type == "date" and not pd.api.types.is_datetime64_dtype(df[col]):
            results["valid"] = False
            results["errors"].append(f"Column '{col}' should be datetime but is {df[col].dtype}")
        
        # Range checks for numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            if "min" in constraint and (df[col] < constraint["min"]).any():
                results["valid"] = False
                results["errors"].append(f"Column '{col}' has values below minimum {constraint['min']}")
            if "max" in constraint and (df[col] > constraint["max"]).any():
                results["valid"] = False
                results["errors"].append(f"Column '{col}' has values above maximum {constraint['max']}")
        
        # Allowed values
        if "allowed_values" in constraint:
            invalid_values = df[~df[col].isna() & ~df[col].isin(constraint["allowed_values"])][col].unique()
            if len(invalid_values) > 0:
                results["valid"] = False
                results["errors"].append(f"Column '{col}' has invalid values: {invalid_values[:5]}")
        
        # Regex pattern for string columns
        if "regex" in constraint and pd.api.types.is_string_dtype(df[col]):
            import re
            pattern = re.compile(constraint["regex"])
            mask = df[col].apply(lambda x: bool(pattern.match(str(x))) if pd.notna(x) else True)
            if not mask.all():
                results["valid"] = False
                results["errors"].append(f"Column '{col}' has values not matching pattern '{constraint['regex']}'")
    
    return results

def generate_missing_data_plot(df: pd.DataFrame) -> str:
    """
    Generate a visualization of missing data.
    
    Args:
        df: Pandas DataFrame to analyze
        
    Returns:
        Base64 encoded image of the missing data heatmap
    """
    # Calculate missing data percentages
    missing_data = df.isna().mean().sort_values(ascending=False) * 100
    
    # Only keep columns with missing data
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) == 0:
        return None
    
    # Create a bar chart
    fig = px.bar(
        x=missing_data.index, 
        y=missing_data.values,
        labels={'x': 'Column', 'y': 'Missing Data (%)'},
        title='Missing Data Analysis',
        height=400
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    # Convert to base64 image
    img_bytes = io.BytesIO()
    fig.write_image(img_bytes, format='png')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode()

def generate_data_type_summary(df: pd.DataFrame) -> str:
    """
    Generate a visualization of data types.
    
    Args:
        df: Pandas DataFrame to analyze
        
    Returns:
        Base64 encoded image of the data type pie chart
    """
    # Get data type counts
    dtypes = df.dtypes.astype(str)
    type_counts = dtypes.value_counts().reset_index()
    type_counts.columns = ['Data Type', 'Count']
    
    # Create a pie chart
    fig = px.pie(
        type_counts, 
        values='Count', 
        names='Data Type',
        title='Data Type Distribution',
        height=400
    )
    
    # Convert to base64 image
    img_bytes = io.BytesIO()
    fig.write_image(img_bytes, format='png')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode()

def detect_encoding(file_path: str, n_bytes: int = 10000) -> str:
    """
    Attempt to detect the encoding of a file.
    
    Args:
        file_path: Path to the file
        n_bytes: Number of bytes to read for detection
        
    Returns:
        Detected encoding
    """
    try:
        import chardet
        with open(file_path, 'rb') as f:
            rawdata = f.read(n_bytes)
        result = chardet.detect(rawdata)
        return result['encoding']
    except ImportError:
        return 'utf-8'  # Default to UTF-8

def infer_delimiter(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Infer the delimiter of a CSV file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        Detected delimiter
    """
    import csv
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            sample = f.readline() + f.readline()
        
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except:
        return ','  # Default to comma
