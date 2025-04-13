import pandas as pd
import numpy as np
import os
import sys
import json
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to import core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from core.cleaner import DataJanitor, clean_dataset
from core.utils import (
    profile_dataset, 
    validate_constraints, 
    generate_missing_data_plot,
    generate_data_type_summary,
    detect_encoding,
    infer_delimiter
)

class DataProcessor:
    """Main class for handling data processing in the app."""
    
    def __init__(self):
        self.janitor = None
        self.original_df = None
        self.cleaned_df = None
        self.profile = None
        self.cleaning_steps = []
        self.temp_dir = tempfile.mkdtemp()
        
    def load_data(self, file_path: str, **kwargs) -> Tuple[bool, str]:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for loading
            
        Returns:
            (success, message) tuple
        """
        try:
            # Try to detect encoding and delimiter for CSV files
            if file_path.lower().endswith('.csv'):
                encoding = kwargs.get('encoding', detect_encoding(file_path))
                delimiter = kwargs.get('delimiter', infer_delimiter(file_path, encoding))
                
                self.janitor = DataJanitor()
                self.janitor.df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
            else:
                self.janitor = DataJanitor()
                self.janitor.load_data(file_path)
            
            self.original_df = self.janitor.df.copy()
            self.cleaned_df = self.janitor.df.copy()
            self.cleaning_steps = []
            
            # Profile the dataset
            self.profile = profile_dataset(self.original_df)
            
            return True, f"Successfully loaded data: {len(self.original_df)} rows, {len(self.original_df.columns)} columns"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def get_data_preview(self, n_rows: int = 100) -> Dict:
        """
        Get a preview of the original and cleaned data.
        
        Args:
            n_rows: Number of rows to preview
            
        Returns:
            Dictionary with original and cleaned data previews
        """
        if self.original_df is None:
            return {"error": "No data loaded"}
        
        result = {
            "original": {
                "columns": list(self.original_df.columns),
                "data": self.original_df.head(n_rows).to_dict(orient='records')
            }
        }
        
        if self.cleaned_df is not None:
            result["cleaned"] = {
                "columns": list(self.cleaned_df.columns),
                "data": self.cleaned_df.head(n_rows).to_dict(orient='records')
            }
        
        return result
    
    def get_profile(self) -> Dict:
        """
        Get the profile of the dataset.
        
        Returns:
            Profile dictionary
        """
        if self.profile is None:
            return {"error": "No data loaded or profiled"}
        
        # Add visualizations
        if self.original_df is not None:
            self.profile["missing_data_plot"] = generate_missing_data_plot(self.original_df)
            self.profile["data_type_plot"] = generate_data_type_summary(self.original_df)
        
        return self.profile
    
    def apply_cleaning_step(self, step_type: str, **kwargs) -> Tuple[bool, str]:
        """
        Apply a cleaning step to the data.
        
        Args:
            step_type: Type of cleaning step ('normalize', 'fix_types', etc.)
            **kwargs: Additional arguments for the step
            
        Returns:
            (success, message) tuple
        """
        if self.janitor is None or self.janitor.df is None:
            return False, "No data loaded"
        
        try:
            # Save current state for comparison
            previous_df = self.janitor.df.copy()
            
            # Initialize assumptions list to record any assumptions made
            assumptions = []
            
            # Apply the step
            if step_type == "normalize":
                self.janitor.normalize_column_names()
                message = "Normalized column names"
                assumptions.append("Column names with spaces, special characters, or inconsistent casing need standardization")
                # Generate list of changes
                old_columns = list(previous_df.columns)
                new_columns = list(self.janitor.df.columns)
                if old_columns != new_columns:
                    message += "\n• Changes:"
                    for i, (old, new) in enumerate(zip(old_columns, new_columns)):
                        if old != new:
                            message += f"\n  • '{old}' → '{new}'"
                else:
                    message += "\n• No column names needed normalization"
            
            elif step_type == "fix_types":
                self.janitor.fix_data_types()
                message = "Fixed data types"
                assumptions.append("Columns may have incorrect data types based on their content")
                # Compare data types before and after
                old_dtypes = previous_df.dtypes
                new_dtypes = self.janitor.df.dtypes
                changes = []
                for col in previous_df.columns:
                    if old_dtypes[col] != new_dtypes[col]:
                        changes.append(f"'{col}': {old_dtypes[col]} → {new_dtypes[col]}")
                        assumptions.append(f"Column '{col}' contains {new_dtypes[col]} data despite being stored as {old_dtypes[col]}")
                
                if changes:
                    message += "\n• Changes:"
                    for change in changes:
                        message += f"\n  • {change}"
                else:
                    message += "\n• No data types needed fixing"
            
            elif step_type == "remove_duplicates":
                subset = kwargs.get('columns', None)
                self.janitor.remove_duplicates(subset)
                
                # Record assumptions
                if subset:
                    assumptions.append(f"Rows with identical values in columns {subset} are true duplicates and can be removed")
                else:
                    assumptions.append("Rows with identical values across all columns are true duplicates and can be removed")
                
                # Count duplicates removed
                rows_before = len(previous_df)
                rows_after = len(self.janitor.df)
                dups_removed = rows_before - rows_after
                
                message = "Removed duplicate rows"
                if subset:
                    col_list = ", ".join(subset)
                    message += f" based on columns: {col_list}"
                
                if dups_removed > 0:
                    message += f"\n• Removed {dups_removed} duplicate rows"
                    if subset:
                        message += f" based on {len(subset)} columns"
                    message += f" ({(dups_removed/rows_before)*100:.1f}% of data)"
                else:
                    message += "\n• No duplicates found to remove"
            
            elif step_type == "handle_missing":
                strategy = kwargs.get('strategy', {})
                self.janitor.handle_missing_values(strategy)
                
                # Record assumptions for each strategy
                for col, strat in strategy.items():
                    if strat == 'mean':
                        assumptions.append(f"Missing values in '{col}' can be represented by the mean (data follows normal distribution)")
                    elif strat == 'median':
                        assumptions.append(f"Missing values in '{col}' can be represented by the median (handles skewed data better)")
                    elif strat == 'most_frequent':
                        assumptions.append(f"Missing values in '{col}' can be represented by the most common value")
                    elif strat == 'constant':
                        assumptions.append(f"Missing values in '{col}' can be replaced with a placeholder constant")
                    elif strat == 'drop':
                        assumptions.append(f"Rows with missing values in '{col}' can be removed without biasing the dataset")
                
                # Default strategy assumption
                if not strategy:
                    assumptions.append("Numeric columns with missing values can be represented by their median")
                    assumptions.append("Non-numeric columns with missing values can be represented by their most frequent value")
                
                # Count missing values before and after
                missing_before = previous_df.isna().sum().sum()
                missing_after = self.janitor.df.isna().sum().sum()
                
                message = "Handled missing values"
                if missing_before > missing_after:
                    message += f"\n• Filled {missing_before - missing_after} missing values"
                    # Add details on strategies used
                    if strategy:
                        message += "\n• Strategies used:"
                        for col, strat in strategy.items():
                            missing_in_col = previous_df[col].isna().sum()
                            if missing_in_col > 0:
                                message += f"\n  • '{col}': {strat} ({missing_in_col} values)"
                else:
                    message += "\n• No missing values were filled"
            
            elif step_type == "remove_outliers":
                columns = kwargs.get('columns', None)
                method = kwargs.get('method', 'iqr')
                threshold = kwargs.get('threshold', 1.5)
                
                # Record assumptions
                if method == 'iqr':
                    if columns:
                        assumptions.append(f"Values beyond {threshold} times the IQR in columns {columns} are outliers and can be removed")
                    else:
                        assumptions.append(f"Values beyond {threshold} times the IQR in numeric columns are outliers and can be removed")
                    assumptions.append(f"The data follows a distribution where IQR-based detection is appropriate")
                elif method == 'zscore':
                    if columns:
                        assumptions.append(f"Values beyond {threshold} standard deviations in columns {columns} are outliers and can be removed")
                    else:
                        assumptions.append(f"Values beyond {threshold} standard deviations in numeric columns are outliers and can be removed")
                    assumptions.append("The data follows a normal distribution")
                
                # Count rows before
                rows_before = len(previous_df)
                
                self.janitor.remove_outliers(columns, method, threshold)
                
                # Count rows after
                rows_after = len(self.janitor.df)
                outliers_removed = rows_before - rows_after
                
                message = f"Removed outliers using {method} method"
                if columns:
                    col_list = ", ".join(columns)
                    message += f" for columns: {col_list}"
                
                if outliers_removed > 0:
                    message += f"\n• Removed {outliers_removed} outliers"
                    message += f" ({(outliers_removed/rows_before)*100:.1f}% of data)"
                    message += f"\n• Method: {method.upper()}"
                    message += f"\n• Threshold: {threshold}"
                    if columns:
                        message += f"\n• Applied to {len(columns)} columns"
                else:
                    message += f"\n• No outliers found using {method.upper()} method with threshold {threshold}"
            
            elif step_type == "standardize_formats":
                format_dict = kwargs.get('format_dict', {})
                self.janitor.standardize_formats(format_dict)
                
                # Record assumptions
                for col, format_type in format_dict.items():
                    assumptions.append(f"Values in '{col}' can be standardized to a consistent {format_type} format")
                
                message = "Standardized formats"
                if format_dict:
                    message += "\n• Formats applied:"
                    for col, format_type in format_dict.items():
                        message += f"\n  • '{col}': {format_type}"
                else:
                    message += "\n• No format standardization applied"
            
            else:
                return False, f"Unknown cleaning step: {step_type}"
            
            # Calculate changes
            rows_before = len(previous_df)
            cols_before = len(previous_df.columns)
            rows_after = len(self.janitor.df)
            cols_after = len(self.janitor.df.columns)
            
            # Record the step with assumptions
            step_info = {
                "type": step_type,
                "params": kwargs,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "cols_before": cols_before,
                "cols_after": cols_after,
                "message": message,
                "assumptions": assumptions
            }
            
            self.cleaning_steps.append(step_info)
            self.cleaned_df = self.janitor.df.copy()
            
            # Add overall change info to message
            message += "\n\n• Summary:"
            if rows_before != rows_after:
                row_diff = rows_before - rows_after
                message += f"\n  • {row_diff} rows removed ({(row_diff/rows_before)*100:.1f}% reduction)"
            else:
                message += "\n  • No rows were removed"
                
            if cols_before != cols_after:
                if cols_after > cols_before:
                    message += f"\n  • {cols_after - cols_before} columns added"
                else:
                    message += f"\n  • {cols_before - cols_after} columns removed"
            else:
                message += "\n  • No columns were added or removed"
            
            # Add assumptions to message
            if assumptions:
                message += "\n\n• Assumptions:"
                for assumption in assumptions:
                    message += f"\n  • {assumption}"
            
            return True, message
            
        except Exception as e:
            return False, f"Error applying {step_type}: {str(e)}"
    
    def undo_last_step(self) -> Tuple[bool, str]:
        """
        Undo the last cleaning step.
        
        Returns:
            (success, message) tuple
        """
        if not self.cleaning_steps:
            return False, "No steps to undo"
        
        try:
            # Remove the last step from the list
            removed_step = self.cleaning_steps.pop()
            
            # Re-apply all remaining steps from the beginning
            self.janitor.df = self.original_df.copy()
            
            for step in self.cleaning_steps:
                if step["type"] == "normalize":
                    self.janitor.normalize_column_names()
                
                elif step["type"] == "fix_types":
                    self.janitor.fix_data_types()
                
                elif step["type"] == "remove_duplicates":
                    subset = step["params"].get('columns', None)
                    self.janitor.remove_duplicates(subset)
                
                elif step["type"] == "handle_missing":
                    strategy = step["params"].get('strategy', {})
                    self.janitor.handle_missing_values(strategy)
                
                elif step["type"] == "remove_outliers":
                    columns = step["params"].get('columns', None)
                    method = step["params"].get('method', 'iqr')
                    threshold = step["params"].get('threshold', 1.5)
                    self.janitor.remove_outliers(columns, method, threshold)
                
                elif step["type"] == "standardize_formats":
                    format_dict = step["params"].get('format_dict', {})
                    self.janitor.standardize_formats(format_dict)
            
            self.cleaned_df = self.janitor.df.copy()
            
            return True, f"Undid step: {removed_step['message']}"
            
        except Exception as e:
            return False, f"Error undoing step: {str(e)}"
    
    def save_data(self, file_path: str) -> Tuple[bool, str]:
        """
        Save the cleaned data to a file.
        
        Args:
            file_path: Path to save the file
            
        Returns:
            (success, message) tuple
        """
        if self.cleaned_df is None:
            return False, "No cleaned data to save"
        
        try:
            # Make sure the janitor has the latest cleaned data
            self.janitor.df = self.cleaned_df
            self.janitor.save_data(file_path)
            
            return True, f"Successfully saved data to {file_path}"
        except Exception as e:
            return False, f"Error saving data: {str(e)}"
    
    def export_cleaning_pipeline(self, file_path: str) -> Tuple[bool, str]:
        """
        Export the cleaning pipeline as a Python script.
        
        Args:
            file_path: Path to save the script
            
        Returns:
            (success, message) tuple
        """
        if not self.cleaning_steps:
            return False, "No cleaning steps to export"
        
        try:
            # Generate the script
            script_lines = [
                "import pandas as pd",
                "from data_janitor import DataJanitor",
                "",
                "# Load the data",
                "janitor = DataJanitor()",
                "# Replace with your file path",
                "janitor.load_data('your_data_file.csv')",
                ""
            ]
            
            # Add each cleaning step
            for step in self.cleaning_steps:
                if step["type"] == "normalize":
                    script_lines.append("# Normalize column names")
                    script_lines.append("janitor.normalize_column_names()")
                
                elif step["type"] == "fix_types":
                    script_lines.append("# Fix data types")
                    script_lines.append("janitor.fix_data_types()")
                
                elif step["type"] == "remove_duplicates":
                    script_lines.append("# Remove duplicates")
                    subset = step["params"].get('columns', None)
                    if subset:
                        subset_str = ", ".join([f"'{col}'" for col in subset])
                        script_lines.append(f"janitor.remove_duplicates(subset=[{subset_str}])")
                    else:
                        script_lines.append("janitor.remove_duplicates()")
                
                elif step["type"] == "handle_missing":
                    script_lines.append("# Handle missing values")
                    strategy = step["params"].get('strategy', {})
                    if strategy:
                        strategy_str = json.dumps(strategy, indent=4)
                        script_lines.append(f"strategy = {strategy_str}")
                        script_lines.append("janitor.handle_missing_values(strategy=strategy)")
                    else:
                        script_lines.append("janitor.handle_missing_values()")
                
                elif step["type"] == "remove_outliers":
                    script_lines.append("# Remove outliers")
                    columns = step["params"].get('columns', None)
                    method = step["params"].get('method', 'iqr')
                    threshold = step["params"].get('threshold', 1.5)
                    
                    params = []
                    if columns:
                        cols_str = ", ".join([f"'{col}'" for col in columns])
                        params.append(f"columns=[{cols_str}]")
                    if method != 'iqr':
                        params.append(f"method='{method}'")
                    if threshold != 1.5:
                        params.append(f"threshold={threshold}")
                    
                    script_lines.append(f"janitor.remove_outliers({', '.join(params)})")
                
                elif step["type"] == "standardize_formats":
                    script_lines.append("# Standardize formats")
                    format_dict = step["params"].get('format_dict', {})
                    if format_dict:
                        format_str = json.dumps(format_dict, indent=4)
                        script_lines.append(f"format_dict = {format_str}")
                        script_lines.append("janitor.standardize_formats(format_dict=format_dict)")
                
                script_lines.append("")
            
            # Add save code
            script_lines.append("# Save the cleaned data")
            script_lines.append("janitor.save_data('cleaned_data.csv')")
            
            # Write the script to file
            with open(file_path, 'w') as f:
                f.write('\n'.join(script_lines))
            
            return True, f"Successfully exported cleaning pipeline to {file_path}"
        except Exception as e:
            return False, f"Error exporting pipeline: {str(e)}"
    
    def validate_data(self, constraints: Dict) -> Dict:
        """
        Validate the cleaned data against constraints.
        
        Args:
            constraints: Dictionary of constraints
            
        Returns:
            Validation results
        """
        if self.cleaned_df is None:
            return {"valid": False, "errors": ["No cleaned data available"]}
        
        try:
            return validate_constraints(self.cleaned_df, constraints)
        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"]}
    
    def get_cleaning_steps(self) -> List[Dict]:
        """
        Get the list of applied cleaning steps.
        
        Returns:
            List of cleaning steps
        """
        return self.cleaning_steps
        
    def generate_cleaning_documentation(self) -> str:
        """
        Generate a comprehensive documentation report of all cleaning steps, decisions, and assumptions.
        
        Returns:
            A formatted string containing the cleaning documentation
        """
        if not self.cleaning_steps:
            return "No cleaning steps have been applied."
            
        doc_lines = ["# Data Cleaning Documentation", ""]
        
        # Overview statistics
        doc_lines.append("## Overview")
        if self.original_df is not None and self.cleaned_df is not None:
            orig_rows, orig_cols = len(self.original_df), len(self.original_df.columns)
            clean_rows, clean_cols = len(self.cleaned_df), len(self.cleaned_df.columns)
            
            doc_lines.append(f"* Original dataset: {orig_rows} rows, {orig_cols} columns")
            doc_lines.append(f"* Cleaned dataset: {clean_rows} rows, {clean_cols} columns")
            
            if orig_rows > clean_rows:
                doc_lines.append(f"* Rows removed: {orig_rows - clean_rows} ({((orig_rows - clean_rows)/orig_rows)*100:.1f}%)")
            
            if orig_cols != clean_cols:
                if clean_cols > orig_cols:
                    doc_lines.append(f"* Columns added: {clean_cols - orig_cols}")
                else:
                    doc_lines.append(f"* Columns removed: {orig_cols - clean_cols}")
            
        doc_lines.append("")
        
        # Summary of all steps
        doc_lines.append("## Cleaning Steps Summary")
        for i, step in enumerate(self.cleaning_steps, 1):
            doc_lines.append(f"{i}. {step['type'].replace('_', ' ').title()}")
        
        doc_lines.append("")
        
        # Detailed breakdown of each step
        doc_lines.append("## Detailed Cleaning Steps")
        for i, step in enumerate(self.cleaning_steps, 1):
            doc_lines.append(f"### Step {i}: {step['type'].replace('_', ' ').title()}")
            
            # Parameters used
            if step['params']:
                doc_lines.append("**Parameters:**")
                for param, value in step['params'].items():
                    doc_lines.append(f"* {param}: {value}")
            
            # Extract message content (without the bullet points)
            message_parts = step['message'].split('\n')
            main_message = message_parts[0]
            doc_lines.append(f"**Action:** {main_message}")
            
            # Changes made
            doc_lines.append("**Changes:**")
            bullet_points = [line for line in message_parts[1:] if line.strip().startswith('•')]
            if bullet_points:
                doc_lines.extend(bullet_points)
            else:
                doc_lines.append("* No significant changes detected")
            
            # Assumptions
            if 'assumptions' in step and step['assumptions']:
                doc_lines.append("**Assumptions:**")
                for assumption in step['assumptions']:
                    doc_lines.append(f"* {assumption}")
            
            # Data impact
            doc_lines.append("**Data Impact:**")
            rows_diff = step['rows_before'] - step['rows_after']
            cols_diff = step['cols_before'] - step['cols_after']
            
            if rows_diff > 0:
                doc_lines.append(f"* Removed {rows_diff} rows ({(rows_diff/step['rows_before'])*100:.1f}% of data)")
            
            if cols_diff != 0:
                if cols_diff > 0:
                    doc_lines.append(f"* Removed {cols_diff} columns")
                else:
                    doc_lines.append(f"* Added {abs(cols_diff)} columns")
            
            doc_lines.append("")
        
        # Generate recommendations section
        doc_lines.append("## Recommendations and Notes")
        doc_lines.append("* This cleaning pipeline should be reviewed and updated as data evolves")
        doc_lines.append("* Consider additional validation steps for critical data fields")
        doc_lines.append("* Monitor data quality over time to ensure consistency")
        
        # Compile all assumptions in one place
        all_assumptions = []
        for step in self.cleaning_steps:
            if 'assumptions' in step:
                all_assumptions.extend(step['assumptions'])
        
        if all_assumptions:
            doc_lines.append("\n## All Assumptions")
            for assumption in set(all_assumptions):  # Use set to remove duplicates
                doc_lines.append(f"* {assumption}")
        
        return "\n".join(doc_lines)
