import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import base64
import tempfile
from typing import Dict, List, Any
from io import StringIO
import plotly.express as px

# Add parent directory to path to import core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from app.processing import DataProcessor

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Upload"

if 'cleaning_operations' not in st.session_state:
    st.session_state.cleaning_operations = []

# Helper functions
def display_dataframe_preview(df, max_rows=5):
    """Display a preview of the DataFrame with styling."""
    if df is None:
        return
    
    st.dataframe(df.head(max_rows), use_container_width=True)
    st.caption(f"Showing {min(max_rows, len(df))} of {len(df)} rows • {len(df.columns)} columns")

def render_image_from_base64(base64_string):
    """Render an image from a base64 string."""
    if base64_string is None:
        return
    
    html = f'<img src="data:image/png;base64,{base64_string}" style="max-width: 100%;">'
    st.markdown(html, unsafe_allow_html=True)

def get_download_link(file_path, link_text):
    """Generate a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(file_path)}" target="_blank">{link_text}</a>'
    return href

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Data Janitor",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    st.sidebar.title("Data Janitor 🧹")
    st.sidebar.caption("Clean your data with ease")
    
    tabs = ["Upload", "Explore", "Clean", "Export"]
    current_tab = st.sidebar.radio("Navigation", tabs, index=tabs.index(st.session_state.current_tab))
    st.session_state.current_tab = current_tab
    
    # Show cleanup steps if data is loaded
    if st.session_state.data_processor.cleaning_steps:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Cleaning Steps")
        
        for i, step in enumerate(st.session_state.data_processor.cleaning_steps):
            step_msg = step["message"]
            st.sidebar.markdown(f"{i+1}. {step_msg}")
        
        if st.sidebar.button("Undo Last Step"):
            success, message = st.session_state.data_processor.undo_last_step()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
            st.experimental_rerun()
    
    # Main content
    st.title("Data Janitor")
    st.caption("A pragmatic data cleaning utility")
    
    # Upload tab
    if current_tab == "Upload":
        render_upload_tab()
    
    # Explore tab
    elif current_tab == "Explore":
        render_explore_tab()
    
    # Clean tab
    elif current_tab == "Clean":
        render_clean_tab()
    
    # Export tab
    elif current_tab == "Export":
        render_export_tab()


def render_upload_tab():
    """Render the upload tab."""
    st.header("Upload Data")
    st.markdown("Upload your dataset to begin cleaning. Supported formats: CSV, Excel, JSON.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"])
        
        if uploaded_file is not None:
            # Save uploaded file to temp location
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Add file options based on type
            if file_extension == 'csv':
                encoding_options = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
                encoding = st.selectbox("File Encoding", encoding_options)
                
                delimiter_options = [",", ";", "\t", "|"]
                delimiter = st.selectbox("Delimiter", delimiter_options)
                
                if st.button("Load Data"):
                    with st.spinner("Loading data..."):
                        success, message = st.session_state.data_processor.load_data(
                            temp_file_path, 
                            encoding=encoding,
                            delimiter=delimiter
                        )
                        
                        if success:
                            st.success(message)
                            st.session_state.current_tab = "Explore"
                            st.experimental_rerun()
                        else:
                            st.error(message)
            else:
                if st.button("Load Data"):
                    with st.spinner("Loading data..."):
                        success, message = st.session_state.data_processor.load_data(temp_file_path)
                        
                        if success:
                            st.success(message)
                            st.session_state.current_tab = "Explore"
                            st.experimental_rerun()
                        else:
                            st.error(message)
    
    with col2:
        st.markdown("### Sample Data")
        st.markdown("""
        Don't have a dataset? Get one from:
        - [Kaggle Datasets](https://www.kaggle.com/datasets)
        - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)
        - [OpenML](https://www.openml.org/search?type=data&sort=runs&page=1)
        """)


def render_explore_tab():
    """Render the explore tab."""
    st.header("Explore Data")
    
    # Check if data is loaded
    if st.session_state.data_processor.original_df is None:
        st.warning("No data loaded. Please upload a dataset first.")
        return
    
    # Get data preview
    preview = st.session_state.data_processor.get_data_preview(n_rows=10)
    profile = st.session_state.data_processor.get_profile()
    
    # Data preview section
    st.subheader("Data Preview")
    
    if "original" in preview:
        original_df = pd.DataFrame(preview["original"]["data"])
        display_dataframe_preview(original_df, max_rows=10)
    
    # Data profile section
    st.subheader("Data Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Rows", profile["row_count"])
        st.metric("Memory Usage", f"{profile['memory_usage'] / (1024 * 1024):.2f} MB")
        
        # Render missing data plot if available
        if "missing_data_plot" in profile and profile["missing_data_plot"]:
            st.markdown("### Missing Data")
            render_image_from_base64(profile["missing_data_plot"])
    
    with col2:
        st.metric("Columns", profile["column_count"])
        
        # Calculate data quality score (simple metric)
        missing_percentages = [
            profile["columns"][col]["missing_percentage"] 
            for col in profile["columns"]
        ]
        
        quality_score = 100 - (sum(missing_percentages) / len(missing_percentages))
        st.metric("Data Quality Score", f"{quality_score:.1f}%")
        
        # Render data type plot if available
        if "data_type_plot" in profile and profile["data_type_plot"]:
            st.markdown("### Data Types")
            render_image_from_base64(profile["data_type_plot"])
    
    # Column details section
    st.subheader("Column Details")
    
    # Create tabs for each data type category
    numeric_cols = []
    datetime_cols = []
    categorical_cols = []
    
    for col, details in profile["columns"].items():
        dtype = details["dtype"]
        if "float" in dtype or "int" in dtype:
            numeric_cols.append(col)
        elif "datetime" in dtype:
            datetime_cols.append(col)
        else:
            categorical_cols.append(col)
    
    tabs = []
    if numeric_cols:
        tabs.append("Numeric Columns")
    if datetime_cols:
        tabs.append("DateTime Columns")
    if categorical_cols:
        tabs.append("Categorical Columns")
    
    if tabs:
        selected_tab = st.radio("Column Types", tabs)
        
        if selected_tab == "Numeric Columns":
            render_numeric_columns(numeric_cols, profile)
        elif selected_tab == "DateTime Columns":
            render_datetime_columns(datetime_cols, profile)
        elif selected_tab == "Categorical Columns":
            render_categorical_columns(categorical_cols, profile)


def render_numeric_columns(columns, profile):
    """Render details for numeric columns."""
    for col in columns:
        with st.expander(col):
            details = profile["columns"][col]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min", f"{details.get('min', 'N/A')}")
            with col2:
                st.metric("Max", f"{details.get('max', 'N/A')}")
            with col3:
                st.metric("Mean", f"{details.get('mean', 'N/A')}")
            with col4:
                st.metric("Median", f"{details.get('median', 'N/A')}")
            
            st.metric("Missing Values", f"{details['missing_count']} ({details['missing_percentage']}%)")
            st.metric("Unique Values", details["unique_count"])
            
            st.markdown("#### Sample Values")
            st.write(details["sample_values"])


def render_datetime_columns(columns, profile):
    """Render details for datetime columns."""
    for col in columns:
        with st.expander(col):
            details = profile["columns"][col]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Min Date", f"{details.get('min', 'N/A')}")
            with col2:
                st.metric("Max Date", f"{details.get('max', 'N/A')}")
            
            st.metric("Missing Values", f"{details['missing_count']} ({details['missing_percentage']}%)")
            st.metric("Unique Values", details["unique_count"])
            
            st.markdown("#### Sample Values")
            st.write(details["sample_values"])


def render_categorical_columns(columns, profile):
    """Render details for categorical columns."""
    for col in columns:
        with st.expander(col):
            details = profile["columns"][col]
            
            st.metric("Missing Values", f"{details['missing_count']} ({details['missing_percentage']}%)")
            st.metric("Unique Values", details["unique_count"])
            
            if "top_values" in details:
                st.markdown("#### Top Values")
                
                # Create bar chart of top values
                top_values = details["top_values"]
                df = pd.DataFrame({
                    'Value': list(top_values.keys()),
                    'Count': list(top_values.values())
                })
                
                fig = px.bar(df, x='Value', y='Count', title=f"Top Values for {col}")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Sample Values")
            st.write(details["sample_values"])


def render_clean_tab():
    """Render the clean tab."""
    st.header("Clean Data")
    
    # Check if data is loaded
    if st.session_state.data_processor.original_df is None:
        st.warning("No data loaded. Please upload a dataset first.")
        return
    
    # Display current state
    st.subheader("Current Data")
    if st.session_state.data_processor.cleaned_df is not None:
        display_dataframe_preview(st.session_state.data_processor.cleaned_df, max_rows=5)
    
    # Cleaning operations
    st.subheader("Cleaning Operations")
    
    cleaning_options = [
        "Normalize Column Names",
        "Fix Data Types",
        "Remove Duplicates", 
        "Handle Missing Values",
        "Remove Outliers",
        "Standardize Formats"
    ]
    
    # Two columns: one for selecting operation, one for parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_operation = st.selectbox("Select Operation", cleaning_options)
    
    with col2:
        # Different parameters based on selected operation
        if selected_operation == "Normalize Column Names":
            st.info("This will convert column names to lowercase and replace spaces with underscores.")
            
            if st.button("Apply Normalization"):
                with st.spinner("Normalizing column names..."):
                    success, message = st.session_state.data_processor.apply_cleaning_step("normalize")
                    
                    if success:
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(message)
        
        elif selected_operation == "Fix Data Types":
            st.info("This will attempt to convert columns to appropriate data types (numeric, datetime, etc.).")
            
            if st.button("Fix Data Types"):
                with st.spinner("Fixing data types..."):
                    success, message = st.session_state.data_processor.apply_cleaning_step("fix_types")
                    
                    if success:
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(message)
        
        elif selected_operation == "Remove Duplicates":
            st.info("Remove duplicate rows based on all or selected columns.")
            
            # Get columns for subset selection
            if st.session_state.data_processor.cleaned_df is not None:
                columns = list(st.session_state.data_processor.cleaned_df.columns)
                subset = st.multiselect("Select columns to check for duplicates (leave empty for all columns)", columns)
                
                if st.button("Remove Duplicates"):
                    with st.spinner("Removing duplicates..."):
                        success, message = st.session_state.data_processor.apply_cleaning_step(
                            "remove_duplicates",
                            columns=subset if subset else None
                        )
                        
                        if success:
                            st.success(message)
                            st.experimental_rerun()
                        else:
                            st.error(message)
        
        elif selected_operation == "Handle Missing Values":
            st.info("Handle missing values using different strategies.")
            
            # Get columns for strategy selection
            if st.session_state.data_processor.cleaned_df is not None:
                df = st.session_state.data_processor.cleaned_df
                missing_columns = [col for col in df.columns if df[col].isna().any()]
                
                if not missing_columns:
                    st.success("No missing values found in the dataset.")
                else:
                    st.write(f"Found {len(missing_columns)} columns with missing values.")
                    
                    # Create strategy selection for each column
                    strategy = {}
                    
                    for col in missing_columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            options = ["mean", "median", "constant", "drop"]
                            default_index = 1  # median
                        else:
                            options = ["most_frequent", "constant", "drop"]
                            default_index = 0  # most_frequent
                        
                        col_strategy = st.selectbox(
                            f"Strategy for '{col}' ({df[col].isna().sum()} missing)",
                            options,
                            index=default_index
                        )
                        
                        strategy[col] = col_strategy
                    
                    if st.button("Handle Missing Values"):
                        with st.spinner("Handling missing values..."):
                            success, message = st.session_state.data_processor.apply_cleaning_step(
                                "handle_missing",
                                strategy=strategy
                            )
                            
                            if success:
                                st.success(message)
                                st.experimental_rerun()
                            else:
                                st.error(message)
        
        elif selected_operation == "Remove Outliers":
            st.info("Detect and remove outliers in numerical columns.")
            
            # Get numeric columns
            if st.session_state.data_processor.cleaned_df is not None:
                df = st.session_state.data_processor.cleaned_df
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_columns:
                    st.warning("No numeric columns found in the dataset.")
                else:
                    columns = st.multiselect(
                        "Select columns to check for outliers (leave empty for all numeric columns)",
                        numeric_columns
                    )
                    
                    method = st.selectbox(
                        "Outlier detection method",
                        ["iqr", "zscore"],
                        index=0
                    )
                    
                    if method == "iqr":
                        threshold = st.slider(
                            "IQR threshold (higher = less strict)",
                            min_value=1.0,
                            max_value=3.0,
                            value=1.5,
                            step=0.1
                        )
                    else:
                        threshold = st.slider(
                            "Z-score threshold (higher = less strict)",
                            min_value=2.0,
                            max_value=5.0,
                            value=3.0,
                            step=0.1
                        )
                    
                    if st.button("Remove Outliers"):
                        with st.spinner("Removing outliers..."):
                            success, message = st.session_state.data_processor.apply_cleaning_step(
                                "remove_outliers",
                                columns=columns if columns else None,
                                method=method,
                                threshold=threshold
                            )
                            
                            if success:
                                st.success(message)
                                st.experimental_rerun()
                            else:
                                st.error(message)
        
        elif selected_operation == "Standardize Formats":
            st.info("Standardize formats for specified columns (e.g., dates, phone numbers, currency).")
            
            # Get columns for format selection
            if st.session_state.data_processor.cleaned_df is not None:
                columns = list(st.session_state.data_processor.cleaned_df.columns)
                
                format_dict = {}
                format_options = ["None", "date", "phone", "currency"]
                
                # Let user select columns to standardize
                selected_columns = st.multiselect("Select columns to standardize", columns)
                
                if selected_columns:
                    st.write("Select format for each column:")
                    
                    for col in selected_columns:
                        format_type = st.selectbox(
                            f"Format for '{col}'",
                            format_options,
                            index=0,
                            key=f"format_{col}"
                        )
                        
                        if format_type != "None":
                            format_dict[col] = format_type
                    
                    if st.button("Standardize Formats"):
                        if not format_dict:
                            st.warning("Please select at least one column and format.")
                        else:
                            with st.spinner("Standardizing formats..."):
                                success, message = st.session_state.data_processor.apply_cleaning_step(
                                    "standardize_formats",
                                    format_dict=format_dict
                                )
                                
                                if success:
                                    st.success(message)
                                    st.experimental_rerun()
                                else:
                                    st.error(message)

    # Before and after comparison if cleaning steps have been applied
    if st.session_state.data_processor.cleaning_steps:
        st.subheader("Before & After Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Data")
            display_dataframe_preview(st.session_state.data_processor.original_df, max_rows=5)
        
        with col2:
            st.markdown("#### Cleaned Data")
            display_dataframe_preview(st.session_state.data_processor.cleaned_df, max_rows=5)


def render_export_tab():
    """Render the export tab."""
    st.header("Export Data")
    
    # Check if data is loaded
    if st.session_state.data_processor.cleaned_df is None:
        st.warning("No data loaded. Please upload a dataset first.")
        return
    
    # Check if cleaning steps have been applied
    if not st.session_state.data_processor.cleaning_steps:
        st.warning("No cleaning steps have been applied yet.")
    
    # Display cleaned data
    st.subheader("Cleaned Data Preview")
    display_dataframe_preview(st.session_state.data_processor.cleaned_df, max_rows=5)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Data")
        export_format = st.selectbox(
            "Select export format",
            ["CSV", "Excel", "JSON"]
        )
        
        if export_format == "CSV":
            file_ext = "csv"
        elif export_format == "Excel":
            file_ext = "xlsx"
        else:
            file_ext = "json"
        
        export_filename = st.text_input("Filename", f"cleaned_data.{file_ext}")
        
        if st.button("Export Data"):
            with st.spinner("Exporting data..."):
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                temp_file_path = temp_file.name
                temp_file.close()
                
                success, message = st.session_state.data_processor.save_data(temp_file_path)
                
                if success:
                    st.success(message)
                    
                    # Create download link
                    download_link = get_download_link(temp_file_path, "Download File")
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.error(message)
    
    with col2:
        st.markdown("#### Export Cleaning Pipeline")
        st.write("Export your cleaning steps as a reusable Python script.")
        
        pipeline_filename = st.text_input("Pipeline Filename", "cleaning_pipeline.py")
        
        if st.button("Export Pipeline"):
            with st.spinner("Exporting pipeline..."):
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
                temp_file_path = temp_file.name
                temp_file.close()
                
                success, message = st.session_state.data_processor.export_cleaning_pipeline(temp_file_path)
                
                if success:
                    st.success(message)
                    
                    # Create download link
                    download_link = get_download_link(temp_file_path, "Download Pipeline Script")
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.error(message)

# Run the app
if __name__ == "__main__":
    main()
