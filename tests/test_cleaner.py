import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cleaner import DataJanitor, clean_dataset

class TestDataJanitor(unittest.TestCase):
    """Test cases for the DataJanitor class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with various issues
        self.data = {
            "First Name": ["John", "Jane", "John", None, "Alex"],
            "Last Name": ["Smith", "Doe", "Smith", "Johnson", None],
            "Age": ["25", "30", "25", "40", "35"],
            "Income": ["$50,000", "$60,000", "$50,000", None, "$70,000"],
            "Date Joined": ["2020-01-01", "2019/06/15", "01-01-2020", "2018.12.30", None],
            "Phone Number": ["555-123-4567", "5551234567", "(555) 123-4568", None, "555.123.4569"]
        }
        self.df = pd.DataFrame(self.data)
        self.janitor = DataJanitor(self.df.copy())
        
        # Create a temporary CSV file for testing
        self.temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_csv.name)
    
    def test_normalize_column_names(self):
        """Test normalizing column names."""
        self.janitor.normalize_column_names()
        expected_columns = ['first_name', 'last_name', 'age', 'income', 'date_joined', 'phone_number']
        self.assertEqual(list(self.janitor.df.columns), expected_columns)
    
    def test_remove_duplicates(self):
        """Test removing duplicate rows."""
        self.janitor.remove_duplicates()
        self.assertEqual(len(self.janitor.df), 4)  # One duplicate row removed
        
        # Test with subset
        janitor2 = DataJanitor(self.df.copy())
        janitor2.remove_duplicates(subset=["First Name"])
        self.assertEqual(len(janitor2.df), 3)  # Two rows with "John" become one
    
    def test_fix_data_types(self):
        """Test fixing data types."""
        self.janitor.normalize_column_names()  # First normalize columns
        self.janitor.fix_data_types()
        
        # Age should be converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(self.janitor.df['age']))
        
        # Date joined should be datetime
        self.assertTrue(pd.api.types.is_datetime64_dtype(self.janitor.df['date_joined']))
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        self.janitor.normalize_column_names()
        
        # Test with default strategy
        self.janitor.handle_missing_values()
        
        # Should have no missing values after imputation
        self.assertEqual(self.janitor.df['first_name'].isna().sum(), 0)
        self.assertEqual(self.janitor.df['last_name'].isna().sum(), 0)
        
        # Test with specific strategy
        janitor2 = DataJanitor(self.df.copy())
        janitor2.normalize_column_names()
        
        strategy = {
            'first_name': 'drop',
            'income': 'constant'
        }
        
        janitor2.handle_missing_values(strategy)
        
        # Rows with missing first_name should be dropped
        self.assertEqual(len(janitor2.df), 4)
        
        # Missing income should be filled with a constant
        self.assertEqual(janitor2.df['income'].isna().sum(), 0)
    
    def test_remove_outliers(self):
        """Test removing outliers."""
        # Create a new DataFrame with numeric data and outliers
        numeric_data = {
            "values": [10, 12, 15, 18, 100, 13, 17]  # 100 is an outlier
        }
        df = pd.DataFrame(numeric_data)
        
        janitor = DataJanitor(df)
        janitor.remove_outliers(method='iqr', threshold=1.5)
        
        # The outlier should be removed
        self.assertEqual(len(janitor.df), 6)
        self.assertTrue(100 not in janitor.df['values'].values)
    
    def test_standardize_formats(self):
        """Test standardizing formats."""
        self.janitor.normalize_column_names()
        
        format_dict = {
            'date_joined': 'date',
            'phone_number': 'phone'
        }
        
        self.janitor.standardize_formats(format_dict)
        
        # All dates should be in the same format
        not_null_dates = self.janitor.df['date_joined'].dropna()
        first_date_format = not_null_dates.iloc[0]
        self.assertTrue(all(d == first_date_format for d in not_null_dates))
        
        # All phone numbers should be in the same format (XXX-XXX-XXXX)
        not_null_phones = self.janitor.df['phone_number'].dropna()
        self.assertTrue(all(len(p) == 12 for p in not_null_phones))
        self.assertTrue(all('-' in p for p in not_null_phones))
    
    def test_load_data(self):
        """Test loading data from a file."""
        janitor = DataJanitor()
        janitor.load_data(self.temp_csv.name)
        
        # Should have the same shape as the original DataFrame
        self.assertEqual(janitor.df.shape, self.df.shape)
    
    def test_save_data(self):
        """Test saving data to a file."""
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_output.close()
        
        self.janitor.save_data(temp_output.name)
        
        # Read back the saved file
        df_read = pd.read_csv(temp_output.name)
        
        # Should have the same shape as the original DataFrame
        self.assertEqual(df_read.shape, self.df.shape)
        
        os.unlink(temp_output.name)
    
    def test_clean_dataset_function(self):
        """Test the clean_dataset helper function."""
        cleaned_df = clean_dataset(
            self.temp_csv.name,
            operations=['normalize', 'fix_types', 'remove_duplicates']
        )
        
        # Should have normalized column names
        self.assertTrue('first_name' in cleaned_df.columns)
        
        # Should have removed duplicates
        self.assertEqual(len(cleaned_df), 4)


if __name__ == '__main__':
    unittest.main()
