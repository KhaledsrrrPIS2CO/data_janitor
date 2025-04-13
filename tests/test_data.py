import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import (
    profile_dataset,
    validate_constraints,
    generate_missing_data_plot,
    generate_data_type_summary
)

class TestUtilsFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with various data types and issues
        self.data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", None, "Eve"],
            "age": [25, 30, None, 40, 35],
            "income": [50000.0, 60000.0, 55000.0, None, 70000.0],
            "is_active": [True, False, True, True, None],
            "join_date": pd.to_datetime(["2020-01-01", "2019-06-15", "2021-03-20", None, "2018-12-30"])
        }
        self.df = pd.DataFrame(self.data)
    
    def test_profile_dataset(self):
        """Test generating dataset profile."""
        profile = profile_dataset(self.df)
        
        # Check basic stats
        self.assertEqual(profile["row_count"], 5)
        self.assertEqual(profile["column_count"], 6)
        
        # Check column profiles
        self.assertTrue("id" in profile["columns"])
        self.assertTrue("name" in profile["columns"])
        
        # Check missing value counts
        self.assertEqual(profile["columns"]["name"]["missing_count"], 1)
        self.assertEqual(profile["columns"]["age"]["missing_count"], 1)
        
        # Check data type detection
        self.assertTrue("int" in profile["columns"]["id"]["dtype"])
        self.assertTrue("float" in profile["columns"]["income"]["dtype"])
        self.assertTrue("object" in profile["columns"]["name"]["dtype"] or "string" in profile["columns"]["name"]["dtype"])
        self.assertTrue("bool" in profile["columns"]["is_active"]["dtype"])
        self.assertTrue("datetime" in profile["columns"]["join_date"]["dtype"])
        
        # Check numeric stats
        self.assertEqual(profile["columns"]["id"]["min"], 1)
        self.assertEqual(profile["columns"]["id"]["max"], 5)
        self.assertEqual(profile["columns"]["age"]["min"], 25)
        self.assertEqual(profile["columns"]["income"]["max"], 70000.0)
        
        # Check sample values
        self.assertEqual(len(profile["columns"]["name"]["sample_values"]), min(5, len(self.df["name"].dropna())))
    
    def test_validate_constraints(self):
        """Test validating dataset against constraints."""
        # Define constraints
        constraints = {
            "id": {
                "type": "numeric",
                "required": True,
                "unique": True,
                "min": 1
            },
            "age": {
                "type": "numeric",
                "min": 18,
                "max": 100
            },
            "name": {
                "required": True
            },
            "is_active": {
                "allowed_values": [True, False]
            }
        }
        
        # Test against constraints
        results = validate_constraints(self.df, constraints)
        
        # Should fail due to missing values in required field (name)
        self.assertFalse(results["valid"])
        
        # Fix the issue and test again
        fixed_df = self.df.copy()
        fixed_df["name"] = fixed_df["name"].fillna("Unknown")
        fixed_df["is_active"] = fixed_df["is_active"].fillna(False)
        
        results = validate_constraints(fixed_df, constraints)
        self.assertTrue(results["valid"])
        
        # Test with invalid constraint
        invalid_constraints = {
            "id": {
                "min": 2  # First ID is 1, should fail
            }
        }
        
        results = validate_constraints(fixed_df, invalid_constraints)
        self.assertFalse(results["valid"])
    
    def test_generate_missing_data_plot(self):
        """Test generating missing data plot."""
        plot_data = generate_missing_data_plot(self.df)
        
        # Should return a base64 encoded image string
        self.assertTrue(isinstance(plot_data, str) or plot_data is None)
        
        # Create DataFrame with no missing values
        no_missing_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })
        
        plot_data_none = generate_missing_data_plot(no_missing_df)
        self.assertIsNone(plot_data_none)
    
    def test_generate_data_type_summary(self):
        """Test generating data type summary."""
        plot_data = generate_data_type_summary(self.df)
        
        # Should return a base64 encoded image string
        self.assertTrue(isinstance(plot_data, str))


if __name__ == '__main__':
    unittest.main()
