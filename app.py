import streamlit as st
import os
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the ui.py file directly with streamlit
if __name__ == "__main__":
    import app.ui 