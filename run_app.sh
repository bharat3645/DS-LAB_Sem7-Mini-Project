#!/bin/bash

# Crime Analytics Dashboard - Quick Start Script

echo "========================================"
echo "Crime Analytics Dashboard - Starting..."
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if required CSV files exist
if [ ! -f "districtwise-missing-persons-2017-2022.csv" ]; then
    echo "❌ Missing file: districtwise-missing-persons-2017-2022.csv"
    echo "Please ensure the CSV file is in the current directory."
    exit 1
fi

if [ ! -f "districtwise-ipc-crime-by-juveniles-2017-onwards.csv" ]; then
    echo "❌ Missing file: districtwise-ipc-crime-by-juveniles-2017-onwards.csv"
    echo "Please ensure the CSV file is in the current directory."
    exit 1
fi

echo "✓ Data files found"
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null
then
    echo "📦 Installing required packages..."
    pip install -q streamlit pandas numpy matplotlib seaborn scikit-learn plotly
    echo "✓ Packages installed"
else
    echo "✓ Streamlit is already installed"
fi

echo ""
echo "🚀 Starting Streamlit application..."
echo ""
echo "📊 Dashboard will open in your browser automatically"
echo "   If not, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

# Run streamlit
streamlit run streamlit_app.py
