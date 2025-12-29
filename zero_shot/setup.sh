#!/bin/bash

echo "=== Setting up Gemma-3 Tweet Evaluation Environment ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the evaluation:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Test with sample tweets: python test_gemma_simple.py"
echo "3. Run full evaluation: python evaluate_tweets_gemma.py"
echo ""
echo "Note: The first run will download the Gemma-3 model (several GB)."
echo "Make sure you have sufficient disk space and a good internet connection."