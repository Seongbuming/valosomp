#!/usr/bin/env python3
"""
Helper script to set up HuggingFace token
"""

import os
import sys
from pathlib import Path

def setup_token():
    """Help user set up HuggingFace token"""

    print("=== HuggingFace Token Setup ===\n")

    # Check if .env already exists
    env_path = Path(".env")
    if env_path.exists():
        print("⚠️  .env file already exists.")
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return

    print("\n📝 To get your HuggingFace token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'read' permissions")
    print("3. Accept Gemma terms at: https://huggingface.co/google/gemma-3-4b-it")
    print("\n")

    token = input("Enter your HuggingFace token: ").strip()

    if not token:
        print("❌ No token provided. Setup cancelled.")
        return

    # Write token to .env file
    with open(".env", "w") as f:
        f.write(f"# HuggingFace Token Configuration\n")
        f.write(f"HUGGINGFACE_TOKEN={token}\n")

    print("\n✅ Token saved to .env file!")
    print("You can now run the evaluation scripts.")

    # Verify token can be loaded
    try:
        from dotenv import load_dotenv
        load_dotenv()
        loaded_token = os.getenv("HUGGINGFACE_TOKEN")
        if loaded_token == token:
            print("✅ Token verification successful!")
        else:
            print("⚠️  Warning: Token verification failed. Please check .env file.")
    except ImportError:
        print("\n⚠️  Note: python-dotenv not installed yet. Run ./setup.sh to install dependencies.")

if __name__ == "__main__":
    setup_token()