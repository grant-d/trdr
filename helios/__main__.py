#!/usr/bin/env python3
"""
Helios Trading Analysis Toolkit
Main entry point for command line execution
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from main import main

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Run main function
    sys.exit(main())
