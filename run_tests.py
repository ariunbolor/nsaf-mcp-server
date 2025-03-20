#!/usr/bin/env python
"""
Script to run tests for the NSAF prototype.
"""

import os
import sys
import pytest


def main():
    """
    Run tests for the NSAF prototype.
    """
    print("=" * 80)
    print("Running tests for the NSAF prototype")
    print("=" * 80)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run the tests
    args = ["-v", "tests/"]
    
    # Add any additional arguments passed to this script
    args.extend(sys.argv[1:])
    
    # Run pytest with the arguments
    exit_code = pytest.main(args)
    
    # Return the exit code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
