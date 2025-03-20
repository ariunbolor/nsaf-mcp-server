#!/usr/bin/env python
"""
Script to run the example for the NSAF prototype.
"""

import os
import sys
import argparse


def main():
    """
    Run the example for the NSAF prototype.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run NSAF prototype examples")
    parser.add_argument("--scma", action="store_true", help="Run the SCMA example only")
    parser.add_argument("--comparison", action="store_true", help="Run the agent comparison example only")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations for evolution")
    parser.add_argument("--population", type=int, default=20, help="Population size for evolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Running example for the NSAF prototype")
    print("=" * 80)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Import the main module
    sys.path.insert(0, script_dir)
    from main import run_scma_example, run_agent_comparison_example
    
    # Set environment variables based on arguments
    os.environ["NSAF_GENERATIONS"] = str(args.generations)
    os.environ["NSAF_POPULATION_SIZE"] = str(args.population)
    os.environ["NSAF_RANDOM_SEED"] = str(args.seed)
    
    # Run the examples
    if args.scma or not (args.scma or args.comparison):
        run_scma_example()
    
    if args.comparison or not (args.scma or args.comparison):
        print("\n" + "=" * 80 + "\n")
        run_agent_comparison_example()
    
    print("\nExamples completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
