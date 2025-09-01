#!/usr/bin/env python
"""
Script to run the Triton drafter tests with proper Python path setup.
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import torch

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

def run_test(test_module, verbose=False):
    """Run a specific test module."""
    print(f"\n{'='*80}\nRunning {test_module}\n{'='*80}")
    cmd = [sys.executable, "-m", test_module]
    if verbose:
        cmd.append("-v")
    
    result = subprocess.run(cmd, capture_output=not verbose)
    if result.returncode != 0:
        print(f"❌ Test failed: {test_module}")
        if not verbose:
            print(result.stdout.decode())
            print(result.stderr.decode())
        return False
    else:
        print(f"✅ Test passed: {test_module}")
        return True

def run_benchmark(model_size="8B", runs=5, no_plot=False):
    """Run the benchmark script."""
    print(f"\n{'='*80}\nRunning benchmark (model_size={model_size}, runs={runs})\n{'='*80}")
    cmd = [
        sys.executable, 
        "-m", 
        "eagle.model.triton_drafttoken_gen.benchmark",
        "--model-size", model_size,
        "--runs", str(runs)
    ]
    if no_plot:
        cmd.append("--no-plot")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EAGLE Triton drafter tests")
    parser.add_argument("--test", choices=["all", "unit", "integration", "benchmark"], default="all", 
                        help="Which tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--model-size", choices=["8B", "70B"], default="8B", 
                        help="Model size for benchmarks")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--no-plot", action="store_true", help="Don't generate benchmark plots")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA is not available. Some tests may be skipped.")
    
    # Check if Triton is available
    try:
        import triton
        print(f"✅ Triton is available (version: {triton.__version__})")
    except ImportError:
        print("⚠️ Warning: Triton is not available. Some tests may be skipped.")
    
    all_passed = True
    
    # Run tests based on the selected option
    if args.test in ["all", "unit"]:
        all_passed &= run_test("eagle.model.triton_drafttoken_gen.test_drafter", args.verbose)
    
    if args.test in ["all", "integration"]:
        all_passed &= run_test("eagle.model.triton_drafttoken_gen.integration_test_drafter", args.verbose)
    
    if args.test in ["all", "benchmark"]:
        run_benchmark(args.model_size, args.runs, args.no_plot)
    
    # Print summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)