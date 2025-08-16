"""
Script to run all tests for EAGLE Triton kernels.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the root directory to the Python path
# Get the absolute path to the project root (3 levels up from this file)
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, project_root)


def run_test(test_name, args=None):
    """
    Run a test and print the result.
    
    Args:
        test_name: Name of the test module
        args: Additional arguments to pass to the test
    """
    print(f"Running {test_name}...")
    cmd = [sys.executable, "-m", f"eagle.model.triton_kernels_2.{test_name}"]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {test_name} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {test_name} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """
    Run all tests for EAGLE Triton kernels.
    """
    # Check if model paths are provided
    if len(sys.argv) < 3:
        print("Usage: python -m eagle.model.triton_kernels_2.run_tests <base_model_path> <ea_model_path>")
        sys.exit(1)
    
    base_model_path = sys.argv[1]
    ea_model_path = sys.argv[2]
    
    # Run tests
    tests = [
        ("test_triton_kernels", [
            "--base_model_path", base_model_path,
            "--ea_model_path", ea_model_path,
            "--prompt", "Hello, world!",
            "--max_new_tokens", "20"
        ]),
    ]
    
    success_count = 0
    for test_name, args in tests:
        if run_test(test_name, args):
            success_count += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Test Summary: {success_count}/{len(tests)} tests passed")
    print("=" * 80)
    
    # Exit with appropriate status code
    sys.exit(0 if success_count == len(tests) else 1)


if __name__ == "__main__":
    main()