"""
Script to run all tests for EAGLE Triton kernels.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def run_test(test_name, args=None):
    """
    Run a test and print the result.
    
    Args:
        test_name: Name of the test module
        args: Additional arguments to pass to the test
    """
    print(f"Running {test_name}...")
    cmd = [sys.executable, "-m", f"eagle.model.triton_kernels.{test_name}"]
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
    # Check if model path is provided
    if len(sys.argv) < 2:
        print("Usage: python -m eagle.model.triton_kernels.run_tests <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Run tests
    tests = [
        ("test_triton_kernels", ["--model_path", model_path, "--prompt", "Hello, world!", "--max_new_tokens", "20"]),
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