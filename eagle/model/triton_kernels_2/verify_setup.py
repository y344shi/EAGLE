"""
Script to verify that all Triton kernel files have been properly copied and updated.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def check_file_exists(file_path):
    """Check if a file exists and print the result."""
    exists = os.path.exists(file_path)
    print(f"{file_path}: {'✅' if exists else '❌'}")
    return exists


def check_import_path(file_path, old_path, new_path):
    """Check if a file has been updated to use the new import path."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    has_old_path = old_path in content
    has_new_path = new_path in content
    
    if has_old_path:
        print(f"{file_path}: ❌ Still contains old import path")
        return False
    elif has_new_path:
        print(f"{file_path}: ✅ Uses new import path")
        return True
    else:
        print(f"{file_path}: ⚠️ Neither old nor new import path found")
        return True  # Not a problem if the file doesn't use either path


def main():
    """Verify that all Triton kernel files have been properly copied and updated."""
    print("Checking if all files exist...")
    
    # List of files that should exist
    files = [
        "tree_attention.py",
        "topk_expand.py",
        "posterior_eval.py",
        "kv_block_copy.py",
        "mask_preparation.py",
        "integration.py",
        "ea_model_patch.py",
        "test_triton_kernels.py",
        "README_TRITON.md",
        "usage_example.py",
        "__init__.py",
        "benchmark.py",
        "run_tests.py",
        "README.md",
    ]
    
    # Check if all files exist
    all_exist = True
    for file in files:
        file_path = os.path.join("eagle/model/triton_kernels_2", file)
        if not check_file_exists(file_path):
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some files are missing!")
        return
    
    print("\nChecking if import paths have been updated...")
    
    # Files that should have updated import paths
    files_to_check = [
        "test_triton_kernels.py",
        "run_tests.py",
        "benchmark.py",
        "usage_example.py",
    ]
    
    # Check if import paths have been updated
    all_updated = True
    for file in files_to_check:
        file_path = os.path.join("eagle/model/triton_kernels_2", file)
        if not check_import_path(file_path, "eagle.model.triton_kernels", "eagle.model.triton_kernels_2"):
            all_updated = False
    
    if not all_updated:
        print("\n❌ Some import paths have not been updated!")
        return
    
    print("\n✅ All files have been properly copied and updated!")


if __name__ == "__main__":
    main()