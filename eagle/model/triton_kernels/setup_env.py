import os
import sys
import subprocess

def setup_environment():
    """
    Set up the environment for running the Triton kernel tests.
    This ensures that the package structure is properly recognized.
    """
    # Get the absolute path to the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    
    # Add the project root to the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Print environment information
    print(f"Project root: {project_root}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    
    # Check if Triton is installed
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton is not installed. Please install it with:")
        print("pip install triton")
        print("For more information, visit: https://github.com/openai/triton")
        return False
    
    # Check if PyTorch is installed
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch is not installed. Please install it with:")
        print("pip install torch")
        return False
    
    return True

def install_dependencies():
    """
    Install required dependencies for running the tests.
    """
    print("Installing required dependencies...")
    
    # Install Triton
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "triton"])
        print("Triton installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install Triton.")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up the environment for Triton kernel tests")
    parser.add_argument("--install", action="store_true", help="Install required dependencies")
    args = parser.parse_args()
    
    if args.install:
        success = install_dependencies()
        if not success:
            print("Failed to install dependencies.")
            sys.exit(1)
    
    success = setup_environment()
    if success:
        print("\nEnvironment set up successfully. You can now run the tests.")
        print("To run the tests, use: python run_tests.py")
    else:
        print("\nFailed to set up the environment.")
        sys.exit(1)