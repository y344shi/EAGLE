#!/bin/bash

# Set up environment
echo "Setting up environment..."
python setup_env.py

# Check if setup was successful
if [ $? -ne 0 ]; then
    echo "Failed to set up environment. Exiting."
    exit 1
fi

# Run tests
echo -e "\nRunning tests..."
python run_tests.py "$@"

# Check if tests were successful
if [ $? -ne 0 ]; then
    echo "Tests failed. See output for details."
    exit 1
else
    echo "All tests passed successfully!"
fi