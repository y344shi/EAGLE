# PowerShell script to run Triton kernel tests

# Set up environment
Write-Host "Setting up environment..."
python setup_env.py

# Check if setup was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to set up environment. Exiting."
    exit 1
}

# Run tests
Write-Host "`nRunning tests..."
python run_tests.py $args

# Check if tests were successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed. See output for details."
    exit 1
} else {
    Write-Host "All tests passed successfully!"
}