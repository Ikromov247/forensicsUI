import sys
import subprocess
from pathlib import Path

def main():
    """Run all tests"""
    # Check if tests directory exists
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("Error: Tests directory not found")
        sys.exit(1)
    
    # Run tests
    print("Running tests...")
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "tests", "-v"])
    
    # Return appropriate exit code
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 