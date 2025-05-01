import subprocess
import sys
from pathlib import Path

def main():
    """Install dependencies"""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create virtual environment
    venv_dir = Path("venv")
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)])
    
    # Determine pip path
    pip_path = venv_dir / "bin" / "pip"
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip.exe"
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    print("Installation complete!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")

if __name__ == "__main__":
    main() 