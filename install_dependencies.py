#!/usr/bin/env python3
"""
Install missing dependencies for AQS Web Dashboard
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install required packages"""
    
    print("Installing AQS Dashboard Dependencies")
    print("=" * 40)
    
    # Essential packages for the dashboard
    packages = [
        "opencv-python",
        "streamlit", 
        "plotly",
        "pandas",
        "numpy"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
    
    print(f"\n🎉 Installation complete!")
    print(f"Now you can run: python run_web_dashboard.py")

if __name__ == "__main__":
    main()