#!/usr/bin/env python3
"""
Launch the AQS Web Dashboard with all fixes applied
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    
    # Map package names to their import names
    package_imports = {
        'streamlit': 'streamlit',
        'torch': 'torch', 
        'opencv-python': 'cv2',
        'plotly': 'plotly',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    
    print("🚀 Launching AQS Web Dashboard...")
    print("=" * 50)
    print("✅ All fixes applied:")
    print("  - Fixed dilution effect (max aggregation)")
    print("  - Updated quality thresholds") 
    print("  - Fixed display values (actual impact)")
    print("  - Proper artifact detection")
    print("=" * 50)
    
    # Check if model exists
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print("   Dashboard will run with demo data")
    else:
        print(f"✅ Model found at {model_path}")
    
    print(f"\n🌐 Starting web server...")
    print(f"📱 Dashboard will open in your browser")
    print(f"🔗 URL: http://localhost:8502")
    print(f"\n💡 Features available:")
    print(f"  - Real-time video analysis")
    print(f"  - Interactive charts")
    print(f"  - Artifact breakdown")
    print(f"  - Quality metrics")
    print(f"  - Demo data for testing")
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "aqs_web_dashboard.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print(f"\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")

def main():
    """Main function"""
    
    print("AQS Web Dashboard Launcher")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    print("✅ All dependencies found")
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()