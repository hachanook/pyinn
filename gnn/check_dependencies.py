#!/usr/bin/env python3
"""
Check if required dependencies are available
"""

import sys

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'torch_geometric',
        'numpy',
        'pandas',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print("pip install torch torch-geometric numpy pandas matplotlib")
        return False
    else:
        print("\n✓ All required packages are available!")
        return True

if __name__ == "__main__":
    print("=== Checking Dependencies ===")
    check_dependencies() 