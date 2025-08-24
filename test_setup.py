#!/usr/bin/env python3
"""
Test script to verify the Jetson environment and dependencies for the Squirrel Tracker.
Run this script to check if your system is ready to run the main application.
"""

import sys
import subprocess
import platform

def test_python_version():
    """Test Python version compatibility."""
    print("Testing Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.10+ is required")
        return False

def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting package imports...")
    
    packages = [
        ('ultralytics', 'YOLO'),
        ('supervision', 'sv'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('cv2', 'cv2'),
        ('numpy', 'np'),
        ('PIL', 'PIL')
    ]
    
    all_imports_ok = True
    
    for package, import_name in packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ {package} imported successfully")
            elif package == 'numpy':
                import numpy as np
                print(f"✓ {package} imported successfully")
            elif package == 'PIL':
                from PIL import Image
                print(f"✓ {package} imported successfully")
            else:
                __import__(import_name)
                print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_cuda():
    """Test CUDA availability and PyTorch CUDA support."""
    print("\nTesting CUDA support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
            
            # Test CUDA tensor operations
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✓ CUDA tensor operations working")
            return True
        else:
            print("✗ CUDA is not available")
            return False
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_opencv_cuda():
    """Test OpenCV CUDA support."""
    print("\nTesting OpenCV CUDA support...")
    
    try:
        import cv2
        cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_device_count > 0:
            print(f"✓ OpenCV CUDA support available")
            print(f"  CUDA devices: {cuda_device_count}")
            return True
        else:
            print("✗ OpenCV CUDA support not available")
            return False
    except Exception as e:
        print(f"✗ OpenCV CUDA test failed: {e}")
        return False

def test_gstreamer():
    """Test GStreamer availability."""
    print("\nTesting GStreamer...")
    
    try:
        result = subprocess.run(['gst-launch-1.0', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ GStreamer is available")
            # Extract version from output
            version_line = result.stdout.split('\n')[0]
            print(f"  {version_line}")
            return True
        else:
            print("✗ GStreamer test failed")
            return False
    except FileNotFoundError:
        print("✗ GStreamer not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("✗ GStreamer test timed out")
        return False
    except Exception as e:
        print(f"✗ GStreamer test failed: {e}")
        return False

def test_tensorrt():
    """Test TensorRT availability."""
    print("\nTesting TensorRT...")
    
    try:
        import tensorrt as trt
        print(f"✓ TensorRT is available")
        print(f"  Version: {trt.__version__}")
        return True
    except ImportError:
        print("✗ TensorRT not available")
        return False
    except Exception as e:
        print(f"✗ TensorRT test failed: {e}")
        return False

def test_system_info():
    """Display system information."""
    print("\nSystem Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")
    
    # Check if running on Jetson
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            if 'jetson' in model.lower():
                print(f"  Device: {model}")
                print("✓ Running on Jetson platform")
            else:
                print(f"  Device: {model}")
                print("⚠ Not running on Jetson platform")
    except FileNotFoundError:
        print("  Device: Unknown (not Linux or no device tree)")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Jetson Squirrel Tracker - Environment Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("CUDA Support", test_cuda),
        ("OpenCV CUDA", test_opencv_cuda),
        ("GStreamer", test_gstreamer),
        ("TensorRT", test_tensorrt)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Display system info
    test_system_info()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready for the Squirrel Tracker.")
        print("\nNext steps:")
        print("1. Ensure you have a YOLOv11s model file (.pt format)")
        print("2. Connect your USB webcam")
        print("3. Run: python squirrel_tracker.py --model your_model.pt")
    else:
        print("❌ Some tests failed. Please fix the issues before running the Squirrel Tracker.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- For Jetson: Install PyTorch from NVIDIA wheels")
        print("- Install GStreamer: sudo apt-get install gstreamer1.0-*")
        print("- Check CUDA installation and GPU drivers")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
