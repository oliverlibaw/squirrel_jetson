#!/usr/bin/env python3
"""
Faster YOLOv9c to TensorRT conversion with optimized settings for Jetson
"""

import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

def convert_yolov9c_optimized():
    """
    Convert YOLOv9c to TensorRT with optimized settings for faster conversion
    """
    
    PT_MODEL_PATH = "yolov9c-squirrel.pt"
    ENGINE_OUTPUT_PATH = "yolov9c-squirrel-fp16.engine"
    
    print("🚀 YOLOv9c to TensorRT Engine (Optimized for Speed)")
    print(f"📁 Input: {PT_MODEL_PATH}")
    print(f"💾 Output: {ENGINE_OUTPUT_PATH}")
    
    if not os.path.exists(PT_MODEL_PATH):
        print(f"❌ Model file {PT_MODEL_PATH} not found!")
        return False
    
    try:
        print("🧠 Loading YOLO model...")
        start_time = time.time()
        model = YOLO(PT_MODEL_PATH)
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
        
        print("🔧 Starting optimized TensorRT export...")
        print("   Using FP16 precision for faster conversion")
        print("   Reducing workspace size for Jetson compatibility")
        
        export_start = time.time()
        
        # Export with optimized settings for Jetson
        success = model.export(
            format='engine',          # TensorRT format
            device='0',              # GPU device  
            half=True,               # Use FP16 for speed
            int8=False,              # Skip INT8 for now (faster conversion)
            workspace=4,             # Reduced workspace for Jetson (4GB instead of 8GB)
            verbose=False,           # Reduce verbose output
            batch=1,                # Single batch
            imgsz=640,              # Standard input size
            simplify=True,          # Simplify ONNX graph
            opset=11                # Use stable ONNX opset
        )
        
        export_time = time.time() - export_start
        print(f"✅ TensorRT engine export completed in {export_time:.2f} seconds")
        
        # Check for the created engine file
        default_engine = PT_MODEL_PATH.replace('.pt', '.engine')
        if os.path.exists(default_engine):
            if default_engine != ENGINE_OUTPUT_PATH:
                os.rename(default_engine, ENGINE_OUTPUT_PATH)
                print(f"📝 Renamed to {ENGINE_OUTPUT_PATH}")
            
            file_size = os.path.getsize(ENGINE_OUTPUT_PATH) / (1024*1024)
            print(f"🎉 SUCCESS! Engine created: {ENGINE_OUTPUT_PATH}")
            print(f"📏 File size: {file_size:.1f} MB")
            return True
        else:
            print(f"❌ Engine file not found")
            return False
            
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        return False

def quick_test():
    """Quick performance test"""
    ENGINE_PATH = "yolov9c-squirrel-fp16.engine"
    
    if not os.path.exists(ENGINE_PATH):
        return
        
    try:
        print("\n🧪 Quick performance test...")
        model = YOLO(ENGINE_PATH)
        
        # Create test image
        import numpy as np
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            _ = model(test_img, verbose=False)
        
        # Speed test
        start = time.time()
        for _ in range(10):
            results = model(test_img, verbose=False)
        avg_time = (time.time() - start) / 10
        
        print(f"✅ Average inference time: {avg_time*1000:.1f}ms")
        print(f"⚡ Estimated FPS: {1/avg_time:.1f}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def main():
    print("=" * 60)
    print("YOLOv9c TensorRT Converter - Optimized for Jetson")
    print("=" * 60)
    
    success = convert_yolov9c_optimized()
    
    if success:
        quick_test()
        print("\n🎉 Conversion completed!")
        print("📝 To use in your script:")
        print("   MODEL_ENGINE_PATH = 'yolov9c-squirrel-fp16.engine'")
        print("\n💡 Note: This uses FP16 precision.")
        print("   For INT8, run the original script when you have more time.")
    else:
        print("\n❌ Conversion failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

