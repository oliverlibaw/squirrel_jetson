#!/usr/bin/env python3
"""
Convert YOLOv9c PyTorch model to TensorRT engine with INT8 quantization
"""

import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO
import subprocess

def convert_pt_to_engine_int8():
    """
    Convert YOLOv9c .pt model to TensorRT .engine with INT8 quantization
    """
    
    # Configuration
    PT_MODEL_PATH = "yolov9c_squirrel_08312025.pt"
    ENGINE_OUTPUT_PATH = "yolov9c_squirrel_08312025-int8.engine"
    CALIBRATION_DATA_DIR = "squirrel_calibration"
    CALIBRATION_YAML = "calibration_dataset.yaml"
    
    print("🚀 Starting YOLOv9c to TensorRT Engine conversion with INT8 quantization")
    print(f"📁 Input model: {PT_MODEL_PATH}")
    print(f"💾 Output engine: {ENGINE_OUTPUT_PATH}")
    print(f"📊 Calibration data: {CALIBRATION_DATA_DIR}")
    
    # Check if files exist
    if not os.path.exists(PT_MODEL_PATH):
        print(f"❌ Error: Model file {PT_MODEL_PATH} not found!")
        return False
    
    if not os.path.exists(CALIBRATION_DATA_DIR):
        print(f"❌ Error: Calibration directory {CALIBRATION_DATA_DIR} not found!")
        return False
    
    # Count calibration images
    calibration_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        calibration_images.extend(Path(CALIBRATION_DATA_DIR).glob(f"**/{ext}"))
    
    print(f"🖼️  Found {len(calibration_images)} calibration images")
    
    if len(calibration_images) < 10:
        print("⚠️  Warning: Less than 10 calibration images found. INT8 quantization may not be optimal.")
    
    try:
        print("🧠 Loading YOLO model...")
        start_time = time.time()
        model = YOLO(PT_MODEL_PATH)
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
        
        print("🔧 Starting TensorRT engine export with INT8 quantization...")
        export_start = time.time()
        
        # Export to TensorRT with INT8 quantization
        success = model.export(
            format='engine',                    # TensorRT format
            device='0',                        # GPU device
            half=False,                        # Don't use FP16 (we're using INT8)
            int8=True,                         # Enable INT8 quantization
            data=CALIBRATION_YAML,             # Calibration dataset configuration
            workspace=8,                       # TensorRT workspace size in GB
            verbose=True,                      # Verbose output
            batch=1,                          # Batch size
            imgsz=640                         # Input image size
        )
        
        export_time = time.time() - export_start
        print(f"✅ TensorRT engine export completed in {export_time:.2f} seconds")
        
        # The exported engine will have a default name, let's rename it
        default_engine_name = PT_MODEL_PATH.replace('.pt', '.engine')
        if os.path.exists(default_engine_name) and default_engine_name != ENGINE_OUTPUT_PATH:
            os.rename(default_engine_name, ENGINE_OUTPUT_PATH)
            print(f"📝 Renamed engine to {ENGINE_OUTPUT_PATH}")
        
        # Verify the engine file
        if os.path.exists(ENGINE_OUTPUT_PATH):
            file_size = os.path.getsize(ENGINE_OUTPUT_PATH) / (1024*1024)  # MB
            print(f"🎉 SUCCESS! Engine created: {ENGINE_OUTPUT_PATH}")
            print(f"📏 Engine file size: {file_size:.1f} MB")
            return True
        else:
            print(f"❌ Error: Engine file {ENGINE_OUTPUT_PATH} was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return False

def verify_engine_performance():
    """
    Quick test to verify the engine works and measure performance
    """
    ENGINE_PATH = "yolov9c_squirrel_08312025-int8.engine"
    
    if not os.path.exists(ENGINE_PATH):
        print(f"❌ Engine file {ENGINE_PATH} not found for verification")
        return
    
    try:
        print("\n🧪 Testing engine performance...")
        model = YOLO(ENGINE_PATH)
        
        # Test with a sample image
        test_image = "squirrel_test_2.mp4"  # Will use first frame
        if os.path.exists(test_image):
            start_time = time.time()
            results = model(test_image, verbose=False)
            inference_time = time.time() - start_time
            
            print(f"✅ Engine verification successful!")
            print(f"⚡ Inference time: {inference_time*1000:.1f}ms")
            print(f"🔍 Detections found: {len(results[0].boxes) if results[0].boxes else 0}")
        else:
            print("⚠️  No test image available for verification")
            
    except Exception as e:
        print(f"❌ Engine verification failed: {str(e)}")

def main():
    print("=" * 60)
    print("YOLOv9c to TensorRT Engine Converter with INT8 Quantization")
    print("=" * 60)
    
    # Convert model
    success = convert_pt_to_engine_int8()
    
    if success:
        # Verify performance
        verify_engine_performance()
        
        print("\n🎉 Conversion completed successfully!")
        print("📝 Next steps:")
        print("   1. Update your inference script to use the new engine:")
        print("      MODEL_ENGINE_PATH = 'yolov9c_squirrel_08312025-int8.engine'")
        print("   2. Test the new engine with your squirrel detection script")
        print("   3. Compare performance with the previous model")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
