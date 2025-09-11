#!/usr/bin/env python3
"""
Debug script to check YOLOv9c detection format
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def debug_yolov9c_detections():
    """Debug the YOLOv9c detection format"""
    
    # Configuration
    MODEL_PATH = "yolov9c-squirrel.pt"  # Use PyTorch model for debugging
    VIDEO_PATH = "squirrel_test_2.mp4"
    
    print("🔍 Debugging YOLOv9c Detection Format")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return False
    
    try:
        # Load model
        print(f"Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        # Open video
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {VIDEO_PATH}")
            return False
        
        print("✅ Video opened successfully")
        
        # Process frames until we find some detections
        frame_num = 0
        detections_found = 0
        max_frames_to_check = 100
        
        while frame_num < max_frames_to_check and detections_found < 3:
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"\n📋 Frame {frame_num + 1}:")
            
            # Run inference
            results = model(frame, verbose=False)[0]
            
            # Check results structure
            print(f"  Results type: {type(results)}")
            print(f"  Has boxes: {hasattr(results, 'boxes')}")
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes
                print(f"  Boxes type: {type(boxes)}")
                print(f"  Boxes data shape: {boxes.data.shape if boxes.data is not None else 'None'}")
                
                if boxes.data is not None and len(boxes.data) > 0:
                    data = boxes.data.cpu().numpy()
                    print(f"  Number of detections: {len(data)}")
                    print(f"  Detection format: {data.shape}")
                    print(f"  Sample detection: {data[0] if len(data) > 0 else 'None'}")
                    
                    detections_found += 1
                    
                    # Analyze each detection
                    for i, detection in enumerate(data[:3]):  # Show first 3 detections
                        print(f"    Detection {i+1}: {detection}")
                        if len(detection) == 6:
                            x1, y1, x2, y2, conf, cls = detection
                            print(f"      x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                            print(f"      confidence={conf:.3f}, class={int(cls)}")
                        elif len(detection) == 5:
                            x1, y1, x2, y2, conf = detection
                            print(f"      x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                            print(f"      confidence={conf:.3f}, class=unknown")
                        else:
                            print(f"      Unexpected format with {len(detection)} values")
                    
                    # Check if there are high-confidence detections
                    high_conf = data[data[:, 4] > 0.3]  # Confidence > 0.3
                    print(f"  High confidence detections (>0.3): {len(high_conf)}")
                    
                else:
                    print("  No detections found")
            else:
                print("  No boxes in results")
            
            frame_num += 1
        
        cap.release()
        print("\n✅ Debug completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        return False

def main():
    """Main function"""
    success = debug_yolov9c_detections()
    
    if success:
        print("\n💡 If you see detections with 6 values:")
        print("   Format: [x1, y1, x2, y2, confidence, class_id]")
        print("\n💡 If you see detections with 5 values:")
        print("   Format: [x1, y1, x2, y2, confidence] (single class model)")
        print("\n🔧 The squirrel tracker has been updated to handle both formats")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
