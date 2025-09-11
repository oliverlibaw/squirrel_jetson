#!/usr/bin/env python3
"""
Simple test script for the new yolov9c_squirrel_08312025-int8.engine
"""

import cv2
import time
from ultralytics import YOLO

def test_engine():
    print("Testing yolov9c_squirrel_08312025-int8.engine...")
    
    # Load model
    model = YOLO("yolov9c_squirrel_08312025-int8.engine")
    print("✅ Model loaded successfully")
    
    # Open video
    cap = cv2.VideoCapture("squirrel_test_2.mp4")
    if not cap.isOpened():
        print("❌ Error opening video")
        return
    
    print("✅ Video opened successfully")
    
    frame_count = 0
    detection_count = 0
    total_inference_time = 0
    
    print("🎬 Processing frames...")
    
    # Process first 100 frames
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run inference
        start_time = time.time()
        results = model(frame, verbose=False)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Count detections
        if results[0].boxes is not None:
            detections = len(results[0].boxes)
            if detections > 0:
                detection_count += detections
                print(f"Frame {frame_count}: {detections} squirrel(s) detected")
        
        # Print progress every 25 frames
        if frame_count % 25 == 0:
            avg_fps = frame_count / total_inference_time
            print(f"Processed {frame_count} frames - Average FPS: {avg_fps:.1f}")
    
    cap.release()
    
    # Final stats
    avg_fps = frame_count / total_inference_time
    avg_inference_time = (total_inference_time / frame_count) * 1000
    
    print("\n📊 Test Results:")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {detection_count}")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average inference time: {avg_inference_time:.1f}ms")
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_engine()
