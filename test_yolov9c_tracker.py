#!/usr/bin/env python3
"""
Test script for the updated squirrel tracker using YOLOv9c INT8 engine
"""

import os
import sys

def test_yolov9c_tracker():
    """Test the updated squirrel tracker with YOLOv9c"""
    
    # Configuration
    MODEL_PATH = "yolov9c-squirrel-int8.engine"
    VIDEO_PATH = "squirrel_test_2.mp4"
    
    print("🧪 Testing YOLOv9c Squirrel Tracker")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("💡 Please complete the INT8 conversion first:")
        print("   python convert_yolov9c_to_engine.py")
        return False
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return False
    
    print(f"✅ Model found: {MODEL_PATH}")
    print(f"✅ Video found: {VIDEO_PATH}")
    
    # Get model file size
    model_size = os.path.getsize(MODEL_PATH) / (1024*1024)  # MB
    print(f"📏 Model size: {model_size:.1f} MB")
    
    print("\n🚀 Starting squirrel tracker test...")
    print("Press 'q' to quit, 'p' to pause")
    
    try:
        # Import and run the tracker
        from squirrel_tracker import SquirrelTracker
        
        # Create tracker instance
        tracker = SquirrelTracker(
            model_path=MODEL_PATH,
            video_path=VIDEO_PATH,
            confidence_threshold=0.3,  # Optimized for YOLOv9c
            headless=False  # Show display
        )
        
        # Run the tracker
        tracker.run()
        
        print("✅ Test completed successfully!")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Main function"""
    print("YOLOv9c Squirrel Tracker Test")
    print("=" * 40)
    
    success = test_yolov9c_tracker()
    
    if success:
        print("\n🎉 YOLOv9c tracker is ready for use!")
        print("\n📝 Usage examples:")
        print("   # Use default INT8 engine:")
        print("   python squirrel_tracker.py --video squirrel_test_2.mp4")
        print("")
        print("   # Specify custom model:")
        print("   python squirrel_tracker.py --model yolov9c-squirrel-int8.engine --video squirrel_test_2.mp4")
        print("")
        print("   # Headless mode:")
        print("   python squirrel_tracker.py --video squirrel_test_2.mp4 --headless")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
