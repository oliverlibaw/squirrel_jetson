#!/usr/bin/env python3
"""
Configuration examples for different Jetson and webcam setups.
Copy and modify these settings as needed for your specific hardware.
"""

# Example 1: Standard USB Webcam (most common)
STANDARD_USB_CONFIG = {
    "webcam_id": 0,
    "resolution": (1280, 720),
    "fps": 30,
    "gstreamer_pipeline": (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! appsink"
    )
}

# Example 2: High-resolution USB Webcam
HIGH_RES_USB_CONFIG = {
    "webcam_id": 0,
    "resolution": (1920, 1080),
    "fps": 30,
    "gstreamer_pipeline": (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=1920,height=1080,framerate=30/1 ! "
        "videoconvert ! appsink"
    )
}

# Example 3: Multiple Webcam Setup
MULTI_WEBCAM_CONFIG = {
    "webcam_id": 1,  # Second webcam
    "resolution": (1280, 720),
    "fps": 30,
    "gstreamer_pipeline": (
        "v4l2src device=/dev/video1 ! "
        "video/x-raw,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! appsink"
    )
}

# Example 4: Jetson CSI Camera (if available)
CSI_CAMERA_CONFIG = {
    "webcam_id": 0,
    "resolution": (1280, 720),
    "fps": 30,
    "gstreamer_pipeline": (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! appsink"
    )
}

# Example 5: Network Camera (IP Camera)
IP_CAMERA_CONFIG = {
    "webcam_id": "rtsp://username:password@ip_address:port/stream",
    "resolution": (1280, 720),
    "fps": 30,
    "gstreamer_pipeline": (
        "rtspsrc location=rtsp://username:password@ip_address:port/stream ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! appsink"
    )
}

# Example 6: Low-latency configuration for real-time tracking
LOW_LATENCY_CONFIG = {
    "webcam_id": 0,
    "resolution": (640, 480),  # Lower resolution for speed
    "fps": 60,
    "gstreamer_pipeline": (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=640,height=480,framerate=60/1 ! "
        "videoconvert ! appsink"
    )
}

# Example 7: High-quality configuration for detailed detection
HIGH_QUALITY_CONFIG = {
    "webcam_id": 0,
    "resolution": (1920, 1080),
    "fps": 15,  # Lower FPS for higher quality
    "gstreamer_pipeline": (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=1920,height=1080,framerate=15/1 ! "
        "videoconvert ! appsink"
    )
}

# Zone configuration examples
ZONE_CONFIGS = {
    # Standard left/right zones (50% each)
    "standard": {
        "right_zone_percentage": 0.5,  # Right 50%
        "left_zone_percentage": 0.5,   # Left 50%
        "zone_height_percentage": 1.0  # Full height
    },
    
    # Narrow crossing zones (30% each in center)
    "narrow": {
        "right_zone_percentage": 0.35,  # Right 35%
        "left_zone_percentage": 0.35,   # Left 35%
        "zone_height_percentage": 0.8   # 80% of height
    },
    
    # Wide crossing zones (70% each, overlapping)
    "wide": {
        "right_zone_percentage": 0.7,   # Right 70%
        "left_zone_percentage": 0.7,    # Left 70%
        "zone_height_percentage": 1.0   # Full height
    }
}

# Model optimization settings
MODEL_CONFIGS = {
    # Fast inference (lower accuracy, higher speed)
    "fast": {
        "confidence_threshold": 0.3,
        "iou_threshold": 0.5,
        "max_detections": 10
    },
    
    # Balanced (default settings)
    "balanced": {
        "confidence_threshold": 0.4,
        "iou_threshold": 0.45,
        "max_detections": 20
    },
    
    # High accuracy (higher accuracy, lower speed)
    "accurate": {
        "confidence_threshold": 0.6,
        "iou_threshold": 0.4,
        "max_detections": 50
    }
}

# Jetson-specific optimization settings
JETSON_CONFIGS = {
    # Jetson Orin Nano (8GB)
    "orin_nano_8gb": {
        "tensorrt_precision": "fp16",
        "max_batch_size": 1,
        "workspace_size": 1024,
        "gpu_memory_fraction": 0.8
    },
    
    # Jetson Orin Nano (4GB)
    "orin_nano_4gb": {
        "tensorrt_precision": "fp16",
        "max_batch_size": 1,
        "workspace_size": 512,
        "gpu_memory_fraction": 0.6
    },
    
    # Jetson Xavier NX
    "xavier_nx": {
        "tensorrt_precision": "fp16",
        "max_batch_size": 1,
        "workspace_size": 1024,
        "gpu_memory_fraction": 0.7
    }
}

def get_config_for_hardware(hardware_type="orin_nano_8gb"):
    """
    Get recommended configuration for specific hardware.
    
    Args:
        hardware_type: Type of Jetson device
        
    Returns:
        Dictionary with recommended settings
    """
    if hardware_type in JETSON_CONFIGS:
        return JETSON_CONFIGS[hardware_type]
    else:
        print(f"Unknown hardware type: {hardware_type}")
        print("Available types:", list(JETSON_CONFIGS.keys()))
        return JETSON_CONFIGS["orin_nano_8gb"]

def print_config_examples():
    """Print all available configuration examples."""
    print("Available Configuration Examples:")
    print("=" * 50)
    
    print("\n1. Standard USB Webcam:")
    print(f"   {STANDARD_USB_CONFIG}")
    
    print("\n2. High-Resolution USB Webcam:")
    print(f"   {HIGH_RES_USB_CONFIG}")
    
    print("\n3. Multiple Webcam Setup:")
    print(f"   {MULTI_WEBCAM_CONFIG}")
    
    print("\n4. Jetson CSI Camera:")
    print(f"   {CSI_CAMERA_CONFIG}")
    
    print("\n5. Network Camera (IP Camera):")
    print(f"   {IP_CAMERA_CONFIG}")
    
    print("\n6. Low-Latency Configuration:")
    print(f"   {LOW_LATENCY_CONFIG}")
    
    print("\n7. High-Quality Configuration:")
    print(f"   {HIGH_QUALITY_CONFIG}")
    
    print("\nZone Configurations:")
    for name, config in ZONE_CONFIGS.items():
        print(f"   {name}: {config}")
    
    print("\nModel Optimization Configurations:")
    for name, config in MODEL_CONFIGS.items():
        print(f"   {name}: {config}")
    
    print("\nJetson-Specific Configurations:")
    for name, config in JETSON_CONFIGS.items():
        print(f"   {name}: {config}")

if __name__ == "__main__":
    print_config_examples()


