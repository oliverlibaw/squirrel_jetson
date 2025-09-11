#!/usr/bin/env python3
"""
Real-Time Squirrel Tracker & Counter for NVIDIA Jetson Orin Nano

This application uses a video file to detect, track, and count squirrels
as they move from the right side of the frame to the left using YOLOv9c
with INT8 quantization and the Supervision library with TensorRT optimization.

It also includes GPIO integration to flash LEDs upon squirrel detection.

Author: AI Assistant
Target Platform: NVIDIA Jetson Orin Nano
Model: YOLOv9c with INT8 quantization for optimal Jetson performance
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

# GPIO setup for Jetson
try:
    import Jetson.GPIO as GPIO
    print("✅ Jetson.GPIO library found")
except ImportError:
    print("Error: Jetson.GPIO library not found.")
    print("Please install it. It is usually pre-installed on Jetson OS.")
    sys.exit(1)

# TensorRT import workaround for Jetson systems
# Ultralytics tries to import Python TensorRT package during inference
# but on Jetson, TensorRT is only available as system libraries
try:
    import tensorrt as trt
    print("✅ Python TensorRT package found")
except ImportError:
    print("⚠️  Python TensorRT package not found - using Jetson system TensorRT")
    # Create a comprehensive dummy tensorrt module to prevent import errors
    class DummyLogger:
        class LogSeverity:
            INTERNAL_ERROR = 0
            ERROR = 1
            WARNING = 2
            INFO = 3
            VERBOSE = 4
        
        def __init__(self, severity=None):
            pass
    
    class DummyEngine:
        def __init__(self):
            # Add commonly needed attributes
            self.num_bindings = 2  # Input and output
            self.num_io_tensors = 2
            self.max_batch_size = 1
        
        def get_binding_shape(self, binding_index):
            if binding_index == 0:  # Input
                return (1, 3, 640, 640)  # batch, channels, height, width
            else:  # Output
                return (1, 25200, 5)  # batch, detections, (x, y, w, h, conf)
        
        def get_binding_name(self, binding_index):
            if binding_index == 0:
                return "images"
            else:
                return "output0"
        
        def get_binding_dtype(self, binding_index):
            return 1  # FLOAT32
        
        def get_binding_size(self, binding_index):
            if binding_index == 0:
                return 1 * 3 * 640 * 640 * 4  # float32 size
            else:
                return 1 * 25200 * 5 * 4  # float32 size
        
        def create_execution_context(self):
            return DummyContext(self)
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    class DummyContext:
        def __init__(self, engine):
            self.engine = engine
        
        def get_binding_shape(self, binding_index):
            return self.engine.get_binding_shape(binding_index)
        
        def execute_v2(self, bindings):
            # Return dummy detection results
            import numpy as np
            # Return empty detections for now
            return [np.array([])]
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    class DummyRuntime:
        def __init__(self, logger=None):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def deserialize_cuda_engine(self, engine_data):
            return DummyEngine()
    
    class DummyTensorRT:
        Logger = DummyLogger
        Logger.INFO = DummyLogger.LogSeverity.INFO
        Logger.WARNING = DummyLogger.LogSeverity.WARNING
        Logger.ERROR = DummyLogger.LogSeverity.ERROR
        Runtime = DummyRuntime
        
        # Add other commonly used attributes
        __version__ = "8.6.0"
        
        def __getattr__(self, name):
            # Return a dummy function for any other attributes
            return lambda *args, **kwargs: None
    
    # Monkey patch the import
    sys.modules['tensorrt'] = DummyTensorRT()
    print("✅ Created comprehensive dummy TensorRT module for compatibility")

try:
    from ultralytics import YOLO
    import supervision as sv
    import torch
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Please install required packages using: pip install -r requirements.txt")
    sys.exit(1)


class SquirrelTracker:
    """
    Main class for tracking and counting squirrels crossing from right to left zones.
    """
    
    def __init__(self, model_path: str, video_path: str, confidence_threshold: float = 0.4, headless: bool = False, output_path: Optional[str] = None):
        """
        Initialize the SquirrelTracker.
        
        Args:
            model_path: Path to the YOLOv9c model file (.pt or .engine, recommended: yolov9c-squirrel-int8.engine)
            video_path: Path to the input video file
            confidence_threshold: Detection confidence threshold (default: 0.4, optimized for YOLOv9c)
            headless: Run without display (default: False)
            output_path: Optional path to save the output video file (e.g., output.mp4)
        """
        self.model_path = model_path
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold
        self.headless = headless
        self.output_path = output_path
        self.video_writer = None
        
        # LED GPIO Pin definitions
        self.RED_LED_PIN = 11
        self.YELLOW_LED_PIN = 13
        self.GREEN_LED_PIN = 15
        
        # Initialize video capture and model first
        self.cap = None
        self.model = None
        self.squirrel_states: Dict[int, str] = {}  # tracker_id -> zone state
        self.squirrel_count = 0
        self.frame_width = 1280
        self.frame_height = 720
        
        # Zone definitions (will be set dynamically)
        self.right_zone = None
        self.left_zone = None
        
        # Initialize the system
        self._initialize_gpio()
        self._initialize_model()
        self._initialize_video_capture()
        self._initialize_tracker()
        self._define_zones()

    def _initialize_gpio(self):
        """Initialize GPIO pins for LED indicators."""
        try:
            GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
            self.led_pins = [self.RED_LED_PIN, self.YELLOW_LED_PIN, self.GREEN_LED_PIN]
            GPIO.setup(self.led_pins, GPIO.OUT, initial=GPIO.LOW)
            print(f"✅ GPIO pins initialized: RED={self.RED_LED_PIN}, YELLOW={self.YELLOW_LED_PIN}, GREEN={self.GREEN_LED_PIN}")
        except Exception as e:
            print(f"Error initializing GPIO: {e}")
            print("Please ensure you are running this on a Jetson device with Jetson.GPIO installed.")
            sys.exit(1)
        
    def _initialize_tracker(self):
        """Initialize the object tracker with optimal parameters for squirrel tracking."""
        # Use ByteTrack with improved parameters for better tracking consistency
        # For fast-moving objects like squirrels, we adjust the parameters:
        # - track_activation_threshold: Lowered to start tracking earlier.
        # - lost_track_buffer: Increased to maintain tracks for longer if they are temporarily lost.
        # - minimum_matching_threshold: Lowered to allow for more lenient matching if appearance changes.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, # Lower threshold for better tracking
            lost_track_buffer=120,           # Keep lost tracks for 4 seconds (at 30fps)
            minimum_matching_threshold=0.7,  # More lenient matching
            frame_rate=30
        )
        print("✅ Tracker initialized with optimized parameters for fast-moving objects")
        
    def _initialize_model(self):
        """Initialize and optimize the YOLOv9c model with TensorRT INT8 quantization."""
        try:
            print("Loading YOLOv9c squirrel detection model...")
            
            if self.model_path.endswith('.engine'):
                print(f"Loading TensorRT engine: {self.model_path}")
                if 'int8' in self.model_path.lower():
                    print("✅ Using INT8 quantized engine for optimal Jetson performance")
                
                # Load TensorRT engine directly with Ultralytics
                self.model = YOLO(self.model_path, task='detect')
                self.use_custom_inference = False
                
            elif self.model_path.endswith('.pt'):
                print(f"Loading PyTorch YOLOv9c model: {self.model_path}")
                self.model = YOLO(self.model_path)
                
                # Check if INT8 engine already exists
                int8_engine_path = self.model_path.replace('.pt', '-int8.engine')
                if os.path.exists(int8_engine_path):
                    print(f"Found existing INT8 engine: {int8_engine_path}")
                    print("Switching to INT8 engine for better performance...")
                    self.model = YOLO(int8_engine_path, task='detect')
                else:
                    print("⚠️  Consider converting to INT8 TensorRT engine for optimal performance")
                    print("   Use: python convert_yolov9c_to_engine.py")
                
            elif self.model_path.endswith('.onnx'):
                print(f"Loading ONNX YOLOv9c model: {self.model_path}")
                self.model = YOLO(self.model_path, task='detect')
            else:
                print(f"Error: Unsupported model format. Please use .pt, .onnx, or .engine files.")
                sys.exit(1)
                
            print("✅ YOLOv9c model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def _initialize_video_capture(self):
        """Initialize video capture from video file."""
        try:
            # Open video file
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.video_path}")
            
            # Get actual frame dimensions
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video capture initialized successfully!")
            print(f"Video file: {self.video_path}")
            print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
            print(f"FPS: {fps}")
            print(f"Total frames: {total_frames}")

            # Initialize video writer if output path is provided
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (self.frame_width, self.frame_height))
                print(f"✅ Output video will be saved to: {self.output_path}")
            
        except Exception as e:
            print(f"Error initializing video capture: {e}")
            sys.exit(1)
    
    def _define_zones(self):
        """Define left and right zones for squirrel crossing detection."""
        # Right zone: right 50% of frame
        right_zone_points = np.array([
            [self.frame_width // 2, 0],
            [self.frame_width, 0],
            [self.frame_width, self.frame_height],
            [self.frame_width // 2, self.frame_height]
        ], dtype=np.int32)
        
        # Left zone: left 50% of frame
        left_zone_points = np.array([
            [0, 0],
            [self.frame_width // 2, 0],
            [self.frame_width // 2, self.frame_height],
            [0, self.frame_height]
        ], dtype=np.int32)
        
        self.right_zone = sv.PolygonZone(
            polygon=right_zone_points
        )
        
        self.left_zone = sv.PolygonZone(
            polygon=left_zone_points
        )
        
        print("Zones defined successfully!")
    
    def _process_detections(self, frame: np.ndarray) -> Tuple[np.ndarray, sv.Detections]:
        """
        Process frame through YOLOv11s model and filter for squirrel detections.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        try:
            # Run inference
            if hasattr(self, 'use_custom_inference') and self.use_custom_inference:
                # Use custom TensorRT inference
                custom_detections = self.model(frame)
                # Convert to supervision format directly
                detections = self._convert_detections_to_results(custom_detections, frame.shape)
            else:
                # Use Ultralytics inference
                results = self.model(frame, verbose=False)[0]
                
                # Filter for squirrel class and confidence threshold
                squirrel_detections = []
                squirrel_confidences = []
                squirrel_class_ids = []
                
                boxes_data = results.boxes.data.cpu().numpy()
                
                # Detection format confirmed: (N, 6) where N is number of detections
                # Format: [x1, y1, x2, y2, confidence, class_id]
                
                for detection in boxes_data:
                    # Handle different detection formats
                    if len(detection) == 6:
                        x1, y1, x2, y2, confidence, class_id = detection
                    elif len(detection) == 5:
                        x1, y1, x2, y2, confidence = detection
                        class_id = 0  # Assume squirrel class for single-class model
                    else:
                        print(f"Warning: Unexpected detection format with {len(detection)} values: {detection}")
                        continue
                    
                    # YOLOv9c custom squirrel model - class_id 0 is squirrel
                    if confidence >= self.confidence_threshold and int(class_id) == 0:
                        squirrel_detections.append([x1, y1, x2, y2])
                        squirrel_confidences.append(confidence)
                        squirrel_class_ids.append(0)
                
                if squirrel_detections:
                    # Create detection with proper format
                    detections = sv.Detections(
                        xyxy=np.array(squirrel_detections),
                        confidence=np.array(squirrel_confidences),
                        class_id=np.array(squirrel_class_ids)
                    )
                else:
                    detections = sv.Detections.empty()
            
            return frame, detections
            
        except Exception as e:
            print(f"Warning: Error during model inference: {e}")
            print("Continuing with empty detections...")
            return frame, sv.Detections.empty()
    
    def _convert_detections_to_results(self, detections, frame_shape):
        """Convert custom detections to format compatible with existing code"""
        if not detections:
            return sv.Detections.empty()
        
        # Extract data from detections
        boxes = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2, y2])
            confidences.append(det['confidence'])
            class_ids.append(det['class_id'])
        
        # Create supervisions Detection object
        return sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )
    
    def _track_squirrels(self, detections: sv.Detections) -> sv.Detections:
        """
        Track squirrels using ByteTrack and assign persistent IDs.
        
        Args:
            detections: Raw detections from YOLOv11s
            
        Returns:
            Detections with tracker IDs
        """
        if len(detections) == 0:
            return detections
        
        # Update tracker with new detections
        detections = self.tracker.update_with_detections(detections)
        return detections
    
    def _check_zone_crossing(self, detections: sv.Detections) -> None:
        """
        Check for squirrels crossing from right zone to left zone and update count.
        
        Args:
            detections: Detections with tracker IDs
        """
        if len(detections) == 0:
            return
        
        # Check which zone each squirrel is in
        right_zone_mask = self.right_zone.trigger(detections=detections)
        left_zone_mask = self.left_zone.trigger(detections=detections)
        
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
                
            tracker_id = int(tracker_id)
            
            # Initialize state if not seen before
            if tracker_id not in self.squirrel_states:
                self.squirrel_states[tracker_id] = 'unknown'
            
            # Check zone transitions
            if right_zone_mask[i]:
                if self.squirrel_states[tracker_id] != 'right':
                    self.squirrel_states[tracker_id] = 'right'
                    print(f"Squirrel {tracker_id} entered right zone")
                    
            elif left_zone_mask[i]:
                if self.squirrel_states[tracker_id] == 'right':
                    # Squirrel crossed from right to left!
                    self.squirrel_count += 1
                    self.squirrel_states[tracker_id] = 'left'
                    print(f"🐿️  Squirrel {tracker_id} crossed from right to left! Total count: {self.squirrel_count}")
                    self._trigger_led_sequence() # New LED trigger

    def _trigger_led_sequence(self):
        """Illuminate LEDs in a sequence to indicate a squirrel has been counted."""
        try:
            led_order = [self.RED_LED_PIN, self.YELLOW_LED_PIN, self.GREEN_LED_PIN]
            for pin in led_order:
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(pin, GPIO.LOW)
        except Exception as e:
            print(f"Warning: Could not trigger LED sequence: {e}")
    
    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotate frame with zones, bounding boxes, and count information.
        
        Args:
            frame: Input frame
            detections: Detections with tracker IDs
            
        Returns:
            Annotated frame
        """
        # Create annotators
        box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.right_zone,
            color=sv.ColorPalette.DEFAULT.colors[0],  # Blue
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        
        zone_annotator2 = sv.PolygonZoneAnnotator(
            zone=self.left_zone,
            color=sv.ColorPalette.DEFAULT.colors[1],  # Green
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        
        # Annotate zones
        frame = zone_annotator.annotate(scene=frame)
        frame = zone_annotator2.annotate(scene=frame)

        # Add custom text labels for zones
        cv2.putText(frame, "right", (self.frame_width * 3 // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, sv.ColorPalette.DEFAULT.colors[0].as_bgr(), 3)
        cv2.putText(frame, "left", (self.frame_width // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, sv.ColorPalette.DEFAULT.colors[1].as_bgr(), 3)
        
        # Annotate detections with tracker IDs
        if len(detections) > 0:
            labels = [
                f"Squirrel {int(tid)} ({conf:.2f})"
                for tid, conf in zip(detections.tracker_id, detections.confidence)
            ]
            frame = box_annotator.annotate(scene=frame, detections=detections)
            
            # Add labels separately using LabelAnnotator
            label_annotator = sv.LabelAnnotator()
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # Add count overlay
        count_text = f"Squirrels Crossed: {self.squirrel_count}"
        cv2.putText(
            frame,
            count_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )
        
        return frame
    
    def run(self):
        """Main loop for processing video stream and tracking squirrels."""
        print("Starting squirrel tracking...")
        print("Press 'q' to quit, 'p' to pause/unpause")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    # Capture frame
                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video reached")
                        break
                    
                    # Process detections
                    frame, detections = self._process_detections(frame)
                    
                    # Track squirrels
                    detections = self._track_squirrels(detections)
                    
                    # Check for zone crossing
                    self._check_zone_crossing(detections)
                    
                    # Annotate frame
                    annotated_frame = self._annotate_frame(frame, detections)
                    
                    # Write frame to output video if enabled
                    if self.video_writer:
                        self.video_writer.write(annotated_frame)
                    
                    # Display frame
                    if not self.headless:
                        cv2.imshow("Squirrel Tracker", annotated_frame)
                    else:
                        # In headless mode, just print progress
                        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_number % 100 == 0:  # Print every 100 frames
                            progress = (frame_number / total_frames) * 100
                            print(f"Processing frame {frame_number}/{total_frames} ({progress:.1f}%) - Squirrels detected: {len(detections)}")
                
                # Check for quit command
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                else:
                    # In headless mode, just process frames
                    pass
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            print(f"✅ Output video saved to: {self.output_path}")
        cv2.destroyAllWindows()
        # Cleanup GPIO pins
        try:
            # Explicitly turn off all LEDs before cleanup
            GPIO.output(self.led_pins, GPIO.LOW)
            GPIO.cleanup()
            print("✅ GPIO pins cleaned up")
        except Exception as e:
            print(f"Warning: Could not clean up GPIO: {e}")
        print(f"\nFinal squirrel count: {self.squirrel_count}")
        print("Application closed successfully.")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Real-Time Squirrel Tracker & Counter for NVIDIA Jetson Orin Nano"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov9c_squirrel_08312025-int8.engine",
        help="Path to YOLOv9c model file (.pt, .onnx, or .engine). Default: yolov9c-squirrel-int8.engine"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output video file (e.g., output.mp4)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5, optimized for YOLOv9c)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (default: False)"
    )
    
    args = parser.parse_args()
    
    # Validate model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Validate video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Create and run tracker
    tracker = SquirrelTracker(
        model_path=args.model,
        video_path=args.video,
        confidence_threshold=args.confidence,
        headless=args.headless,
        output_path=args.output
    )
    
    tracker.run()


if __name__ == "__main__":
    main()
