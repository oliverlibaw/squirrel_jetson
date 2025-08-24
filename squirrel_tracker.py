#!/usr/bin/env python3
"""
Real-Time Squirrel Tracker & Counter for NVIDIA Jetson Orin Nano

This application uses a live webcam feed to detect, track, and count squirrels
as they move from the right side of the frame to the left using YOLOv11s
and the Supervision library with TensorRT optimization.

Author: AI Assistant
Target Platform: NVIDIA Jetson Orin Nano
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

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
    
    def __init__(self, model_path: str, webcam_id: int = 0, confidence_threshold: float = 0.4):
        """
        Initialize the SquirrelTracker.
        
        Args:
            model_path: Path to the YOLOv11s model file (.pt or .engine)
            webcam_id: Webcam device ID (default: 0)
            confidence_threshold: Detection confidence threshold (default: 0.4)
        """
        self.model_path = model_path
        self.webcam_id = webcam_id
        self.confidence_threshold = confidence_threshold
        
        # Initialize tracking and counting variables
        self.tracker = sv.ByteTrack()
        self.squirrel_states: Dict[int, str] = {}  # tracker_id -> zone state
        self.squirrel_count = 0
        
        # Initialize video capture and model
        self.cap = None
        self.model = None
        self.frame_width = 1280
        self.frame_height = 720
        
        # Zone definitions (will be set dynamically)
        self.right_zone = None
        self.left_zone = None
        
        # Initialize the system
        self._initialize_model()
        self._initialize_video_capture()
        self._define_zones()
        
    def _initialize_model(self):
        """Initialize and optimize the YOLOv11s model with TensorRT."""
        try:
            print("Loading YOLOv11s model...")
            
            # Check if TensorRT engine exists
            engine_path = self.model_path.replace('.pt', '.engine')
            
            if os.path.exists(engine_path):
                print(f"Loading existing TensorRT engine: {engine_path}")
                self.model = YOLO(engine_path)
            else:
                print(f"Loading PyTorch model: {self.model_path}")
                self.model = YOLO(self.model_path)
                
                # Export to TensorRT for Jetson optimization
                print("Exporting model to TensorRT format for Jetson optimization...")
                self.model.export(format='engine', device=0, half=True)
                print(f"TensorRT engine saved to: {engine_path}")
                
                # Reload the TensorRT model
                self.model = YOLO(engine_path)
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def _initialize_video_capture(self):
        """Initialize video capture with GStreamer pipeline for Jetson optimization."""
        try:
            # GStreamer pipeline optimized for Jetson
            gst_pipeline = (
                f"v4l2src device=/dev/video{self.webcam_id} ! "
                "video/x-raw,width=1280,height=720,framerate=30/1 ! "
                "videoconvert ! appsink"
            )
            
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                print(f"Failed to open GStreamer pipeline. Trying standard capture...")
                self.cap = cv2.VideoCapture(self.webcam_id)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open webcam with ID {self.webcam_id}")
                
                # Set resolution for standard capture
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Get actual frame dimensions
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video capture initialized successfully!")
            print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
            
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
            polygon=right_zone_points,
            frame_resolution_wh=(self.frame_width, self.frame_height)
        )
        
        self.left_zone = sv.PolygonZone(
            polygon=left_zone_points,
            frame_resolution_wh=(self.frame_width, self.frame_height)
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
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        # Filter for squirrel class and confidence threshold
        squirrel_detections = []
        for detection in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Check if it's a squirrel (class_id 0 for custom model, adjust if needed)
            if confidence >= self.confidence_threshold:
                squirrel_detections.append([x1, y1, x2, y2, confidence, class_id])
        
        if squirrel_detections:
            detections = sv.Detections(
                xyxy=np.array(squirrel_detections),
                confidence=np.array([d[4] for d in squirrel_detections]),
                class_id=np.array([d[5] for d in squirrel_detections])
            )
        else:
            detections = sv.Detections.empty()
        
        return frame, detections
    
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
                    print(f"Squirrel {tracker_id} crossed from right to left! Total count: {self.squirrel_count}")
    
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
            thickness=2,
            text_thickness=2,
            text_scale=1
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
        
        # Annotate detections with tracker IDs
        if len(detections) > 0:
            labels = [f"Squirrel {int(tracker_id)}" for tracker_id in detections.tracker_id]
            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )
        
        # Add count overlay
        count_text = f"Squirrels Crossed: {self.squirrel_count}"
        cv2.putText(
            frame,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def run(self):
        """Main loop for processing video stream and tracking squirrels."""
        print("Starting squirrel tracking...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    break
                
                # Process detections
                frame, detections = self._process_detections(frame)
                
                # Track squirrels
                detections = self._track_squirrels(detections)
                
                # Check for zone crossing
                self._check_zone_crossing(detections)
                
                # Annotate frame
                annotated_frame = self._annotate_frame(frame, detections)
                
                # Display frame
                cv2.imshow("Squirrel Tracker", annotated_frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
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
        cv2.destroyAllWindows()
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
        required=True,
        help="Path to YOLOv11s model file (.pt or .engine)"
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=0,
        help="Webcam device ID (default: 0)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)"
    )
    
    args = parser.parse_args()
    
    # Validate model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create and run tracker
    tracker = SquirrelTracker(
        model_path=args.model,
        webcam_id=args.webcam,
        confidence_threshold=args.confidence
    )
    
    tracker.run()


if __name__ == "__main__":
    main()
