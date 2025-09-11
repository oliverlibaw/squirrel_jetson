import cv2
import supervision as sv
import time
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# Path to your TensorRT engine file (GPU optimized) 
MODEL_ENGINE_PATH = "squirrel_detector_yolov11n_5-2025.engine"

# Path to your source video file
SOURCE_VIDEO_PATH = "squirrel_test_2.mp4"

# Path where the output video will be saved
OUTPUT_VIDEO_PATH = "squirrel_tracking_output.mp4"

# Define the virtual line for tracking.
# Line positioned at center of frame to detect left-to-right movement
# Will be dynamically positioned based on actual video dimensions
# Format: (start_x, start_y), (end_x, end_y)
LINE_START = None  # Will be set based on video dimensions
LINE_END = None    # Will be set based on video dimensions

# --- Main Script ---

# This dictionary will store the tracker IDs of squirrels that have crossed
crossed_squirrels = {}

# Track recent crossings for visual display
recent_crossings = []

def process_line_crossings(detections, crossed_in, crossed_out, frame_number):
    """
    Process line crossings and update the crossed squirrels count.
    """
    # Process squirrels that crossed from outside to inside (left to right)
    for i, crossed in enumerate(crossed_in):
        if crossed and detections.tracker_id is not None:
            tracker_id = detections.tracker_id[i]
            # Check if we've already counted this squirrel to avoid double-counting
            if tracker_id not in crossed_squirrels:
                crossed_squirrels[tracker_id] = True
                # Add to recent crossings for visual display (keep last 30 frames = ~1 second at 30fps)
                recent_crossings.append({'tracker_id': tracker_id, 'frame': frame_number, 'direction': 'LEFT_TO_RIGHT'})
                print(f"🐿️  CROSSING DETECTED! Squirrel ID #{tracker_id} crossed LEFT to RIGHT at frame {frame_number}")
                
    # Also log squirrels crossing right to left for completeness
    for i, crossed in enumerate(crossed_out):
        if crossed and detections.tracker_id is not None:
            tracker_id = detections.tracker_id[i]
            recent_crossings.append({'tracker_id': tracker_id, 'frame': frame_number, 'direction': 'RIGHT_TO_LEFT'})
            print(f"↩️  Squirrel ID #{tracker_id} crossed RIGHT to LEFT at frame {frame_number}")
    
    # Keep only recent crossings (last 90 frames = ~3 seconds)
    recent_crossings[:] = [c for c in recent_crossings if frame_number - c['frame'] < 90]

def add_visual_indicators(frame, detections, frame_number):
    """
    Add visual indicators to the frame showing tracking status and recent crossings.
    """
    height, width = frame.shape[:2]
    
    # Draw status information
    status_y = 30
    cv2.putText(frame, f"Frame: {frame_number}", (10, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw detection line position indicator
    height, width = frame.shape[:2]
    center_x = width // 2
    cv2.putText(frame, f"Line: x={center_x}", (width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    status_y += 30
    cv2.putText(frame, f"Detections: {len(detections)}", (10, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    status_y += 30
    cv2.putText(frame, f"Total Crossings: {len(crossed_squirrels)}", (10, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show active trackers with more detail
    if detections.tracker_id is not None:
        active_trackers = [tid for tid in detections.tracker_id if tid is not None]
        status_y += 30
        cv2.putText(frame, f"Active Trackers: {len(active_trackers)}", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show tracker IDs
        if len(active_trackers) > 0:
            status_y += 30
            tracker_ids_str = ", ".join([str(tid) for tid in active_trackers[:5]])  # Show first 5
            if len(active_trackers) > 5:
                tracker_ids_str += "..."
            cv2.putText(frame, f"IDs: {tracker_ids_str}", (10, status_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Draw recent crossings alert
    if recent_crossings:
        alert_y = height - 150
        cv2.rectangle(frame, (width - 300, alert_y - 40), (width - 10, height - 10), (0, 0, 255), -1)
        cv2.putText(frame, "RECENT CROSSINGS:", (width - 290, alert_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, crossing in enumerate(recent_crossings[-4:]):  # Show last 4 crossings
            alert_y += 25
            direction_symbol = "🐿️→" if crossing['direction'] == 'LEFT_TO_RIGHT' else "←🐿️"
            frames_ago = frame_number - crossing['frame']
            cv2.putText(frame, f"ID {crossing['tracker_id']} ({frames_ago}f ago)", 
                        (width - 290, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Highlight detections that have crossed before
    if detections.tracker_id is not None:
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id in crossed_squirrels:
                # Draw a special border around squirrels that have already crossed
                if i < len(detections.xyxy):
                    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 3)
                    cv2.putText(frame, "CROSSED!", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def main():
    print("🚀 Starting GPU-optimized squirrel tracking script...")
    print(f"🎯 TensorRT Engine: {MODEL_ENGINE_PATH}")
    print(f"📹 Video path: {SOURCE_VIDEO_PATH}")
    print(f"💾 Output path: {OUTPUT_VIDEO_PATH}")
    
    # Load the TensorRT engine (GPU optimized)
    print("🧠 Loading TensorRT engine...")
    start_time = time.time()
    
    print("🚀 Loading TensorRT engine...")
    model = YOLO(MODEL_ENGINE_PATH)
    print(f"✅ TensorRT engine loaded successfully in {time.time() - start_time:.2f} seconds")

    # Initialize the tracker with optimized settings for better ID consistency
    print("🎯 Initializing tracker with optimized settings...")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.15,    # Lower threshold for tracking
        lost_track_buffer=60,               # Longer buffer to maintain IDs
        minimum_matching_threshold=0.7,     # Lower match threshold for better association
        frame_rate=30                       # Match your video frame rate
    )

    # Initialize the line zone counter (will be set up after getting video dimensions)
    line_zone = None

    # Initialize the annotators for drawing boxes, traces, and the line
    print("🎨 Initializing annotators...")
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=100)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # Get video information (frame size, fps) to set up the video writer
    print("📊 Getting video information...")
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"📺 Video info: {video_info.width}x{video_info.height} @ {video_info.fps}fps, {video_info.total_frames} frames")
    print("🎬 Processing entire video")
    
    # Set up the detection line at the center of the frame
    center_x = video_info.width // 2
    line_start = sv.Point(center_x, 0)
    line_end = sv.Point(center_x, video_info.height)
    print(f"📏 Detection line positioned at center: x={center_x} (left-to-right crossing detection)")
    
    # Initialize the line zone with the calculated positions
    line_zone = sv.LineZone(start=line_start, end=line_end)
    
    frame_number = 0
    last_progress_time = time.time()
    processing_start_time = time.time()
    
    # Set up a video writer to save the output
    print("🎬 Starting video processing...")
    with sv.VideoSink(OUTPUT_VIDEO_PATH, video_info) as sink:
        # Loop through each frame of the source video
        for frame in sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH):
            frame_number += 1
            
            # Print progress every 60 frames (roughly every 2 seconds at 30fps)
            current_time = time.time()
            if frame_number % 60 == 0 or current_time - last_progress_time > 10:
                progress_percent = (frame_number / video_info.total_frames) * 100
                elapsed_time = current_time - processing_start_time
                fps = frame_number / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = (video_info.total_frames - frame_number) / fps if fps > 0 else 0
                eta_minutes = eta_seconds / 60
                print(f"⏳ Frame {frame_number}/{video_info.total_frames} ({progress_percent:.1f}%) - {fps:.1f} FPS - ETA: {eta_minutes:.1f}min - {len(crossed_squirrels)} crossings")
                last_progress_time = current_time
            
            # Run TensorRT inference
            results = model(frame, verbose=False)[0]
            
            # Convert YOLO results to supervision Detections object
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter detections with lower confidence threshold for better recall
            detections = detections[detections.confidence > 0.2]
            
            # Log detection info for debugging every 120 frames (reduce logging overhead)
            if frame_number % 120 == 0 and len(detections) > 0:
                print(f"🔍 Frame {frame_number}: Found {len(detections)} detections with confidence > 0.2")
            
            # Update detections with tracking information
            detections = tracker.update_with_detections(detections)
            
            # Trigger the line zone to check for crossings
            crossed_in, crossed_out = line_zone.trigger(detections)
            
            # Enhanced debugging for line crossing detection
            if len(detections) > 0:
                # Log tracker IDs and positions every 60 frames when there are detections
                if frame_number % 60 == 0:
                    tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
                    for i, bbox in enumerate(detections.xyxy):
                        if i < len(tracker_ids) and tracker_ids[i] is not None:
                            tracker_id = tracker_ids[i]
                            x1, y1, x2, y2 = bbox
                            center_x_det = (x1 + x2) / 2
                            print(f"🎯 Tracker ID {tracker_id}: center_x={center_x_det:.1f}, line_x={center_x}")
            
            # Log crossing activity for debugging
            if np.any(crossed_in) or np.any(crossed_out):
                print(f"🚨 Frame {frame_number}: Line crossing activity detected! In: {np.sum(crossed_in)}, Out: {np.sum(crossed_out)}")
                # Log which specific detections crossed
                if detections.tracker_id is not None:
                    for i, (crossed_in_det, crossed_out_det, tracker_id) in enumerate(zip(crossed_in, crossed_out, detections.tracker_id)):
                        if crossed_in_det:
                            print(f"   ↪️  Tracker ID {tracker_id} crossed IN (left to right)")
                        if crossed_out_det:
                            print(f"   ↩️  Tracker ID {tracker_id} crossed OUT (right to left)")
            
            # Process any line crossings
            process_line_crossings(detections, crossed_in, crossed_out, frame_number)

            # Annotate the frame with detection boxes, tracker IDs, and traces
            # Get class names safely
            try:
                class_names = model.names if hasattr(model, 'names') else model.model.names
            except:
                class_names = {0: 'object'}  # fallback
            
            labels = [
                f"#{tracker_id} {class_names.get(class_id, 'unknown')} {confidence:0.2f}"
                for _, _, confidence, class_id, tracker_id, _
                in detections
            ]
            annotated_frame = frame.copy()
            
            # Always add annotations to show bounding boxes and confidence scores
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            # Annotate the frame with the line and the crossing counts
            line_zone_annotator.annotate(frame=annotated_frame, line_counter=line_zone)
            
            # Add custom visual indicators every frame for better visibility
            annotated_frame = add_visual_indicators(annotated_frame, detections, frame_number)

            # Write the annotated frame to the output video file
            sink.write_frame(annotated_frame)

    total_time = time.time() - processing_start_time
    print(f"\n🎉 Processing complete!")
    print(f"📊 Total processing time: {total_time:.2f} seconds")
    print(f"📊 Average FPS: {frame_number / total_time:.2f}")
    print(f"🐿️  Total unique squirrels crossed left-to-right: {len(crossed_squirrels)}")
    if crossed_squirrels:
        print(f"🆔 Crossed squirrel IDs: {list(crossed_squirrels.keys())}")
    print(f"💾 Output video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"🎬 Video features:")
    print(f"   • Confidence threshold: 0.2 (improved detection)")
    print(f"   • Detection line at center: x={center_x}")
    print(f"   • Optimized tracker settings for better ID consistency")
    print(f"   • Enhanced debugging output for line crossings")
    print(f"   • Bounding boxes with confidence scores")
    print(f"   • Frame numbers and detection counts")
    print(f"   • Active tracker information with IDs")
    print(f"   • Real-time crossing alerts")
    print(f"   • Green highlighting for squirrels that have crossed")
    print(f"   • Detection traces and line zone visualization")
    print(f"   • MP4 format for smaller file size")

if __name__ == "__main__":
    main()
