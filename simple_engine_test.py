#!/usr/bin/env python3
"""
Simple TensorRT Engine Test Script
Based on your working script - tests .engine model directly with TensorRT
"""

import cv2
import numpy as np
import sys
import os

# Add system TensorRT to path
sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    print("✅ TensorRT and PyCUDA imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: sudo apt-get install python3-pycuda")
    sys.exit(1)

# Configuration
ENGINE_PATH = "squirrel_detector_yolov11n_5-2025.engine"
INPUT_H = 640
INPUT_W = 640
CONF_THRESHOLD = 0.25
CLASSES = ["squirrel"]  # Adjust with your actual class names

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

class TrtEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Use the working binding approach from your script
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.get_binding_dtype(binding).itemsize
            host_mem = cuda.pagelocked_empty(size // self.engine.get_binding_dtype(binding).itemsize, dtype=trt.nptype(self.engine.get_binding_dtype(binding)))
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def infer(self, host_input):
        # Transfer input data to the GPU
        np.copyto(self.inputs[0].host, host_input.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Run inference
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)

        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()
        return self.outputs[0].host

def preprocess_image(frame, input_w, input_h):
    """Preprocess image for YOLO inference"""
    img = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_output(output, original_frame_shape, input_w, input_h, conf_threshold, classes):
    """Postprocess YOLO output to get detections"""
    boxes, scores, class_ids = [], [], []
    
    # Reshape output based on typical YOLO format
    # Try to detect the format automatically
    if len(output.shape) == 1:
        # Flat output - reshape to (1, features, detections)
        total_features = 5 + len(classes)  # x,y,w,h,conf + classes
        if output.size % total_features == 0:
            num_detections = output.size // total_features
            output = output.reshape(1, total_features, num_detections)
            output = output.transpose((0, 2, 1))  # -> (1, detections, features)
    
    # Process detections
    if len(output.shape) >= 2:
        detections = output[0] if len(output.shape) == 3 else output
        
        for det in detections:
            if len(det) < 5:
                continue
                
            confidence = det[4]  # YOLO objectness confidence
            if confidence >= conf_threshold:
                class_scores = det[5:] if len(det) > 5 else [1.0]
                class_id = np.argmax(class_scores)
                score = class_scores[class_id] * confidence
                
                if score >= conf_threshold:
                    x_center, y_center, box_w, box_h = det[0:4]
                    
                    # Convert to pixel coordinates
                    x1 = int((x_center - box_w / 2) * original_frame_shape[1] / input_w)
                    y1 = int((y_center - box_h / 2) * original_frame_shape[0] / input_h)
                    x2 = int((x_center + box_w / 2) * original_frame_shape[1] / input_w)
                    y2 = int((y_center + box_h / 2) * original_frame_shape[0] / input_h)
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, original_frame_shape[1]))
                    y1 = max(0, min(y1, original_frame_shape[0]))
                    x2 = max(0, min(x2, original_frame_shape[1]))
                    y2 = max(0, min(y2, original_frame_shape[0]))
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(score))
                    class_ids.append(class_id)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                "box": boxes[i],
                "score": scores[i],
                "class_id": class_ids[i],
                "class_name": classes[class_ids[i]] if class_ids[i] < len(classes) else f"class_{class_ids[i]}"
            })
    
    return detections

def test_engine():
    """Test TensorRT engine with a sample frame"""
    print(f"Testing TensorRT engine: {ENGINE_PATH}")
    
    if not os.path.exists(ENGINE_PATH):
        print(f"❌ Engine file not found: {ENGINE_PATH}")
        return False
    
    try:
        # Load engine
        engine = TrtEngine(ENGINE_PATH)
        print("✅ TensorRT engine loaded successfully!")
        
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"🖼️  Created test frame: {test_frame.shape}")
        
        # Preprocess
        input_data = preprocess_image(test_frame, INPUT_W, INPUT_H)
        print(f"🔧 Preprocessed shape: {input_data.shape}")
        
        # Run inference
        output = engine.infer(input_data)
        print(f"🎯 Raw output shape: {output.shape}")
        print(f"🎯 Raw output size: {output.size}")
        
        # Postprocess
        detections = postprocess_output(output, test_frame.shape, INPUT_W, INPUT_H, CONF_THRESHOLD, CLASSES)
        print(f"🎉 Detections found: {len(detections)}")
        
        if detections:
            for i, det in enumerate(detections):
                print(f"   Detection {i+1}: {det['class_name']} ({det['score']:.3f}) at {det['box']}")
        else:
            print("   No detections found (expected for random image)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Simple TensorRT Engine Test")
    print("=" * 50)
    
    success = test_engine()
    
    if success:
        print("\n✅ TensorRT engine test completed successfully!")
        print("Your .engine model is working properly with direct TensorRT inference.")
    else:
        print("\n❌ TensorRT engine test failed!")
        print("Check the error messages above for troubleshooting.")
