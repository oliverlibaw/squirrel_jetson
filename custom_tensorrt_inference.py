#!/usr/bin/env python3
"""
Custom TensorRT Inference Class for Jetson
This class provides direct TensorRT engine inference without relying on Ultralytics
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional
import subprocess
import json

class HostDeviceMem(object):
    """Helper class for TensorRT memory management (from working script)"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class CustomTensorRTInference:
    def __init__(self, engine_path: str, confidence_threshold: float = 0.4):
        """
        Initialize TensorRT inference with a pre-built engine file
        
        Args:
            engine_path: Path to the TensorRT engine file
            confidence_threshold: Minimum confidence for detections
        """
        self.engine_path = engine_path
        self.confidence_threshold = confidence_threshold
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        print(f"✅ Loading TensorRT engine: {engine_path}")
        print(f"📁 Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
        
        # Initialize model parameters (these should match your training configuration)
        self.input_shape = (640, 640)  # YOLOv11 input size
        self.class_names = ['squirrel']  # Your class names
        
        # Try to load class names from data.yaml if it exists
        self._load_class_names()
        
        print(f"🎯 Model configured for {len(self.class_names)} classes: {self.class_names}")
    
    def _load_class_names(self):
        """Try to load class names from data.yaml file"""
        data_yaml_path = "data.yaml"
        if os.path.exists(data_yaml_path):
            try:
                import yaml
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        self.class_names = data['names']
                        print(f"📋 Loaded {len(self.class_names)} classes from data.yaml")
            except Exception as e:
                print(f"⚠️  Could not load data.yaml: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO inference
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Resize image to model input size
        resized = cv2.resize(image, self.input_shape)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W) format and add batch dimension
        # YOLO expects (batch, channels, height, width)
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess_detections(self, raw_output: np.ndarray, original_shape: Tuple[int, int]) -> List[dict]:
        """
        Postprocess raw model output using the approach from working script
        
        Args:
            raw_output: Raw model output from TensorRT
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Reshape output based on typical YOLO format
        # Assuming output is like (1, 5+num_classes, num_detections) -> need to transpose
        try:
            # Try to reshape output to expected format
            num_classes = len(self.class_names)
            expected_features = 5 + num_classes  # x,y,w,h,conf + classes
            
            if raw_output.size % expected_features == 0:
                num_detections = raw_output.size // expected_features
                output = raw_output.reshape(1, expected_features, num_detections)
                
                # Transpose to (1, num_detections, expected_features) like working script
                if output.shape[1] == expected_features:
                    output = output.transpose((0, 2, 1))
            else:
                # Fallback: try to detect format automatically
                output = raw_output.reshape(1, -1, raw_output.shape[-1] if len(raw_output.shape) > 1 else raw_output.size)
            
            # Process detections (based on working script logic)
            for det in output[0]:  # Assuming batch size 1
                if len(det) < 5:
                    continue
                    
                confidence = det[4]  # YOLO confidence for objectness
                if confidence >= self.confidence_threshold:
                    class_scores = det[5:] if len(det) > 5 else [1.0]  # Handle single class
                    class_id = np.argmax(class_scores)
                    score = class_scores[class_id] * confidence  # Combined score
                    
                    if score >= self.confidence_threshold:
                        x_center, y_center, box_w, box_h = det[0:4]
                        
                        # Convert to pixel coordinates (from working script)
                        img_h, img_w = original_shape
                        x1 = int((x_center - box_w / 2) * img_w / self.input_shape[0])
                        y1 = int((y_center - box_h / 2) * img_h / self.input_shape[1])
                        x2 = int((x_center + box_w / 2) * img_w / self.input_shape[0])
                        y2 = int((y_center + box_h / 2) * img_h / self.input_shape[1])
                        
                        # Ensure coordinates are within bounds
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        x2 = max(0, min(x2, img_w))
                        y2 = max(0, min(y2, img_h))
                        
                        detection_dict = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(score),
                            'class_id': int(class_id),
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                        }
                        
                        detections.append(detection_dict)
        
        except Exception as e:
            print(f"⚠️  Postprocessing error: {e}")
            print(f"   Raw output shape: {raw_output.shape}")
            print(f"   Raw output size: {raw_output.size}")
        
        return detections
    
    def infer(self, image: np.ndarray) -> List[dict]:
        """
        Run inference on an image using direct TensorRT API
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Import TensorRT and CUDA
            import sys
            sys.path.insert(0, '/usr/lib/python3.10/dist-packages')
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Initialize TensorRT runtime if not already done
            if not hasattr(self, 'engine'):
                self._initialize_engine()
            
            # Preprocess image
            input_data = self.preprocess_image(image)
            
            # Use the working inference approach
            np.copyto(self.inputs[0].host, input_data.ravel())
            cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
            
            # Run inference
            self.context.execute_async_v2(self.bindings, self.stream.handle, None)
            
            # Transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
            self.stream.synchronize()
            
            # Postprocess results
            detections = self.postprocess_detections(self.outputs[0].host, image.shape[:2])
            
            return detections
            
        except ImportError as e:
            print(f"❌ TensorRT import error: {e}")
            print("   Using Ultralytics CLI fallback method...")
            return self._infer_with_ultralytics_cli(image)
        except Exception as e:
            print(f"❌ TensorRT inference error: {e}")
            return []
    
    def _initialize_engine(self):
        """Initialize TensorRT engine using working approach from previous script"""
        import sys
        sys.path.insert(0, '/usr/lib/python3.10/dist-packages')
        import tensorrt as trt
        import pycuda.driver as cuda
        
        # Create TensorRT logger and runtime (same as working script)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Use the working approach for memory allocation
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.get_binding_dtype(binding).itemsize
            host_mem = cuda.pagelocked_empty(size // self.engine.get_binding_dtype(binding).itemsize, dtype=trt.nptype(self.engine.get_binding_dtype(binding)))
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            
            # Create HostDeviceMem objects
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
        
        print("✅ TensorRT engine initialized successfully using working approach!")
    
    def _infer_with_ultralytics_cli(self, image: np.ndarray) -> List[dict]:
        """Use Ultralytics CLI for inference - bypasses PyTorch CUDA issues"""
        import tempfile
        import json
        
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                cv2.imwrite(temp_img.name, image)
                temp_img_path = temp_img.name
            
            # Create temp output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run Ultralytics CLI prediction
                cmd = [
                    'python3', '-m', 'ultralytics.yolo.engine.predictor',
                    'predict',
                    f'model={self.engine_path}',
                    f'source={temp_img_path}',
                    f'conf={self.confidence_threshold}',
                    f'project={temp_dir}',
                    'name=pred',
                    'save_txt=True',
                    'save_conf=True'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/andrew/Projects/squirrel_jetson')
                
                if result.returncode != 0:
                    print(f"⚠️  CLI inference failed: {result.stderr}")
                    return []
                
                # Parse results from saved txt files
                results_dir = os.path.join(temp_dir, 'pred', 'labels')
                detections = []
                
                if os.path.exists(results_dir):
                    for txt_file in os.listdir(results_dir):
                        if txt_file.endswith('.txt'):
                            txt_path = os.path.join(results_dir, txt_file)
                            with open(txt_path, 'r') as f:
                                lines = f.readlines()
                            
                            img_h, img_w = image.shape[:2]
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 6:
                                    class_id, x_center, y_center, width, height, confidence = map(float, parts[:6])
                                    
                                    # Convert normalized YOLO format to pixel coordinates
                                    x1 = int((x_center - width/2) * img_w)
                                    y1 = int((y_center - height/2) * img_h)
                                    x2 = int((x_center + width/2) * img_w)
                                    y2 = int((y_center + height/2) * img_h)
                                    
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'confidence': confidence,
                                        'class_id': int(class_id),
                                        'class_name': self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f'class_{int(class_id)}'
                                    })
                
                # Clean up temp image
                os.unlink(temp_img_path)
                return detections
                
        except Exception as e:
            print(f"⚠️  CLI inference error: {e}")
            return []
    
    def __call__(self, image: np.ndarray) -> List[dict]:
        """Convenience method for inference"""
        return self.infer(image)


def test_custom_inference():
    """Test the custom inference class"""
    print("🧪 Testing Custom TensorRT Inference Class")
    
    # Create instance
    engine_path = "squirrel_detector_yolov11n_5-2025.engine"
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine file not found: {engine_path}")
        return
    
    try:
        # Initialize inference
        inference = CustomTensorRTInference(engine_path)
        print("✅ Custom inference class initialized successfully")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"🖼️  Created test image: {test_image.shape}")
        
        # Test preprocessing
        preprocessed = inference.preprocess_image(test_image)
        print(f"🔧 Preprocessed image shape: {preprocessed.shape}")
        
        # Test inference (placeholder)
        detections = inference.infer(test_image)
        print(f"🎯 Detections: {len(detections)}")
        
        print("\n🎉 Custom inference class test completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_custom_inference()
