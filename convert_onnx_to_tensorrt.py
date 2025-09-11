#!/usr/bin/env python3
"""
ONNX to TensorRT Engine Converter for NVIDIA Jetson

This script converts ONNX models to optimized TensorRT engine files
specifically optimized for Jetson devices with TensorRT support.

Author: AI Assistant
Target Platform: NVIDIA Jetson Orin Nano
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

try:
    import numpy as np
    import onnx
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Please install required packages:")
    print("pip install onnx numpy")
    sys.exit(1)


class ONNXToTensorRTConverter:
    """
    Converts ONNX models to optimized TensorRT engines for Jetson using trtexec.
    """
    
    def __init__(self, onnx_path: str, engine_path: str = None):
        """
        Initialize the converter.
        
        Args:
            onnx_path: Path to the input ONNX model file
            engine_path: Path for the output TensorRT engine file
        """
        self.onnx_path = onnx_path
        
        if engine_path is None:
            # Generate engine path from ONNX path
            self.engine_path = onnx_path.replace('.onnx', '.engine')
        else:
            self.engine_path = engine_path
            
        # Check for trtexec
        self.trtexec_path = self._find_trtexec()
        
    def _find_trtexec(self):
        """Find the trtexec binary on the system."""
        # Common locations for trtexec on Jetson
        possible_paths = [
            "/usr/src/tensorrt/bin/trtexec",
            "/usr/local/tensorrt/bin/trtexec",
            "/opt/tensorrt/bin/trtexec",
            "/usr/bin/trtexec"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found trtexec at: {path}")
                return path
        
        # Try to find it in PATH
        try:
            result = subprocess.run(['which', 'trtexec'], capture_output=True, text=True)
            if result.returncode == 0:
                trtexec_path = result.stdout.strip()
                print(f"Found trtexec at: {trtexec_path}")
                return trtexec_path
        except:
            pass
        
        print("Error: trtexec not found. Please ensure TensorRT is installed with JetPack SDK.")
        print("Common locations to check:")
        for path in possible_paths:
            print(f"  {path}")
        sys.exit(1)
    
    def _validate_onnx(self):
        """Validate the ONNX model file."""
        try:
            print(f"Validating ONNX model: {self.onnx_path}")
            
            # Load and validate ONNX model
            onnx_model = onnx.load(self.onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Get model info
            input_shape = None
            output_shape = None
            
            for input_info in onnx_model.graph.input:
                if input_info.name == "images":  # YOLO input name
                    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                    break
            
            for output_info in onnx_model.graph.output:
                if output_info.name == "output0":  # YOLO output name
                    output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
                    break
            
            print(f"ONNX model validation successful!")
            if input_shape:
                print(f"Input shape: {input_shape}")
            if output_shape:
                print(f"Output shape: {output_shape}")
                
        except Exception as e:
            print(f"Error validating ONNX model: {e}")
            sys.exit(1)
    
    def _build_engine_with_trtexec(self):
        """Build TensorRT engine using trtexec command-line tool."""
        try:
            print("Building TensorRT engine using trtexec...")
            print("This may take several minutes...")
            
            start_time = time.time()
            
            # Prepare trtexec command
            cmd = [
                self.trtexec_path,
                '--onnx=' + self.onnx_path,
                '--saveEngine=' + self.engine_path,
                '--memPoolSize=workspace:2048',  # 2GB workspace
                '--fp16',  # Enable FP16 for Jetson optimization
                '--verbose'  # Show detailed output
            ]
            
            print("Running command:")
            print(' '.join(cmd))
            print()
            
            # Run trtexec
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if process.returncode != 0:
                print("Error: trtexec failed with return code:", process.returncode)
                print("stdout:", process.stdout)
                print("stderr:", process.stderr)
                sys.exit(1)
            
            build_time = time.time() - start_time
            print(f"TensorRT engine built successfully in {build_time:.2f} seconds!")
            print(f"Engine saved to: {self.engine_path}")
            
            # Get engine info
            if os.path.exists(self.engine_path):
                engine_size = os.path.getsize(self.engine_path) / (1024 * 1024)  # MB
                print(f"Engine file size: {engine_size:.2f} MB")
            else:
                print("Warning: Engine file not found after build")
            
        except subprocess.TimeoutExpired:
            print("Error: Build timed out after 30 minutes")
            sys.exit(1)
        except Exception as e:
            print(f"Error building TensorRT engine: {e}")
            sys.exit(1)
    
    def convert(self):
        """Main conversion process."""
        print("=" * 60)
        print("ONNX to TensorRT Engine Converter for Jetson")
        print("=" * 60)
        
        # Check if ONNX file exists
        if not os.path.exists(self.onnx_path):
            print(f"Error: ONNX file not found: {self.onnx_path}")
            sys.exit(1)
        
        # Check if output directory exists
        output_dir = os.path.dirname(self.engine_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Validate ONNX model
        self._validate_onnx()
        
        # Build engine using trtexec
        self._build_engine_with_trtexec()
        
        print("=" * 60)
        print("Conversion completed successfully!")
        print(f"ONNX model: {self.onnx_path}")
        print(f"TensorRT engine: {self.engine_path}")
        print("=" * 60)


def main():
    """Main entry point for the converter."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to optimized TensorRT engines for Jetson"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to the input ONNX model file"
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="Path for the output TensorRT engine file (optional, auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Create converter and run conversion
    converter = ONNXToTensorRTConverter(
        onnx_path=args.onnx,
        engine_path=args.engine
    )
    
    converter.convert()


if __name__ == "__main__":
    main()
