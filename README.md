# Real-Time Squirrel Tracker & Counter for NVIDIA Jetson Orin Nano

A standalone Python application that uses a live webcam feed to detect, track, and count squirrels as they move from the right side of the frame to the left using YOLOv11s and the Supervision library with TensorRT optimization.

## Features

- **Real-time Detection**: Uses YOLOv11s model for accurate squirrel detection
- **Object Tracking**: Implements ByteTrack for persistent squirrel identification
- **Zone-based Counting**: Counts squirrels crossing from right to left zones
- **TensorRT Optimization**: Automatically exports and uses TensorRT engines for Jetson performance
- **GStreamer Pipeline**: Optimized video capture for NVIDIA Jetson platform
- **Real-time Visualization**: Live video feed with bounding boxes, tracker IDs, and zone overlays

## Hardware Requirements

- NVIDIA Jetson Orin Nano Developer Kit (8GB recommended)
- Standard USB Webcam (capable of at least 1280x720 resolution at 30 FPS)
- microSD card or NVMe SSD with JetPack installed

## Software Requirements

- Jetson Linux (via JetPack 6.x)
- Python 3.10+
- CUDA-enabled PyTorch and OpenCV

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Jetson_Nano_Squirrel_1
```

### 2. Create Virtual Environment

```bash
python3 -m venv squirrel_env
source squirrel_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For Jetson platforms, you may need to install PyTorch and OpenCV from NVIDIA's pre-built wheels:

```bash
# Install PyTorch for Jetson
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install OpenCV with CUDA support
pip install opencv-python-headless
```

### 4. Install GStreamer Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio
```

## Usage

### Basic Usage

```bash
python squirrel_tracker.py --model path/to/your/squirrel_model.pt
```

### Advanced Usage

```bash
python squirrel_tracker.py \
    --model path/to/your/squirrel_model.pt \
    --webcam 1 \
    --confidence 0.5
```

### Command Line Arguments

- `--model`: **Required**. Path to your YOLOv11s model file (.pt or .engine)
- `--webcam`: Webcam device ID (default: 0)
- `--confidence`: Detection confidence threshold (default: 0.4)

### Model Requirements

Your YOLOv11s model should be trained to detect squirrels. The application expects:

- Model format: `.pt` (PyTorch) or `.engine` (TensorRT)
- Class 0 should represent squirrels (adjust in code if different)
- Model should be compatible with YOLOv11s architecture

## How It Works

### 1. Model Initialization

- Loads your YOLOv11s model
- Automatically exports to TensorRT format on first run for Jetson optimization
- Subsequent runs use the pre-exported TensorRT engine

### 2. Video Capture

- Uses GStreamer pipeline optimized for Jetson
- Falls back to standard OpenCV capture if GStreamer fails
- Default resolution: 1280x720

### 3. Detection & Tracking

- Processes each frame through YOLOv11s model
- Filters detections by confidence threshold
- Assigns persistent tracker IDs using ByteTrack

### 4. Zone Management

- **Right Zone**: Right 50% of frame (x = width/2 to width)
- **Left Zone**: Left 50% of frame (x = 0 to width/2)
- Zones cover entire frame height

### 5. Counting Logic

- Tracks each squirrel's zone state
- Increments counter only when a squirrel moves from right zone to left zone
- Prevents double-counting unless squirrel returns to right zone first

### 6. Visualization

- Real-time video display with annotations
- Zone overlays (blue for right, green for left)
- Bounding boxes with tracker IDs
- Live count display

## Performance Optimization

### TensorRT Export

The application automatically exports your PyTorch model to TensorRT format for optimal Jetson performance. This happens on the first run and can take several minutes.

### GStreamer Pipeline

The optimized GStreamer pipeline reduces latency and improves frame capture performance on Jetson devices.

### Expected Performance

- **PyTorch Model**: 5-10 FPS
- **TensorRT Model**: 15-20+ FPS

## Troubleshooting

### Common Issues

#### 1. Webcam Not Found

```bash
# Check available video devices
ls /dev/video*

# Try different webcam ID
python squirrel_tracker.py --model model.pt --webcam 1
```

#### 2. GStreamer Pipeline Failure

The application automatically falls back to standard OpenCV capture if GStreamer fails. Check GStreamer installation:

```bash
gst-launch-1.0 --version
```

#### 3. Model Loading Errors

- Ensure your model file exists and is accessible
- Check model compatibility with YOLOv11s
- Verify file permissions

#### 4. Low Performance

- Ensure TensorRT engine was created successfully
- Check CUDA availability: `nvidia-smi`
- Monitor GPU memory usage

#### 5. Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# For Jetson-specific packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Debug Mode

To enable verbose output, modify the model inference call in the code:

```python
# Change this line in _process_detections method
results = self.model(frame, verbose=True)[0]  # Set verbose=True
```

## File Structure

```
Jetson_Nano_Squirrel_1/
├── squirrel_tracker.py      # Main application
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .qodo/                  # IDE configuration
```

## Dependencies

- **ultralytics**: YOLOv11s model loading and inference
- **supervision**: Object tracking and visualization
- **torch & torchvision**: PyTorch framework
- **opencv-python**: Computer vision operations
- **numpy**: Numerical operations
- **tensorrt**: NVIDIA TensorRT optimization

## License

This project is provided as-is for educational and research purposes.

## Support

For issues related to:

- **Jetson Platform**: Check NVIDIA Jetson documentation
- **YOLOv11s**: Refer to Ultralytics documentation
- **Supervision**: Check Supervision library documentation
- **Application Logic**: Review the code comments and this README

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.
