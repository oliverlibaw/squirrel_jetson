#!/bin/bash

# Jetson Squirrel Tracker Installation Script
# This script automates the installation process for the Jetson platform

set -e  # Exit on any error

echo "=========================================="
echo "Jetson Squirrel Tracker - Installation"
echo "=========================================="

# Check if running on Jetson
if [[ ! -f "/proc/device-tree/model" ]] || ! grep -qi "jetson" /proc/device-tree/model; then
    echo "⚠️  Warning: This script is designed for NVIDIA Jetson devices."
    echo "   It may work on other platforms but is not guaranteed."
    echo ""
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ is required. Found: $python_version"
    exit 1
else
    echo "✅ Python version: $python_version"
fi

# Update package list
echo ""
echo "Updating package list..."
sudo apt-get update

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-venv \
    python3-pip \
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

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ -d "squirrel_env" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf squirrel_env
fi

python3 -m venv squirrel_env
echo "✅ Virtual environment created: squirrel_env"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source squirrel_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for Jetson (if available)
echo ""
echo "Installing PyTorch for Jetson..."
if pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir; then
    echo "✅ PyTorch installed successfully"
else
    echo "⚠️  Failed to install PyTorch from NVIDIA wheels. Trying standard installation..."
    pip install torch torchvision
fi

# Install other requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Test installation
echo ""
echo "Testing installation..."
python test_setup.py

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To use the Squirrel Tracker:"
echo "1. Activate the virtual environment:"
echo "   source squirrel_env/bin/activate"
echo ""
echo "2. Run the test script to verify everything works:"
echo "   python test_setup.py"
echo ""
echo "3. Run the main application:"
echo "   python squirrel_tracker.py --model your_model.pt"
echo ""
echo "Note: Make sure you have a YOLOv11s model file (.pt format) ready."
echo ""
echo "For help, see README.md or run:"
echo "   python squirrel_tracker.py --help"
echo ""
