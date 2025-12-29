#!/bin/bash

echo "=== Installing PyTorch with CUDA 11.8 support ==="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Uninstall existing PyTorch
echo "Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other requirements..."
pip install transformers>=4.50.0 accelerate pandas tqdm python-dotenv

# Test CUDA availability
echo ""
echo "Testing CUDA availability..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

echo ""
echo "=== Installation complete ==="
echo "If CUDA is still not available, you may need to:"
echo "1. Update your NVIDIA driver (current: 470.199.02)"
echo "2. Or use an older PyTorch version compatible with CUDA 11.4"