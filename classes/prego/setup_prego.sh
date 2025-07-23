#!/bin/bash

# PREGO Setup Script
# This script sets up the PREGO model for online mistake detection in procedural egocentric videos

echo "=== PREGO Setup Script ==="
echo "Setting up environment for PREGO: online mistake detection in PRocedural EGOcentric videos"

# 1. Clone the PREGO repository
echo "Step 1: Cloning PREGO repository..."
if [ ! -d "PREGO" ]; then
    git clone https://github.com/aleflabo/PREGO.git
    cd PREGO
else
    echo "PREGO repository already exists"
    cd PREGO
fi

# 2. Install requirements
echo "Step 2: Installing Python requirements..."
pip install -r ../requirements.txt

# 3. Install Unsloth for LLAMA anticipation
echo "Step 3: Installing Unsloth..."
# Check CUDA version and install appropriate Unsloth
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

if [ "$CUDA_MAJOR" -eq 12 ]; then
    echo "Installing Unsloth for CUDA 12.x"
    pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    echo "Installing Unsloth for CUDA 11.8"
    pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
else
    echo "WARNING: Your CUDA version may not be fully supported. Installing default Unsloth..."
    pip install unsloth
fi

# 4. Download the TSN features for Assembly101-O and Epic-tent-O datasets
echo "Step 4: Downloading dataset features..."
mkdir -p data
cd data

if [ ! -d "Assembly101-O" ] && [ ! -d "Epic-tent-O" ]; then
    echo "Downloading TSN features from Google Drive..."
    gdown --folder --remaining-ok https://drive.google.com/drive/u/1/folders/1gcOIEXhwysCE2o8-5C4vQnTShJ7p3CKH
else
    echo "Dataset features already downloaded"
fi

cd ..

# 5. Create necessary directories
echo "Step 5: Creating necessary directories..."
mkdir -p step_recognition/checkpoint/miniROAD/Assembly101-O
mkdir -p step_recognition/checkpoint/miniROAD/Epic-tent-O
mkdir -p output_miniROAD
mkdir -p step_anticipation/data/predictions
mkdir -p step_anticipation/data/context_prompt
mkdir -p results

# 6. Download LLAMA model (you need to be granted access by Meta)
echo "Step 6: LLAMA Model Setup"
echo "IMPORTANT: You need to download LLAMA-2-7B model from Meta."
echo "1. Request access at: https://www.llama.com/llama-downloads/"
echo "2. Once granted, download llama-2-7b model"
echo "3. Place the model in a directory and update the path in step_anticipation/scripts/anticipation.sh"

# 7. Create config files if they don't exist
echo "Step 7: Creating default config files..."

# Create miniroad_assembly101-O.yaml if it doesn't exist
if [ ! -f "step_recognition/configs/miniroad_assembly101-O.yaml" ]; then
    mkdir -p step_recognition/configs
    cat > step_recognition/configs/miniroad_assembly101-O.yaml << EOF
# MiniROAD config for Assembly101-O
dataset: Assembly101-O
data_path: ../data/Assembly101-O
model:
  name: miniroad
  hidden_dim: 512
  num_layers: 2
  dropout: 0.5
  num_classes: 203  # Assembly101 has 203 action classes
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  checkpoint_interval: 5
evaluation:
  batch_size: 64
  output_path: ../output_miniROAD
EOF
fi

# Create miniroad_epic-tent-O.yaml if it doesn't exist
if [ ! -f "step_recognition/configs/miniroad_epic-tent-O.yaml" ]; then
    cat > step_recognition/configs/miniroad_epic-tent-O.yaml << EOF
# MiniROAD config for Epic-tent-O
dataset: Epic-tent-O
data_path: ../data/Epic-tent-O
model:
  name: miniroad
  hidden_dim: 512
  num_layers: 2
  dropout: 0.5
  num_classes: 97  # Epic-tent has 97 action classes
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  checkpoint_interval: 5
evaluation:
  batch_size: 64
  output_path: ../output_miniROAD
EOF
fi

echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Download LLAMA-2-7B model and place it in the appropriate directory"
echo "2. Update paths in step_anticipation/scripts/anticipation.sh"
echo "3. Train or download pre-trained MiniROAD checkpoints"
echo "4. Run inference using the provided inference script"