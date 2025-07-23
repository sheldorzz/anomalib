# PREGO Quick Start Guide

This guide will help you set up and run PREGO (online mistake detection in PRocedural EGOcentric videos) on your Linux PC.

## Prerequisites

- Linux PC with NVIDIA GPU (CUDA 11.8+ recommended)
- Python 3.10
- ~10GB disk space for data
- (Optional) Access to LLAMA-2-7B model from Meta

## Step 1: Environment Setup

```bash
# Create and activate conda environment
conda create -n prego python=3.10
conda activate prego

# Clone repository and setup
git clone https://github.com/aleflabo/PREGO.git
cd PREGO

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (for CUDA 12.x)
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
# OR for CUDA 11.8
pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
```

## Step 2: Download Data

```bash
# Download TSN features (requires ~10GB)
mkdir -p data && cd data
gdown --folder --remaining-ok https://drive.google.com/drive/u/1/folders/1gcOIEXhwysCE2o8-5C4vQnTShJ7p3CKH
cd ..

# OR create sample data for testing
python download_sample_data.py
```

## Step 3: Train or Download Model

### Option A: Train MiniROAD from scratch
```bash
# Train on Assembly101-O
python step_recognition/main.py --config step_recognition/configs/miniroad_assembly101-O.yaml

# Train on Epic-tent-O
python step_recognition/main.py --config step_recognition/configs/miniroad_epic-tent-O.yaml
```

### Option B: Use pre-trained checkpoint
Download pre-trained checkpoints and place them in:
- `step_recognition/checkpoint/miniROAD/Assembly101-O/`
- `step_recognition/checkpoint/miniROAD/Epic-tent-O/`

## Step 4: Run Inference

### Single Video Inference
```bash
# Basic inference
python run_prego_inference.py \
    --video data/samples/sample_video \
    --checkpoint step_recognition/checkpoint/miniROAD/Assembly101-O/best_model.pth \
    --output results/output.json

# Full pipeline with anticipation
./run_prego.sh \
    --mode inference \
    --dataset Assembly101-O \
    --video sample_video \
    --checkpoint path/to/checkpoint.pth
```

### Batch Inference
```bash
# Process multiple videos
./run_prego.sh \
    --mode batch \
    --dataset Assembly101-O \
    --checkpoint path/to/checkpoint.pth
```

## Step 5: (Optional) Setup LLAMA for Anticipation

1. Request access to LLAMA-2-7B from [Meta](https://www.llama.com/llama-downloads/)
2. Download the model after approval
3. Update paths in `step_anticipation/scripts/anticipation.sh`:
   ```bash
   LLAMA_MODEL_PATH="/path/to/llama-2-7b"
   TOKENIZER_PATH="/path/to/tokenizer.model"
   ```

## Output Format

The inference script produces JSON output with:
- **recognized_actions**: List of detected actions with timestamps
- **detected_mistakes**: List of procedural mistakes found
- **total_frames**: Number of frames processed
- **total_mistakes**: Count of mistakes detected

Example output:
```json
{
  "video": "sample_video",
  "recognized_actions": [
    {"action": 5, "start": 0, "end": 45},
    {"action": 12, "start": 46, "end": 120}
  ],
  "detected_mistakes": [
    {
      "type": "repeated_action",
      "timestamp": 150,
      "action": 12,
      "confidence": 0.8
    }
  ],
  "total_frames": 300,
  "total_mistakes": 1
}
```

## Troubleshooting

### CUDA Issues
- Ensure CUDA toolkit matches PyTorch version
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Reduce batch size in config files
- Use smaller models or features

### Missing Dependencies
- Install missing packages: `pip install <package_name>`
- Check Unsloth compatibility with your CUDA version

## Dataset Structure

PREGO expects data in this structure:
```
data/
├── Assembly101-O/
│   ├── rgb_anet_resnet50/     # RGB features
│   ├── rgb_as_flow/           # Flow features
│   └── target_perframe/       # Ground truth
└── Epic-tent-O/
    ├── rgb_anet_resnet50/
    ├── rgb_as_flow/
    └── target_perframe/
```

## Key Components

1. **Step Recognition (MiniROAD)**: Online action recognition from video features
2. **Aggregation**: Converts frame-level predictions to action segments
3. **Anticipation (LLAMA)**: Predicts expected next actions
4. **Mistake Detection**: Compares recognized vs. anticipated actions

## Performance Tips

- Use GPU for faster inference
- Pre-extract features for multiple videos
- Adjust aggregation window size for better accuracy
- Fine-tune on your specific procedural task

## Citation

If you use PREGO, please cite:
```bibtex
@InProceedings{Flaborea_2024_CVPR,
    author = {Flaborea, Alessandro and others},
    title = {PREGO: Online Mistake Detection in PRocedural EGOcentric Videos},
    booktitle = {CVPR},
    year = {2024}
}
```