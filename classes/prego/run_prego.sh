#!/bin/bash

# PREGO Complete Run Script
# This script runs the complete PREGO pipeline

echo "=== PREGO Complete Pipeline ==="

# Parse command line arguments
DATASET="Assembly101-O"
MODE="inference"  # Can be "train", "eval", or "inference"
VIDEO=""
CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --video)
            VIDEO="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if we're in the PREGO directory
if [ ! -f "step_recognition/main.py" ]; then
    echo "Error: Must run from PREGO directory"
    exit 1
fi

# Function to train MiniROAD
train_miniroad() {
    echo "Training MiniROAD for $DATASET..."
    python step_recognition/main.py \
        --config step_recognition/configs/miniroad_${DATASET,,}.yaml
}

# Function to evaluate MiniROAD
eval_miniroad() {
    if [ -z "$CHECKPOINT" ]; then
        echo "Error: --checkpoint required for evaluation"
        exit 1
    fi
    
    echo "Evaluating MiniROAD on $DATASET..."
    python step_recognition/main.py \
        --config step_recognition/configs/miniroad_${DATASET,,}.yaml \
        --eval $CHECKPOINT
}

# Function to run complete inference
run_inference() {
    if [ -z "$VIDEO" ] || [ -z "$CHECKPOINT" ]; then
        echo "Error: --video and --checkpoint required for inference"
        exit 1
    fi
    
    echo "Running inference on $VIDEO..."
    
    # Step 1: Run step recognition
    echo "Step 1: Running step recognition..."
    python step_recognition/main.py \
        --config step_recognition/configs/miniroad_${DATASET,,}.yaml \
        --eval $CHECKPOINT \
        --video $VIDEO \
        --output output_miniROAD/${VIDEO}_predictions.json
    
    # Step 2: Aggregate predictions
    echo "Step 2: Aggregating predictions..."
    python utils/aggregate.py \
        output_miniROAD/${VIDEO}_predictions.json \
        step_anticipation/data/predictions/${VIDEO}_aggregated.json
    
    # Step 3: Run anticipation (if LLAMA is available)
    if [ -f "step_anticipation/scripts/anticipation.sh" ]; then
        echo "Step 3: Running anticipation module..."
        cd step_anticipation
        ./scripts/anticipation.sh \
            --input data/predictions/${VIDEO}_aggregated.json \
            --dataset $DATASET
        cd ..
    else
        echo "Step 3: Skipping anticipation (LLAMA not configured)"
    fi
    
    # Step 4: Generate final results
    echo "Step 4: Generating final results..."
    python ../run_prego_inference.py \
        --video data/${DATASET}/rgb_anet_resnet50/${VIDEO} \
        --config step_recognition/configs/miniroad_${DATASET,,}.yaml \
        --checkpoint $CHECKPOINT \
        --dataset $DATASET \
        --output results/${VIDEO}_final.json
}

# Function to run batch inference
run_batch_inference() {
    if [ -z "$CHECKPOINT" ]; then
        echo "Error: --checkpoint required for batch inference"
        exit 1
    fi
    
    echo "Running batch inference on test videos..."
    
    # Read test videos list
    if [ -f "data/test_videos.json" ]; then
        # Extract video names using Python
        python -c "
import json
with open('data/test_videos.json', 'r') as f:
    videos = json.load(f)
    for v in videos:
        if v['dataset'] == '$DATASET':
            print(v['name'])
" | while read video; do
            echo "Processing $video..."
            VIDEO=$video CHECKPOINT=$CHECKPOINT run_inference
        done
    else
        echo "Error: data/test_videos.json not found. Run download_sample_data.py first."
        exit 1
    fi
}

# Main execution
case $MODE in
    train)
        train_miniroad
        ;;
    eval)
        eval_miniroad
        ;;
    inference)
        run_inference
        ;;
    batch)
        run_batch_inference
        ;;
    *)
        echo "Error: Unknown mode $MODE"
        echo "Usage: $0 --mode [train|eval|inference|batch] [options]"
        echo "Options:"
        echo "  --dataset [Assembly101-O|Epic-tent-O]"
        echo "  --video <video_name> (for inference mode)"
        echo "  --checkpoint <path> (for eval/inference modes)"
        exit 1
        ;;
esac

echo "=== Pipeline Complete ===