#!/usr/bin/env python3
"""
Script to download sample data for PREGO testing
"""

import os
import gdown
import numpy as np
from pathlib import Path


def create_sample_features():
    """Create sample features for testing if real data is not available"""
    print("Creating sample features for testing...")
    
    # Create sample directory
    os.makedirs("data/samples", exist_ok=True)
    
    # Create sample RGB features (2048-dim, 300 frames)
    rgb_features = np.random.randn(300, 2048).astype(np.float32)
    np.save("data/samples/sample_video_rgb.npy", rgb_features)
    
    # Create sample flow features (2048-dim, 300 frames)
    flow_features = np.random.randn(300, 2048).astype(np.float32)
    np.save("data/samples/sample_video_flow.npy", flow_features)
    
    # Create sample ground truth
    gt_actions = np.random.randint(0, 20, size=300)
    np.save("data/samples/sample_video_gt.npy", gt_actions)
    
    print("Sample features created in data/samples/")
    

def download_real_data():
    """Download real TSN features from Google Drive"""
    print("Downloading real PREGO data...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    # Download TSN features
    url = "https://drive.google.com/drive/u/1/folders/1gcOIEXhwysCE2o8-5C4vQnTShJ7p3CKH"
    
    try:
        gdown.download_folder(url, quiet=False, use_cookies=False)
        print("Data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Creating sample data instead...")
        os.chdir("..")
        create_sample_features()
        return
    
    os.chdir("..")
    

def prepare_test_list():
    """Create a list of test videos"""
    test_videos = []
    
    # Check Assembly101-O directory
    assembly_dir = Path("data/Assembly101-O/rgb_anet_resnet50")
    if assembly_dir.exists():
        for video_file in assembly_dir.glob("*.npy"):
            video_name = video_file.stem
            test_videos.append({
                "name": video_name,
                "dataset": "Assembly101-O",
                "rgb_path": str(assembly_dir / f"{video_name}.npy"),
                "flow_path": str(Path("data/Assembly101-O/rgb_as_flow") / f"{video_name}.npy")
            })
    
    # Check Epic-tent-O directory
    epic_dir = Path("data/Epic-tent-O/rgb_anet_resnet50")
    if epic_dir.exists():
        for video_file in epic_dir.glob("*.npy"):
            video_name = video_file.stem
            test_videos.append({
                "name": video_name,
                "dataset": "Epic-tent-O",
                "rgb_path": str(epic_dir / f"{video_name}.npy"),
                "flow_path": str(Path("data/Epic-tent-O/rgb_as_flow") / f"{video_name}.npy")
            })
    
    # Save test list
    import json
    with open("data/test_videos.json", "w") as f:
        json.dump(test_videos[:10], f, indent=2)  # Save first 10 videos
    
    print(f"Found {len(test_videos)} videos. Saved first 10 to data/test_videos.json")
    

def main():
    print("=== PREGO Sample Data Download ===")
    
    # Check if data already exists
    if os.path.exists("data/Assembly101-O") or os.path.exists("data/Epic-tent-O"):
        print("Data directories already exist. Skipping download.")
        prepare_test_list()
        return
    
    # Ask user preference
    choice = input("Download real data (requires ~10GB) or create sample data? [real/sample]: ").lower()
    
    if choice == "real":
        download_real_data()
        prepare_test_list()
    else:
        create_sample_features()
        
        # Create sample test list
        test_videos = [{
            "name": "sample_video",
            "dataset": "sample",
            "rgb_path": "data/samples/sample_video_rgb.npy",
            "flow_path": "data/samples/sample_video_flow.npy"
        }]
        
        import json
        with open("data/test_videos.json", "w") as f:
            json.dump(test_videos, f, indent=2)
    
    print("\nData preparation complete!")
    print("You can now run inference using:")
    print("python run_prego_inference.py --video <video_name> --checkpoint <checkpoint_path>")


if __name__ == "__main__":
    main()