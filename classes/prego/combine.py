#!/usr/bin/env python3
"""
Script to combine multiple PREGO dataset directories into one
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def combine_datasets(source_dir, output_dir, dataset_name="Assembly101-O"):
    """Combine multiple dataset directories into one"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir) / dataset_name
    
    # Create output directory structure
    dirs_to_create = [
        output_path / "rgb_anet_resnet50",
        output_path / "rgb_as_flow",
        output_path / "target_perframe"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories in source
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} directories to combine")
    
    # Count total files
    total_files = 0
    file_mapping = {"rgb_anet_resnet50": [], "rgb_as_flow": [], "target_perframe": []}
    
    for subdir in subdirs:
        # RGB files
        rgb_dir = subdir / "rgb_anet_resnet50"
        if rgb_dir.exists():
            files = list(rgb_dir.glob("*.npy"))
            file_mapping["rgb_anet_resnet50"].extend([(f, subdir.name) for f in files])
            total_files += len(files)
        
        # Flow files - handle special case with extra subdirectory
        flow_dir = subdir / "rgb_as_flow"
        if flow_dir.exists():
            # Check if files are directly in rgb_as_flow
            direct_files = list(flow_dir.glob("*.npy"))
            if direct_files:
                file_mapping["rgb_as_flow"].extend([(f, subdir.name) for f in direct_files])
                total_files += len(direct_files)
            else:
                # Check for nested rgb_anet_resnet50 directory
                nested_dir = flow_dir / "rgb_anet_resnet50"
                if nested_dir.exists():
                    files = list(nested_dir.glob("*.npy"))
                    file_mapping["rgb_as_flow"].extend([(f, subdir.name) for f in files])
                    total_files += len(files)
        
        # Target files
        target_dir = subdir / "target_perframe"
        if target_dir.exists():
            files = list(target_dir.glob("*.npy"))
            file_mapping["target_perframe"].extend([(f, subdir.name) for f in files])
            total_files += len(files)
    
    print(f"\nTotal files to copy: {total_files}")
    print(f"RGB files: {len(file_mapping['rgb_anet_resnet50'])}")
    print(f"Flow files: {len(file_mapping['rgb_as_flow'])}")
    print(f"Target files: {len(file_mapping['target_perframe'])}")
    
    # Copy files with progress bar
    with tqdm(total=total_files, desc="Copying files") as pbar:
        for category, files in file_mapping.items():
            output_category_dir = output_path / category
            
            for file_path, source_subdir in files:
                # Check for duplicate filenames
                dest_path = output_category_dir / file_path.name
                if dest_path.exists():
                    # Add source directory prefix to avoid conflicts
                    new_name = f"{source_subdir}_{file_path.name}"
                    dest_path = output_category_dir / new_name
                    print(f"\nWarning: Duplicate filename {file_path.name}, renaming to {new_name}")
                
                shutil.copy2(file_path, dest_path)
                pbar.update(1)
    
    # Print summary
    print(f"\n=== Combination Complete ===")
    print(f"Output directory: {output_path}")
    for category in ["rgb_anet_resnet50", "rgb_as_flow", "target_perframe"]:
        count = len(list((output_path / category).glob("*.npy")))
        print(f"{category}: {count} files")
    
    return output_path


def verify_structure(output_dir):
    """Verify the combined dataset has correct structure"""
    
    print("\n=== Verifying Structure ===")
    
    # Check each video has all three components
    output_path = Path(output_dir)
    
    # Get unique video names from each category
    rgb_files = set(f.stem for f in (output_path / "rgb_anet_resnet50").glob("*.npy"))
    flow_files = set(f.stem for f in (output_path / "rgb_as_flow").glob("*.npy"))
    target_files = set(f.stem for f in (output_path / "target_perframe").glob("*.npy"))
    
    # Find videos with missing components
    all_videos = rgb_files | flow_files | target_files
    complete_videos = rgb_files & flow_files & target_files
    
    missing_rgb = all_videos - rgb_files
    missing_flow = all_videos - flow_files
    missing_target = all_videos - target_files
    
    print(f"Total unique videos: {len(all_videos)}")
    print(f"Complete videos (all 3 components): {len(complete_videos)}")
    
    if missing_rgb:
        print(f"Videos missing RGB: {len(missing_rgb)}")
    if missing_flow:
        print(f"Videos missing Flow: {len(missing_flow)}")
    if missing_target:
        print(f"Videos missing Target: {len(missing_target)}")
    
    if len(complete_videos) == len(all_videos):
        print("✅ All videos have complete data!")
    else:
        print("⚠️  Some videos have missing components")


def main():
    parser = argparse.ArgumentParser(description='Combine multiple PREGO dataset directories')
    parser.add_argument('source_dir', help='Directory containing multiple dataset subdirectories')
    parser.add_argument('--output-dir', default='combined_data', 
                        help='Output directory (default: combined_data)')
    parser.add_argument('--dataset-name', default='Assembly101-O',
                        choices=['Assembly101-O', 'Epic-tent-O'],
                        help='Dataset name for output structure')
    parser.add_argument('--verify', action='store_true',
                        help='Verify structure after combining')
    
    args = parser.parse_args()
    
    # Check source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist")
        return
    
    # Combine datasets
    output_path = combine_datasets(args.source_dir, args.output_dir, args.dataset_name)
    
    # Verify if requested
    if args.verify:
        verify_structure(output_path)


if __name__ == "__main__":
    main()