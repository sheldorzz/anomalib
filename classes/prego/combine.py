#!/usr/bin/env python3
"""
Script to combine multiple Assembly101-O directories into one
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def combine_datasets(source_dirs, output_dir="Assembly101-O_combined", move=False):
    """
    Combine multiple Assembly101-O directories into one
    
    Args:
        source_dirs: List of source directory paths
        output_dir: Output directory name
        move: If True, move files instead of copying
    """
    
    # Create output directory structure
    output_path = Path(output_dir)
    subdirs = ["rgb_anet_resnet50", "rgb_as_flow", "target_perframe"]
    
    for subdir in subdirs:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {subdir: 0 for subdir in subdirs}
    total_size = 0
    
    print(f"Combining {len(source_dirs)} directories into {output_dir}")
    print(f"Operation: {'Moving' if move else 'Copying'} files\n")
    
    # Process each source directory
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue
            
        print(f"Processing: {source_dir}")
        
        # Process each subdirectory
        for subdir in subdirs:
            src_subdir = source_path / subdir
            dst_subdir = output_path / subdir
            
            if not src_subdir.exists():
                print(f"  Warning: {src_subdir} does not exist")
                continue
            
            # Get all .npy files
            npy_files = list(src_subdir.glob("*.npy"))
            
            # Process files with progress bar
            for npy_file in tqdm(npy_files, desc=f"  {subdir}", unit="files"):
                dst_file = dst_subdir / npy_file.name
                
                # Handle naming conflicts
                if dst_file.exists():
                    # Check if files are identical
                    if os.path.getsize(npy_file) == os.path.getsize(dst_file):
                        print(f"\n  Skipping duplicate: {npy_file.name}")
                        continue
                    else:
                        # Rename with source directory prefix
                        new_name = f"{source_path.name}_{npy_file.name}"
                        dst_file = dst_subdir / new_name
                        print(f"\n  Renaming conflict: {npy_file.name} -> {new_name}")
                
                # Copy or move file
                if move:
                    shutil.move(str(npy_file), str(dst_file))
                else:
                    shutil.copy2(str(npy_file), str(dst_file))
                
                stats[subdir] += 1
                total_size += os.path.getsize(dst_file)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Output directory: {output_dir}")
    for subdir, count in stats.items():
        print(f"  {subdir}: {count} files")
    print(f"Total files: {sum(stats.values())}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    # Verify completeness
    print("\n=== Verification ===")
    for subdir in subdirs:
        actual_count = len(list((output_path / subdir).glob("*.npy")))
        print(f"{subdir}: {actual_count} files in output")


def main():
    parser = argparse.ArgumentParser(description='Combine multiple Assembly101-O datasets')
    parser.add_argument('source_dirs', nargs='+', help='Source directories to combine')
    parser.add_argument('-o', '--output', default='Assembly101-O_combined', 
                        help='Output directory name (default: Assembly101-O_combined)')
    parser.add_argument('-m', '--move', action='store_true', 
                        help='Move files instead of copying')
    
    args = parser.parse_args()
    
    # Example usage for 8 directories
    if len(args.source_dirs) == 1 and args.source_dirs[0] == 'auto':
        # Auto-detect directories matching pattern
        source_dirs = []
        for i in range(1, 9):
            dirs = [
                f"Assembly101-O_{i}",
                f"Assembly101-O_part{i}",
                f"Assembly101-O-{i}",
                f"assembly101_o_{i}"
            ]
            for d in dirs:
                if Path(d).exists():
                    source_dirs.append(d)
                    break
        
        if not source_dirs:
            print("No directories found. Please specify them manually.")
            return
    else:
        source_dirs = args.source_dirs
    
    combine_datasets(source_dirs, args.output, args.move)


if __name__ == "__main__":
    main()

# Example usage:
# python combine_assembly_datasets.py Assembly101-O_1 Assembly101-O_2 Assembly101-O_3 Assembly101-O_4 Assembly101-O_5 Assembly101-O_6 Assembly101-O_7 Assembly101-O_8
# 
# Or if directories are named differently:
# python combine_assembly_datasets.py dir1 dir2 dir3 dir4 dir5 dir6 dir7 dir8 -o Assembly101-O
#
# To move instead of copy (faster but removes originals):
# python combine_assembly_datasets.py dir1 dir2 dir3 dir4 dir5 dir6 dir7 dir8 --move