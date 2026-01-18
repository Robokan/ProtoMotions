#!/usr/bin/env python3
"""
Split a motion library into N shards for multi-GPU training.

Each shard can be loaded by a different GPU using the slurmrank.pt naming convention.
Example: g1_phuma_slurmrank.pt with 4 GPUs creates g1_phuma_0.pt, g1_phuma_1.pt, etc.

Usage:
    python scripts/split_motion_lib_for_multigpu.py \
        data/motions/g1_phuma_full.pt \
        data/motions/g1_phuma_slurmrank.pt \
        --num-shards 4
        
Then train with:
    python protomotions/train_agent.py \
        --motion-file data/motions/g1_phuma_slurmrank.pt \
        --ngpu 4 ...
"""

import argparse
import torch
from pathlib import Path


def split_motion_lib(input_path: str, output_pattern: str, num_shards: int):
    """
    Split a motion library into N shards.
    
    Args:
        input_path: Path to input .pt motion library
        output_pattern: Output pattern with 'slurmrank' (e.g., 'data/motions/g1_slurmrank.pt')
        num_shards: Number of shards to create
    """
    print(f"Loading motion library from {input_path}")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    
    num_motions = len(data["motion_lengths"])
    print(f"Total motions: {num_motions}")
    print(f"Splitting into {num_shards} shards (~{num_motions // num_shards} motions each)")
    
    # Calculate motions per shard
    motions_per_shard = num_motions // num_shards
    
    for shard_idx in range(num_shards):
        start_motion = shard_idx * motions_per_shard
        if shard_idx == num_shards - 1:
            # Last shard gets remaining motions
            end_motion = num_motions
        else:
            end_motion = (shard_idx + 1) * motions_per_shard
        
        selected_indices = list(range(start_motion, end_motion))
        num_selected = len(selected_indices)
        
        print(f"\nShard {shard_idx}: motions {start_motion}-{end_motion-1} ({num_selected} motions)")
        
        # Get frame ranges for selected motions
        length_starts = data["length_starts"]
        motion_num_frames = data["motion_num_frames"]
        
        frame_indices = []
        new_motion_num_frames = []
        new_motion_lengths = []
        new_motion_dt = []
        new_motion_weights = []
        new_motion_files = []
        
        for idx in selected_indices:
            start = length_starts[idx].item()
            num_frames = motion_num_frames[idx].item()
            frame_indices.extend(range(start, start + num_frames))
            new_motion_num_frames.append(num_frames)
            new_motion_lengths.append(data["motion_lengths"][idx].item())
            new_motion_dt.append(data["motion_dt"][idx].item())
            new_motion_weights.append(data["motion_weights"][idx].item())
            if "motion_files" in data:
                new_motion_files.append(data["motion_files"][idx])
        
        frame_indices = torch.tensor(frame_indices, dtype=torch.long)
        
        # Create shard data
        shard_data = {}
        
        # Frame-indexed fields
        frame_indexed_fields = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts"]
        if "lrs" in data and data["lrs"] is not None:
            frame_indexed_fields.append("lrs")
        
        for field in frame_indexed_fields:
            if field in data and data[field] is not None:
                shard_data[field] = data[field][frame_indices]
        
        # Rebuild length_starts
        new_motion_num_frames_tensor = torch.tensor(new_motion_num_frames, dtype=torch.long)
        lengths_shifted = new_motion_num_frames_tensor.roll(1)
        lengths_shifted[0] = 0
        shard_data["length_starts"] = lengths_shifted.cumsum(0)
        
        # Motion-indexed fields
        shard_data["motion_num_frames"] = new_motion_num_frames_tensor
        shard_data["motion_lengths"] = torch.tensor(new_motion_lengths, dtype=torch.float32)
        shard_data["motion_dt"] = torch.tensor(new_motion_dt, dtype=torch.float32)
        shard_data["motion_weights"] = torch.tensor(new_motion_weights, dtype=torch.float32)
        
        if new_motion_files:
            shard_data["motion_files"] = tuple(new_motion_files)
        
        # Save shard
        output_path = output_pattern.replace("slurmrank.pt", f"{shard_idx}.pt")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(shard_data, output_path)
        
        shard_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({shard_size_mb:.1f} MB, {len(shard_data['gts'])} frames)")
    
    print(f"\n{'='*60}")
    print(f"Created {num_shards} shards")
    print(f"To train with multi-GPU, use:")
    print(f"  --motion-file {output_pattern} --ngpu {num_shards}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split motion library for multi-GPU training")
    parser.add_argument("input_path", help="Path to input motion library (.pt)")
    parser.add_argument("output_pattern", help="Output pattern with 'slurmrank' (e.g., data/motions/g1_slurmrank.pt)")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shards to create")
    
    args = parser.parse_args()
    
    if "slurmrank" not in args.output_pattern:
        print("Warning: output_pattern should contain 'slurmrank' for multi-GPU training")
        print(f"  Example: {Path(args.output_pattern).stem}_slurmrank.pt")
    
    split_motion_lib(args.input_path, args.output_pattern, args.num_shards)
