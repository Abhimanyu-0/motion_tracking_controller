#!/usr/bin/env python3
"""
Convert motion .pkl files to .npz format for C++ consumption.

Usage:
    python3 convert_motions.py <input.pkl> [output.npz]
    python3 convert_motions.py --dir <motion_directory>
"""

import pickle
import numpy as np
import argparse
from pathlib import Path


def convert_pkl_to_npz(pkl_path, npz_path=None):
    """Convert a single .pkl motion file to .npz format."""
    pkl_path = Path(pkl_path)

    if npz_path is None:
        npz_path = pkl_path.with_suffix('.npz')
    else:
        npz_path = Path(npz_path)

    print(f"Loading {pkl_path}...")

    try:
        with open(pkl_path, 'rb') as f:
            motion = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return False

    # Extract motion data
    # Expected format from MotionLib
    root_pos = motion.get('root_pos', motion.get('rb_pos'))
    root_rot = motion.get('root_rot', motion.get('rb_rot'))
    dof_pos = motion.get('dof_pos', motion.get('joint_pos'))
    fps = motion.get('fps', 30.0)

    if root_pos is None or root_rot is None or dof_pos is None:
        print(f"Error: Missing required fields in {pkl_path}")
        print(f"Available keys: {motion.keys()}")
        return False

    # Convert to numpy arrays
    root_pos = np.array(root_pos, dtype=np.float32)
    root_rot = np.array(root_rot, dtype=np.float32)
    dof_pos = np.array(dof_pos, dtype=np.float32)

    # Compute velocities via finite difference
    dt = 1.0 / fps
    root_vel = np.gradient(root_pos, dt, axis=0)

    # Angular velocity from quaternions (simplified - using finite diff on quaternions)
    # For more accurate results, should use proper quaternion derivative
    root_ang_vel = np.gradient(root_rot, dt, axis=0)

    print(f"  Motion frames: {len(root_pos)}")
    print(f"  Duration: {len(root_pos) * dt:.2f}s")
    print(f"  FPS: {fps}")
    print(f"  Root pos shape: {root_pos.shape}")
    print(f"  Root rot shape: {root_rot.shape}")
    print(f"  DOF pos shape: {dof_pos.shape}")

    # Save to npz
    np.savez_compressed(
        npz_path,
        root_pos=root_pos,      # [T, 3]
        root_rot=root_rot,      # [T, 4] (w, x, y, z)
        dof_pos=dof_pos,        # [T, num_joints]
        root_vel=root_vel,      # [T, 3]
        root_ang_vel=root_ang_vel,  # [T, 4] (approximation)
        fps=np.array([fps], dtype=np.float32),
        duration=np.array([len(root_pos) * dt], dtype=np.float32)
    )

    print(f"✓ Saved to {npz_path}")
    print()
    return True


def convert_directory(directory):
    """Convert all .pkl files in a directory."""
    directory = Path(directory)
    pkl_files = list(directory.glob("*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {directory}")
        return

    print(f"Found {len(pkl_files)} motion files")
    print("=" * 80)

    success_count = 0
    for pkl_file in pkl_files:
        if convert_pkl_to_npz(pkl_file):
            success_count += 1

    print("=" * 80)
    print(f"Converted {success_count}/{len(pkl_files)} files successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Convert motion .pkl files to .npz format for C++ consumption"
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='Input .pkl file or directory (with --dir)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output .npz file (default: same name as input)'
    )
    parser.add_argument(
        '--dir',
        action='store_true',
        help='Convert all .pkl files in the input directory'
    )

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        return

    if args.dir:
        convert_directory(args.input)
    else:
        convert_pkl_to_npz(args.input, args.output)


if __name__ == "__main__":
    main()
