#!/usr/bin/env python3
"""
Debug script to verify C++ observations match Python expectations.
Run this after the robot starts to analyze the observation data.
"""

import numpy as np
import sys

def check_observation_structure():
    """Verify observation structure matches sim2sim.py"""

    print("=" * 60)
    print("Expected Observation Structure (sim2sim.py)")
    print("=" * 60)

    print("\n1. Motion Reference [0:600] - 20 frames × 30 dims")
    print("   Each frame (30 dims):")
    print("     [0]: root height (z)")
    print("     [1]: roll")
    print("     [2]: pitch")
    print("     [3:6]: root velocity (x, y, z) in root frame")
    print("     [6]: yaw angular velocity")
    print("     [7:30]: 23 joint positions")
    print("   Frame timesteps: [1, 5, 10, 15, ..., 95] * 0.02s")
    print("   = [0.02s, 0.10s, 0.20s, ..., 1.90s] into future")

    print("\n2. Proprioceptive [600:674] - 74 dims")
    print("     [600:603]: angular velocity × 0.25 (IMU)")
    print("     [603:605]: roll, pitch (NO yaw)")
    print("     [605:628]: 23 joint positions - default_pos")
    print("     [628:651]: 23 joint velocities × 0.05 (ankles [4,5,10,11] zeroed)")
    print("     [651:674]: 23 last actions")

    print("\n3. History [674:2154] - 1480 dims")
    print("     Last 20 timesteps of proprio (20 × 74)")

    print("\n" + "=" * 60)
    print("Expected Value Ranges")
    print("=" * 60)

    print("\nMotion Reference:")
    print("  height: ~0.7-1.0m")
    print("  roll/pitch: ~±0.3 rad (±17°)")
    print("  velocities: ~±2.0 m/s or rad/s")
    print("  joint_pos: ~±2.0 rad")

    print("\nProprioceptive:")
    print("  ang_vel × 0.25: ~±0.5 (original ±2 rad/s)")
    print("  roll/pitch: ~±0.3 rad")
    print("  joint_pos - default: ~±1.0 rad")
    print("  joint_vel × 0.05: ~±0.1 (original ±2 rad/s)")
    print("  last_action: ~±5.0 (raw actions before clip)")

    print("\n" + "=" * 60)
    print("Common Issues")
    print("=" * 60)

    print("\n1. Quaternion format mismatch:")
    print("   - NPZ stores: [x, y, z, w]")
    print("   - Eigen expects: (w, x, y, z)")
    print("   - Symptom: roll/pitch >> 1.0 rad")

    print("\n2. Root velocity not transformed:")
    print("   - Must transform to root frame: quat.inverse() * vel")
    print("   - Symptom: velocities don't match motion")

    print("\n3. Angular velocity indices:")
    print("   - Generalized velocity: [lin_vel(3), ang_vel(3), joint_vel(23)]")
    print("   - Angular velocity is indices [3:6]")

    print("\n4. Ankle velocity zeroing:")
    print("   - Must zero indices [4, 5, 10, 11]")
    print("   - Order: left_ankle_pitch, left_ankle_roll, right_ankle_pitch, right_ankle_roll")

    print("\n5. Action scaling:")
    print("   - Must clip [-10, 10] → scale 0.5 → add default_pos")
    print("   - Symptom if wrong: extreme joint angles, twitching")

    print("\n6. PD gains:")
    print("   - MUST match training (sim2sim.py)")
    print("   - Legs: kp=100-150, kd=2-4")
    print("   - Torso: kp=150, kd=4")
    print("   - Arms: kp=40, kd=5")
    print("   - If too high: jerky, unstable")

def parse_ros_log(log_line):
    """Parse observation values from ROS log"""
    if "Motion ref" in log_line and "[0:5]" in log_line:
        # Extract values
        parts = log_line.split("[0:5]:")[-1].strip().split(",")
        values = [float(v.strip()) for v in parts]

        print("\n" + "=" * 60)
        print("Motion Reference Frame 0 Analysis")
        print("=" * 60)
        height, roll, pitch, vel_x, vel_y = values

        print(f"\nHeight: {height:.3f} m")
        if 0.5 < height < 1.2:
            print("  ✓ GOOD: Reasonable standing height")
        else:
            print(f"  ✗ WARNING: Expected 0.7-1.0m for G1")

        print(f"\nRoll: {roll:.3f} rad ({np.degrees(roll):.1f}°)")
        if abs(roll) < 0.5:
            print("  ✓ GOOD: Small roll angle")
        else:
            print(f"  ✗ ERROR: Too large! Check quaternion format!")
            print(f"    Possible [x,y,z,w] loaded as [w,x,y,z]?")

        print(f"\nPitch: {pitch:.3f} rad ({np.degrees(pitch):.1f}°)")
        if abs(pitch) < 0.5:
            print("  ✓ GOOD: Small pitch angle")
        else:
            print(f"  ✗ WARNING: Large pitch")

        print(f"\nRoot velocity: ({vel_x:.3f}, {vel_y:.3f}) m/s")
        if abs(vel_x) < 3.0 and abs(vel_y) < 3.0:
            print("  ✓ GOOD: Reasonable walking velocity")
        else:
            print("  ✗ WARNING: Very high velocity")

if __name__ == "__main__":
    check_observation_structure()

    print("\n" + "=" * 60)
    print("Usage")
    print("=" * 60)
    print("\n1. Launch the controller")
    print("2. Look for log lines like:")
    print('     [walking_controller]: Motion ref [0:5]: X.XXX, X.XXX, ...')
    print("3. Paste the line here or analyze manually")
    print("\n4. Key checks:")
    print("   - Roll should be near 0, not 3.0+")
    print("   - Joint positions should be ±1.0, not ±10.0")
    print("   - No NaN or Inf values")
