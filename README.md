# motion_tracking_controller

A ROS2 `ros2_control` controller for motion-imitation on the Unitree G1 humanoid robot.

Supports two policy modes:
- **Single-motion mode** (original): ONNX policy with embedded motion reference
- **Generalist mode** (this branch): PyTorch TorchScript policy + runtime `.npz` motion loading

---

## Overview

The controller wraps the `legged_rl_controllers::RlController` base class. In single-motion mode it follows the standard RlController pipeline. In generalist mode it bypasses the `ObservationManager` entirely to manually assemble a 2154-dim observation with precise control over frame ordering, history buffering, and decimation.

### Observation layout (generalist mode)

```
[0:600]    motion_ref   – 20 future frames × 30 dims
[600:674]  proprio      – angular vel, roll/pitch, joint pos, joint vel, last action
[674:2154] history      – last 20 proprioceptive observations
```

---

## Dependencies

| Dependency | Notes |
|---|---|
| ROS2 Humble / Iron | `ros2_control`, `controller_manager` |
| legged_rl_controllers | Base controller framework |
| LibTorch ≥ 2.0 | Installed at `~/.local/libtorch` |
| [cnpy](https://github.com/rogersce/cnpy) | `.npz` loading, installed at `~/.local` |
| Eigen3 | Via system packages |

### Install LibTorch

```bash
# Download CPU or CUDA build from https://pytorch.org/get-started/locally/
unzip libtorch-*.zip -d ~/.local/
```

### Install cnpy

```bash
git clone https://github.com/rogersce/cnpy
cd cnpy && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
make install
```

---

## Build

```bash
cd ~/workspace/colcon_ws
colcon build --packages-select motion_tracking_controller
source install/setup.bash
```

---

## Usage

### Single-motion mode (original ONNX)

```bash
ros2 launch motion_tracking_controller real.launch.py \
  robot_type:=g1 \
  network_interface:=eth0 \
  policy_path:=/path/to/policy.onnx
```

Config (`config/g1/controllers.yaml`):
```yaml
walking_controller:
  ros__parameters:
    policy:
      path: "/path/to/policy.onnx"
```

### Generalist mode (PyTorch + .npz motion)

```bash
ros2 launch motion_tracking_controller real.launch.py \
  robot_type:=g1 \
  network_interface:=eth0 \
  policy_path:=/path/to/generalist.pt \
  use_generalist:=true \
  motion_path:=/path/to/motion.npz
```

Config:
```yaml
walking_controller:
  ros__parameters:
    policy:
      path: "/path/to/generalist.pt"
    motion:
      use_generalist: true
      motion_path: "/path/to/motion.npz"
```

### Motion file format

`.npz` files must contain:
```python
{
  'fps':      float,      # frame rate
  'root_pos': (N, 3),     # root position
  'root_rot': (N, 4),     # root quaternion [w, x, y, z]
  'dof_pos':  (N, 23),    # joint positions (23 joints, no wrists)
}
```

Convert from `.pkl`:
```python
import pickle, numpy as np
data = pickle.load(open("motion.pkl", "rb"))
np.savez("motion.npz", **data)
```

---

## Architecture

```
MotionTrackingController (extends RlController)
├── Single-motion path
│   ├── MotionOnnxPolicy       – ONNX runtime, timestep input
│   ├── MotionCommandTerm      – anchor/body tracking
│   └── ObservationManager     – standard observation assembly
│
└── Generalist path
    ├── TorchPolicy                   – LibTorch JIT inference
    ├── MotionLibrary                 – .npz loading + SLERP interpolation
    ├── GeneralistMotionReferenceObs  – 20 future frames (600 dims)
    ├── GeneralistProprioceptiveObs   – proprioception (74 dims)
    └── GeneralistHistoryObs          – rolling history buffer (1480 dims)
```

**Control loop (generalist):** Controller runs at 500 Hz. Policy inference is decimated to 50 Hz (every 10 cycles). Joint positions are commanded at full 500 Hz using the last computed target.

---

## Changes to main code

The `MotionTrackingController` was extended (not forked) to support both modes with full backward compatibility:

- `on_configure()` – branches on `motion.use_generalist` to load either `TorchPolicy` or `MotionOnnxPolicy`
- `on_activate()` / `on_configure()` – calls `ControllerBase` directly in generalist mode, skipping the `RlController` pipeline that requires `ObservationManager`
- `update()` – overridden to run the generalist control loop with 10× decimation
- `extract23Dofs()` / `expand23DofsTo29()` – new helpers to handle the 29-DOF simulation vs 23-DOF policy mismatch (wrist joints excluded from policy)

**New files added:**

| File | Purpose |
|---|---|
| `src/TorchPolicy.cpp` | LibTorch JIT model loading and inference |
| `src/MotionLibrary.cpp` | `.npz` loading, SLERP/lerp frame interpolation |
| `src/GeneralistObservation.cpp` | Motion reference, proprioception, and history obs terms |

**Build system (`CMakeLists.txt`):** Added `cnpy` (`.npz` loading) and LibTorch (`find_package(Torch)`).

---

## Key implementation notes

**Frame transform** — Angular velocity from `getGeneralizedVelocity()` is in the world frame. It is rotated into the body frame via `base_quat.inverse() * ang_vel_world` before use. This matches the IMU convention the policy was trained with; without it the robot falls immediately.

**DOF mapping** — The G1 simulation has 29 joints; the policy was trained on 23 (wrists excluded). Wrist joints sit at indices 19–21 (left) and 26–28 (right) in the 29-DOF array. `extract23Dofs` / `expand23DofsTo29` handle both directions.

**Action processing** — `action → clip([-10, 10]) → × 0.5 → + default_joint_pos`

**Ankle velocities** — Zeroed at indices 4, 5, 10, 11 in the 23-DOF joint velocity vector to match training.

**Default joint positions (G1):**
```
left leg:   [-0.2, 0.0, 0.0, 0.4, -0.2, 0.0]
right leg:  [-0.2, 0.0, 0.0, 0.4, -0.2, 0.0]
torso:      [ 0.0, 0.0, 0.0]
left arm:   [ 0.0, 0.4, 0.0, 1.2]
right arm:  [ 0.0,-0.4, 0.0, 1.2]
```

---

## PD Gains (generalist mode)

| Joint group | kp | kd |
|---|---|---|
| Legs (hip/knee) | 100–150 | 2–4 |
| Ankles | 40 | 2 |
| Torso | 150 | 4 |
| Arms | 40 | 6 |
| Wrists (passive) | 10 | 1 |

Gains are set once at startup. To tune: start from main-branch values (`config/g1/controllers.yaml`) and reduce by 50–70% if motion is too stiff.

---

## Troubleshooting

**`Expected 2154, got X`** — verify `HISTORY_LENGTH=20`, `PROPRIO_SIZE=74`, motion reference samples exactly 20 frames.

**Segfault on startup** — ensure `jointNameInControl_` lists all 29 joints including wrists.

**Robot falls immediately** — check angular velocity frame transform; world→body rotation is critical.

**Rapid leg oscillation** — policy must run at 50 Hz, not 500 Hz; ensure the decimation counter is intact in `update()`.

**cnpy / LibTorch not found** — confirm install paths match `CMakeLists.txt` hints (`~/.local/lib`, `~/.local/libtorch`).
