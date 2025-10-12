//
// Created for generalist motion tracking
//

#include "motion_tracking_controller/MotionLibrary.h"

#include <cnpy.h>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace legged {

void MotionLibrary::loadMotion(const std::string& npz_path) {
  std::cout << "[MotionLibrary] Loading motion from: " << npz_path << std::endl;

  try {
    // Load npz file
    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    // Extract arrays
    cnpy::NpyArray root_pos_arr = npz["root_pos"];
    cnpy::NpyArray root_rot_arr = npz["root_rot"];
    cnpy::NpyArray dof_pos_arr = npz["dof_pos"];
    cnpy::NpyArray root_vel_arr = npz["root_vel"];
    cnpy::NpyArray root_ang_vel_arr = npz["root_ang_vel"];
    cnpy::NpyArray fps_arr = npz["fps"];
    cnpy::NpyArray duration_arr = npz["duration"];

    // Get metadata
    fps_ = *fps_arr.data<float>();
    duration_ = *duration_arr.data<float>();

    // Get dimensions
    size_t num_frames = root_pos_arr.shape[0];
    num_dofs_ = dof_pos_arr.shape[1];

    std::cout << "  Frames: " << num_frames << std::endl;
    std::cout << "  Duration: " << duration_ << "s" << std::endl;
    std::cout << "  FPS: " << fps_ << std::endl;
    std::cout << "  DOFs: " << num_dofs_ << std::endl;

    // Reserve space
    frames_.reserve(num_frames);

    // Load all frames
    float* root_pos_data = root_pos_arr.data<float>();
    float* root_rot_data = root_rot_arr.data<float>();
    float* dof_pos_data = dof_pos_arr.data<float>();
    float* root_vel_data = root_vel_arr.data<float>();
    float* root_ang_vel_data = root_ang_vel_arr.data<float>();

    for (size_t i = 0; i < num_frames; ++i) {
      MotionFrame frame;
      frame.time = i / fps_;

      // Root position [3]
      frame.root_pos = vector3_t(
          root_pos_data[i * 3 + 0],
          root_pos_data[i * 3 + 1],
          root_pos_data[i * 3 + 2]
      );

      // Root rotation [4] - quaternion stored as [x, y, z, w] in npz
      // Eigen quaternion constructor is (w, x, y, z)
      frame.root_rot = quaternion_t(
          root_rot_data[i * 4 + 3],  // w (from index 3)
          root_rot_data[i * 4 + 0],  // x (from index 0)
          root_rot_data[i * 4 + 1],  // y (from index 1)
          root_rot_data[i * 4 + 2]   // z (from index 2)
      );
      frame.root_rot.normalize();

      // DOF positions [num_dofs]
      frame.dof_pos.resize(num_dofs_);
      for (size_t j = 0; j < num_dofs_; ++j) {
        frame.dof_pos(j) = dof_pos_data[i * num_dofs_ + j];
      }

      // Root velocity [3]
      frame.root_vel = vector3_t(
          root_vel_data[i * 3 + 0],
          root_vel_data[i * 3 + 1],
          root_vel_data[i * 3 + 2]
      );

      // Root angular velocity - stored as [x, y, z, ?] (4D with last element unknown)
      // Use first 3 elements
      frame.root_ang_vel = vector3_t(
          root_ang_vel_data[i * 4 + 0],  // x
          root_ang_vel_data[i * 4 + 1],  // y
          root_ang_vel_data[i * 4 + 2]   // z
      );

      frames_.push_back(frame);
    }

    std::cout << "[MotionLibrary] Motion loaded successfully!" << std::endl;

  } catch (const std::exception& e) {
    throw std::runtime_error("[MotionLibrary] Failed to load motion: " + std::string(e.what()));
  }
}

MotionFrame MotionLibrary::sampleFrame(double time) const {
  if (frames_.empty()) {
    throw std::runtime_error("[MotionLibrary] No motion loaded!");
  }

  // Wrap time to motion duration
  time = wrapTime(time);

  // Find frame indices for interpolation
  double frame_idx_float = time * fps_;
  size_t frame_idx0 = static_cast<size_t>(std::floor(frame_idx_float));
  size_t frame_idx1 = (frame_idx0 + 1) % frames_.size();
  double alpha = frame_idx_float - frame_idx0;

  // Get frames
  const MotionFrame& frame0 = frames_[frame_idx0];
  const MotionFrame& frame1 = frames_[frame_idx1];

  // Interpolate
  MotionFrame result;
  result.time = time;
  result.root_pos = lerpVector3(frame0.root_pos, frame1.root_pos, alpha);
  result.root_rot = slerpQuaternion(frame0.root_rot, frame1.root_rot, alpha);
  result.dof_pos = lerpVector(frame0.dof_pos, frame1.dof_pos, alpha);
  result.root_vel = lerpVector3(frame0.root_vel, frame1.root_vel, alpha);
  result.root_ang_vel = lerpVector3(frame0.root_ang_vel, frame1.root_ang_vel, alpha);

  return result;
}

std::vector<MotionFrame> MotionLibrary::sampleFrames(const std::vector<double>& times) const {
  std::vector<MotionFrame> results;
  results.reserve(times.size());

  for (double time : times) {
    results.push_back(sampleFrame(time));
  }

  return results;
}

double MotionLibrary::wrapTime(double time) const {
  if (duration_ <= 0.0) return 0.0;

  // Wrap time to [0, duration)
  time = std::fmod(time, duration_);
  if (time < 0.0) {
    time += duration_;
  }

  return time;
}

vector3_t MotionLibrary::lerpVector3(const vector3_t& v0, const vector3_t& v1, double alpha) {
  return (1.0 - alpha) * v0 + alpha * v1;
}

vector_t MotionLibrary::lerpVector(const vector_t& v0, const vector_t& v1, double alpha) {
  return (1.0 - alpha) * v0 + alpha * v1;
}

quaternion_t MotionLibrary::slerpQuaternion(const quaternion_t& q0, const quaternion_t& q1, double alpha) {
  // SLERP implementation
  quaternion_t q0_copy = q0;
  quaternion_t q1_copy = q1;

  // Compute dot product
  double dot = q0_copy.w() * q1_copy.w() +
               q0_copy.x() * q1_copy.x() +
               q0_copy.y() * q1_copy.y() +
               q0_copy.z() * q1_copy.z();

  // If dot < 0, negate q1 to take shorter path
  if (dot < 0.0) {
    q1_copy.w() = -q1_copy.w();
    q1_copy.x() = -q1_copy.x();
    q1_copy.y() = -q1_copy.y();
    q1_copy.z() = -q1_copy.z();
    dot = -dot;
  }

  // If quaternions are very close, use linear interpolation
  const double DOT_THRESHOLD = 0.9995;
  if (dot > DOT_THRESHOLD) {
    quaternion_t result(
        q0_copy.w() + alpha * (q1_copy.w() - q0_copy.w()),
        q0_copy.x() + alpha * (q1_copy.x() - q0_copy.x()),
        q0_copy.y() + alpha * (q1_copy.y() - q0_copy.y()),
        q0_copy.z() + alpha * (q1_copy.z() - q0_copy.z())
    );
    result.normalize();
    return result;
  }

  // SLERP
  double theta_0 = std::acos(dot);
  double theta = theta_0 * alpha;
  double sin_theta = std::sin(theta);
  double sin_theta_0 = std::sin(theta_0);

  double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
  double s1 = sin_theta / sin_theta_0;

  quaternion_t result(
      s0 * q0_copy.w() + s1 * q1_copy.w(),
      s0 * q0_copy.x() + s1 * q1_copy.x(),
      s0 * q0_copy.y() + s1 * q1_copy.y(),
      s0 * q0_copy.z() + s1 * q1_copy.z()
  );
  result.normalize();

  return result;
}

}  // namespace legged
