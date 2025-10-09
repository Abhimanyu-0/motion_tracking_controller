//
// Created for generalist motion tracking
//

#pragma once

#include <legged_model/common.h>
#include <string>
#include <vector>
#include <memory>

namespace legged {

/**
 * @brief Single frame of motion data
 */
struct MotionFrame {
  double time;                  // Time in seconds
  vector3_t root_pos;           // Root position [x, y, z]
  quaternion_t root_rot;        // Root orientation [w, x, y, z]
  vector_t dof_pos;             // Joint positions [num_dofs]
  vector3_t root_vel;           // Root linear velocity
  vector3_t root_ang_vel;       // Root angular velocity (approximate)
};

/**
 * @brief Motion library for loading and sampling motion clips
 *
 * Loads motion data from .npz files (converted from .pkl) and provides
 * frame sampling with interpolation for use in generalist policies.
 */
class MotionLibrary {
 public:
  using SharedPtr = std::shared_ptr<MotionLibrary>;

  MotionLibrary() = default;
  ~MotionLibrary() = default;

  /**
   * @brief Load motion from .npz file
   * @param npz_path Path to .npz motion file
   */
  void loadMotion(const std::string& npz_path);

  /**
   * @brief Sample a single motion frame at a specific time
   * @param time Time in seconds (wraps if > duration)
   * @return Interpolated motion frame
   */
  MotionFrame sampleFrame(double time) const;

  /**
   * @brief Sample multiple motion frames
   * @param times Vector of times to sample
   * @return Vector of interpolated motion frames
   */
  std::vector<MotionFrame> sampleFrames(const std::vector<double>& times) const;

  /**
   * @brief Get motion duration in seconds
   */
  double getDuration() const { return duration_; }

  /**
   * @brief Get motion framerate (FPS)
   */
  double getFPS() const { return fps_; }

  /**
   * @brief Get number of frames
   */
  size_t getNumFrames() const { return frames_.size(); }

  /**
   * @brief Get number of DOFs
   */
  size_t getNumDofs() const { return num_dofs_; }

  /**
   * @brief Check if motion is loaded
   */
  bool isLoaded() const { return !frames_.empty(); }

 private:
  std::vector<MotionFrame> frames_;     // All motion frames
  double fps_{30.0};                     // Frames per second
  double duration_{0.0};                 // Total duration in seconds
  size_t num_dofs_{0};                   // Number of DOFs

  /**
   * @brief Linear interpolation for vectors
   */
  static vector3_t lerpVector3(const vector3_t& v0, const vector3_t& v1, double alpha);

  /**
   * @brief Linear interpolation for joint positions
   */
  static vector_t lerpVector(const vector_t& v0, const vector_t& v1, double alpha);

  /**
   * @brief Spherical linear interpolation (SLERP) for quaternions
   */
  static quaternion_t slerpQuaternion(const quaternion_t& q0, const quaternion_t& q1, double alpha);

  /**
   * @brief Wrap time to motion duration
   */
  double wrapTime(double time) const;
};

}  // namespace legged
