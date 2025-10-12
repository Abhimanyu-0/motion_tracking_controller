//
// Created for generalist motion tracking
//

#include "motion_tracking_controller/GeneralistObservation.h"
#include <cmath>

namespace legged {

vector3_t GeneralistMotionReferenceObs::quaternionToEuler(const quaternion_t& q) const {
  // Roll (x-axis rotation)
  double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
  double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
  double roll = std::atan2(sinr_cosp, cosr_cosp);

  // Pitch (y-axis rotation)
  double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
  double pitch = std::asin(std::clamp(sinp, -1.0, 1.0));

  // Yaw (z-axis rotation)
  double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
  double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
  double yaw = std::atan2(siny_cosp, cosy_cosp);

  return vector3_t(roll, pitch, yaw);
}

vector_t GeneralistMotionReferenceObs::evaluate() {
  vector_t obs(600);

  // Sample 20 future frames
  std::vector<double> times;
  for (int step : targetSteps_) {
    times.push_back(currentTime_ + step * controlDt_);
  }

  auto frames = motionLib_->sampleFrames(times);

  // Pack into observation
  for (size_t i = 0; i < 20; ++i) {
    const auto& frame = frames[i];
    size_t offset = i * 30;

    // Get Euler angles
    vector3_t euler = quaternionToEuler(frame.root_rot);

    // Rotate velocities to root frame
    vector3_t root_vel_local = frame.root_rot.inverse() * frame.root_vel;
    vector3_t root_ang_vel_local = frame.root_rot.inverse() * frame.root_ang_vel;

    // Pack 30 dims
    obs(offset + 0) = frame.root_pos.z();       // root height
    obs(offset + 1) = euler.x();                 // roll
    obs(offset + 2) = euler.y();                 // pitch
    obs.segment<3>(offset + 3) = root_vel_local; // lin vel (x, y, z)
    obs(offset + 6) = root_ang_vel_local.z();    // yaw ang vel
    obs.segment(offset + 7, 23) = frame.dof_pos; // joint positions (23)
  }

  return obs;
}

vector3_t GeneralistProprioceptiveObs::quaternionToEuler(const quaternion_t& q) const {
  double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
  double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
  double roll = std::atan2(sinr_cosp, cosr_cosp);

  double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
  double pitch = std::asin(std::clamp(sinp, -1.0, 1.0));

  double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
  double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
  double yaw = std::atan2(siny_cosp, cosy_cosp);

  return vector3_t(roll, pitch, yaw);
}

vector_t GeneralistProprioceptiveObs::evaluate() {
  vector_t obs(74);

  // Get base rotation once (used for ang_vel and euler)
  quaternion_t base_quat = model_->getBaseRotation();

  // Get angular velocity from generalized velocity (indices 3-5)
  // NOTE: Generalized velocity is in world frame, transform to body frame like IMU
  vector3_t ang_vel_world = model_->getGeneralizedVelocity().segment<3>(3);
  vector3_t ang_vel_body = base_quat.inverse() * ang_vel_world;
  obs.segment<3>(0) = ang_vel_body * 0.25;

  // Roll, pitch only (2) - no yaw for orientation invariance
  vector3_t euler = quaternionToEuler(base_quat);
  obs(3) = euler.x();  // roll
  obs(4) = euler.y();  // pitch

  // Joint positions relative to default (23)
  // Generalized position has: [pos(3), quat(4), joints(29)]
  // Extract 23 DOFs from 29 DOFs (skip wrist joints)
  vector_t joint_pos_29 = model_->getGeneralizedPosition().tail(29);
  vector_t joint_pos = extract23Dofs(joint_pos_29);
  obs.segment(5, 23) = joint_pos - defaultJointPos_;

  // Joint velocities (23) × 0.05, ankles zeroed
  // Generalized velocity has: [lin_vel(3), ang_vel(3), joint_vel(29)]
  vector_t joint_vel_29 = model_->getGeneralizedVelocity().tail(29);
  vector_t joint_vel = extract23Dofs(joint_vel_29);
  vector_t scaled_vel = joint_vel * 0.05;

  // Zero out ankle velocities (indices 4, 5, 10, 11)
  scaled_vel(4) = 0.0;   // left_ankle_pitch
  scaled_vel(5) = 0.0;   // left_ankle_roll
  scaled_vel(10) = 0.0;  // right_ankle_pitch
  scaled_vel(11) = 0.0;  // right_ankle_roll
  obs.segment(28, 23) = scaled_vel;

  // Last action (23)
  obs.segment(51, 23) = lastAction_;

  return obs;
}

vector_t GeneralistProprioceptiveObs::extract23Dofs(const vector_t& dofs_29) const {
  // Extract 23 DOFs from 29 DOFs by skipping wrist joints
  // Wrist joints are at indices: 19-21 (left wrist), 26-28 (right wrist)
  vector_t dofs_23(23);

  // Copy first 19 joints (legs + torso + left arm up to elbow)
  dofs_23.segment(0, 19) = dofs_29.segment(0, 19);

  // Skip left wrist (indices 19-21), copy right arm (indices 22-25)
  dofs_23.segment(19, 4) = dofs_29.segment(22, 4);

  return dofs_23;
}

}  // namespace legged
