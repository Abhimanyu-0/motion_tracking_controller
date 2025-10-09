//
// Created for generalist motion tracking
//

#pragma once

#include <legged_rl_controllers/ObservationManager.h>
#include "motion_tracking_controller/MotionLibrary.h"
#include "motion_tracking_controller/common.h"

#include <deque>
#include <vector>

namespace legged {

/**
 * @brief Motion reference observation (600 dims)
 *
 * Samples 20 future frames from motion library at timesteps:
 * [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
 *
 * Each frame contains 30 dims:
 * - root_height (1), roll (1), pitch (1)
 * - root_vel (3), yaw_ang_vel (1)
 * - joint_pos (23)
 */
class GeneralistMotionReferenceObs : public ObservationTerm {
 public:
  GeneralistMotionReferenceObs(MotionLibrary::SharedPtr motionLib, double controlDt)
      : motionLib_(motionLib), controlDt_(controlDt), currentTime_(0.0) {
    // Timesteps to sample (in units of control steps)
    targetSteps_ = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                    50, 55, 60, 65, 70, 75, 80, 85, 90, 95};
  }

  size_t getSize() const override { return 600; }  // 20 frames × 30 dims

  void setCurrentTime(double time) { currentTime_ = time; }

 protected:
  vector_t evaluate() override;

 private:
  MotionLibrary::SharedPtr motionLib_;
  double controlDt_;  // Control timestep (0.02s for 50Hz)
  double currentTime_;
  std::vector<int> targetSteps_;

  // Convert quaternion to Euler angles
  vector3_t quaternionToEuler(const quaternion_t& q) const;
};

/**
 * @brief Proprioceptive observation (74 dims)
 *
 * - ang_vel (3) × 0.25
 * - roll, pitch (2)
 * - joint_pos - default_pos (23)
 * - joint_vel × 0.05 (23, ankles zeroed)
 * - last_action (23)
 */
class GeneralistProprioceptiveObs : public ObservationTerm {
 public:
  GeneralistProprioceptiveObs(vector_t defaultJointPos)
      : defaultJointPos_(defaultJointPos) {}

  size_t getSize() const override { return 74; }

  void setLastAction(const vector_t& action) { lastAction_ = action; }

 protected:
  vector_t evaluate() override;

 private:
  vector_t defaultJointPos_;
  vector_t lastAction_{vector_t::Zero(23)};

  vector3_t quaternionToEuler(const quaternion_t& q) const;
};

/**
 * @brief History observation (1480 dims)
 *
 * Last 20 timesteps of proprioceptive observations (20 × 74)
 */
class GeneralistHistoryObs : public ObservationTerm {
 public:
  explicit GeneralistHistoryObs(std::deque<vector_t>* historyBuffer)
      : historyBuffer_(historyBuffer) {}

  size_t getSize() const override { return 1480; }  // 20 × 74

 protected:
  vector_t evaluate() override {
    vector_t obs(1480);
    for (size_t i = 0; i < 20; ++i) {
      obs.segment<74>(i * 74) = (*historyBuffer_)[i];
    }
    return obs;
  }

 private:
  std::deque<vector_t>* historyBuffer_;
};

}  // namespace legged
