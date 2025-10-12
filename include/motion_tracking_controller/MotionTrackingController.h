#pragma once

#include <legged_rl_controllers/RlController.h>
#include <deque>

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionLibrary.h"
#include "motion_tracking_controller/GeneralistObservation.h"
#include "motion_tracking_controller/common.h"

namespace legged {
class MotionTrackingController : public RlController {
 public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 public:
  // Override update for generalist policy mode
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

 protected:
  bool parserCommand(const std::string& name) override;
  bool parserObservation(const std::string& name) override;

  // Run generalist policy (custom observation assembly)
  void runGeneralistPolicy();

  // Assemble generalist observation (2154 dims)
  vector_t assembleGeneralistObservation();

  // Extract 23 DOFs from 29 DOFs (skip wrist joints at indices 19-21 and 26-28)
  vector_t extract23Dofs(const vector_t& dofs_29) const;

  // Expand 23 DOFs to 29 DOFs (insert zeros for wrist joints at indices 19-21 and 26-28)
  vector_t expand23DofsTo29(const vector_t& dofs_23) const;

  MotionCommandCfg cfg_;
  MotionCommandTerm::SharedPtr commandTerm_;

  // Generalist policy support
  MotionLibrary::SharedPtr motionLibrary_;
  bool useGeneralistPolicy_{false};
  double motionTime_{0.0};              // Current motion playback time
  double controlDt_{0.02};              // Control timestep (50Hz)
  vector_t defaultJointPos_;            // Default joint positions (G1)

  // Generalist observation terms (not managed by ObservationManager)
  std::shared_ptr<GeneralistMotionReferenceObs> motionRefObs_;
  std::shared_ptr<GeneralistProprioceptiveObs> proprioObs_;
  std::shared_ptr<GeneralistHistoryObs> historyObs_;

  // History buffer for proprioceptive observations (20 timesteps × 74 dims)
  std::deque<vector_t> proprioHistoryBuffer_;
  static constexpr size_t HISTORY_LENGTH = 20;
  static constexpr size_t PROPRIO_SIZE = 74;
};

}  // namespace legged
