#pragma once

#include <legged_rl_controllers/RlController.h>
#include <deque>

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionLibrary.h"
#include "motion_tracking_controller/common.h"

namespace legged {
class MotionTrackingController : public RlController {
 public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  bool parserCommand(const std::string& name) override;
  bool parserObservation(const std::string& name) override;

  MotionCommandCfg cfg_;
  MotionCommandTerm::SharedPtr commandTerm_;

  // Generalist policy support
  MotionLibrary::SharedPtr motionLibrary_;
  bool useGeneralistPolicy_{false};

  // History buffer for proprioceptive observations (20 timesteps × 74 dims)
  std::deque<vector_t> proprioHistoryBuffer_;
  static constexpr size_t HISTORY_LENGTH = 20;
  static constexpr size_t PROPRIO_SIZE = 74;
};

}  // namespace legged
