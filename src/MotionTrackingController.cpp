#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"
#include "motion_tracking_controller/TorchPolicy.h"

namespace legged {
controller_interface::CallbackReturn MotionTrackingController::on_init() {
  if (RlController::on_init() != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    // Old single-motion parameters
    auto_declare("motion.start_step", 0);

    // New generalist policy parameters
    auto_declare("motion.use_generalist", false);
    auto_declare("motion.motion_path", "");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during init: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  const auto policyPath = get_node()->get_parameter("policy.path").as_string();
  useGeneralistPolicy_ = get_node()->get_parameter("motion.use_generalist").as_bool();

  // Load policy based on type
  if (useGeneralistPolicy_) {
    // Generalist policy: TorchPolicy + MotionLibrary
    RCLCPP_INFO(get_node()->get_logger(), "Using generalist policy mode");

    const auto motionPath = get_node()->get_parameter("motion.motion_path").as_string();

    if (motionPath.empty()) {
      RCLCPP_ERROR(get_node()->get_logger(), "motion.motion_path must be set for generalist policy!");
      return controller_interface::CallbackReturn::ERROR;
    }

    // Load TorchPolicy
    policy_ = std::make_shared<TorchPolicy>(policyPath);
    policy_->init();
    RCLCPP_INFO_STREAM(get_node()->get_logger(), "Loaded TorchPolicy from " << policyPath);

    // Load motion library
    motionLibrary_ = std::make_shared<MotionLibrary>();
    motionLibrary_->loadMotion(motionPath);
    RCLCPP_INFO_STREAM(get_node()->get_logger(), "Loaded motion from " << motionPath);

    // For generalist, anchor and body names are hardcoded (G1-specific)
    cfg_.anchorBody = "pelvis";
    cfg_.bodyNames = {"pelvis", "torso", "left_hip_yaw_link", "right_hip_yaw_link",
                      "left_knee_link", "right_knee_link", "left_ankle_pitch_link",
                      "right_ankle_pitch_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link"};

    // Initialize history buffer
    for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
      proprioHistoryBuffer_.push_back(vector_t::Zero(PROPRIO_SIZE));
    }

  } else {
    // Single-motion policy: MotionOnnxPolicy
    RCLCPP_INFO(get_node()->get_logger(), "Using single-motion policy mode");

    const auto startStep = static_cast<size_t>(get_node()->get_parameter("motion.start_step").as_int());

    policy_ = std::make_shared<MotionOnnxPolicy>(policyPath, startStep);
    policy_->init();

    auto policy = std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_);
    cfg_.anchorBody = policy->getAnchorBodyName();
    cfg_.bodyNames = policy->getBodyNames();
    RCLCPP_INFO_STREAM(get_node()->get_logger(), "Loaded ONNX model from " << policyPath);
  }

  return RlController::on_configure(previous_state);
}

controller_interface::CallbackReturn MotionTrackingController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

bool MotionTrackingController::parserCommand(const std::string& name) {
  if (RlController::parserCommand(name)) {
    return true;
  }
  if (name == "motion") {
    commandTerm_ = std::make_shared<MotionCommandTerm>(cfg_, std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_));
    commandManager_->addTerm(commandTerm_);
    return true;
  }
  return false;
}

bool MotionTrackingController::parserObservation(const std::string& name) {
  if (RlController::parserObservation(name)) {
    return true;
  }
  if (name == "motion_ref_pos_b" || name == "motion_anchor_pos_b") {
    observationManager_->addTerm(std::make_shared<MotionAnchorPosition>(commandTerm_));
  } else if (name == "motion_ref_ori_b" || name == "motion_anchor_ori_b") {
    observationManager_->addTerm(std::make_shared<MotionAnchorOrientation>(commandTerm_));
  } else if (name == "robot_body_pos") {
    observationManager_->addTerm(std::make_shared<RobotBodyPosition>(commandTerm_));
  } else if (name == "robot_body_ori") {
    observationManager_->addTerm(std::make_shared<RobotBodyOrientation>(commandTerm_));
  } else {
    return false;
  }
  return true;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)
