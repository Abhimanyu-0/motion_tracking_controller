#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"
#include "motion_tracking_controller/MotionOnnxPolicy.h"
#include "motion_tracking_controller/GeneralistObservation.h"
#include "motion_tracking_controller/TorchPolicy.h"

#include <cmath>

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

    // G1 default joint positions (from sim2sim.py:115-121)
    defaultJointPos_.resize(23);
    defaultJointPos_ << -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  // left leg (6)
                        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  // right leg (6)
                         0.0, 0.0, 0.0,                  // torso (3)
                         0.0, 0.4, 0.0, 1.2,             // left arm (4)
                         0.0,-0.4, 0.0, 1.2;             // right arm (4)

    // Initialize history buffer
    for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
      proprioHistoryBuffer_.push_back(vector_t::Zero(PROPRIO_SIZE));
    }

    // Create generalist observation terms (will be used manually, not via ObservationManager)
    motionRefObs_ = std::make_shared<GeneralistMotionReferenceObs>(motionLibrary_, controlDt_);
    proprioObs_ = std::make_shared<GeneralistProprioceptiveObs>(defaultJointPos_);
    historyObs_ = std::make_shared<GeneralistHistoryObs>(&proprioHistoryBuffer_);

    // For generalist, anchor and body names are hardcoded (G1-specific)
    cfg_.anchorBody = "pelvis";
    cfg_.bodyNames = {"pelvis", "torso", "left_hip_yaw_link", "right_hip_yaw_link",
                      "left_knee_link", "right_knee_link", "left_ankle_pitch_link",
                      "right_ankle_pitch_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link"};

    // Initialize desiredPosition_ to 29 DOFs (will be set properly on first update)
    desiredPosition_ = vector_t::Zero(29);

    // Set up joint names for ControllerBase (needed for setPositions to work)
    // G1 robot joint order - all 29 joints including wrists
    jointNameInControl_ = {
      "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
      "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
      "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
      "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
      "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
      "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
      "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
      "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    };

    RCLCPP_INFO(get_node()->get_logger(), "Generalist policy configured successfully");

    // For generalist mode, call ControllerBase::on_configure directly (skip RlController)
    return ControllerBase::on_configure(previous_state);

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
  if (useGeneralistPolicy_) {
    // For generalist mode, skip RlController::on_activate() since we don't use ObservationManager
    // Call ControllerBase::on_activate() directly
    RCLCPP_INFO(get_node()->get_logger(), "Activating generalist policy controller");
    return ControllerBase::on_activate(previous_state);
  }

  // For single-motion mode, use standard RlController activation
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

controller_interface::return_type MotionTrackingController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (useGeneralistPolicy_) {
    // Decimate control to 50Hz (policy frequency)
    // Controller runs at 500Hz, so only run policy every 10 cycles
    static int decimation_counter = 0;
    if (decimation_counter % 10 == 0) {
      // Custom update for generalist policy at 50Hz
      runGeneralistPolicy();
    }
    decimation_counter++;

    // Send commands to hardware (from ControllerBase) at full 500Hz
    setPositions(desiredPosition_);

    // Set PD gains for position control - intermediate values for smoother control
    // Only set once at startup to avoid overhead
    static bool gains_set = false;
    if (!gains_set) {
      // Define gains for 23 DOFs (policy joints, excludes wrists)
      // Reduced gains for smoother, more compliant control (lower impedance)
      vector_t kp_23(23), kd_23(23);
      kp_23 << 100.0, 100.0, 100.0, 150.0, 40.0, 40.0,  // left leg
               100.0, 100.0, 100.0, 150.0, 40.0, 40.0,  // right leg
               150.0, 150.0, 150.0,                      // torso
               40.0, 40.0, 40.0, 40.0,                   // left arm
               40.0, 40.0, 40.0, 40.0;                   // right arm

      kd_23 << 2.0, 2.0, 2.0, 4.0, 2.0, 2.0,  // left leg
               2.0, 2.0, 2.0, 4.0, 2.0, 2.0,  // right leg
               4.0, 4.0, 4.0,                 // torso
               5.0, 5.0, 5.0, 5.0,            // left arm
               5.0, 5.0, 5.0, 5.0;            // right arm

      // Expand to 29 DOFs (add low gains for wrist joints to keep them compliant)
      vector_t kp_29 = expand23DofsTo29(kp_23);
      vector_t kd_29 = expand23DofsTo29(kd_23);

      // Set low stiffness for wrist joints (indices 19-21, 26-28) to keep them passive
      kp_29(19) = kp_29(20) = kp_29(21) = 10.0;  // left wrist
      kp_29(26) = kp_29(27) = kp_29(28) = 10.0;  // right wrist
      kd_29(19) = kd_29(20) = kd_29(21) = 1.0;   // left wrist
      kd_29(26) = kd_29(27) = kd_29(28) = 1.0;   // right wrist

      setStiffnesses(kp_29);
      setDampings(kd_29);
      gains_set = true;

      RCLCPP_INFO(get_node()->get_logger(), "PD gains set (intermediate values ~70%% of main branch)");
    }

    return controller_interface::return_type::OK;
  } else {
    // Use standard RlController update for single-motion mode
    return RlController::update(time, period);
  }
}

void MotionTrackingController::runGeneralistPolicy() {
  // Custom observation assembly for generalist policy
  vector_t observation = assembleGeneralistObservation();

  // Debug: Print observation stats
  static bool first_obs = true;
  if (first_obs) {
    RCLCPP_INFO(get_node()->get_logger(), "First observation assembled: size=%zu", observation.size());
    RCLCPP_INFO(get_node()->get_logger(), "  Motion ref [0:5]: %.3f, %.3f, %.3f, %.3f, %.3f",
                observation(0), observation(1), observation(2), observation(3), observation(4));
    RCLCPP_INFO(get_node()->get_logger(), "  Proprio ang_vel [600:602]: %.3f, %.3f, %.3f",
                observation(600), observation(601), observation(602));
    RCLCPP_INFO(get_node()->get_logger(), "  Proprio roll/pitch [603:604]: %.3f, %.3f",
                observation(603), observation(604));
    RCLCPP_INFO(get_node()->get_logger(), "  Proprio joint_pos [605:610]: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
                observation(605), observation(606), observation(607), observation(608), observation(609), observation(610));

    // Check for NaN or Inf
    bool has_nan = false;
    bool has_inf = false;
    for (int i = 0; i < observation.size(); ++i) {
      if (std::isnan(observation(i))) has_nan = true;
      if (std::isinf(observation(i))) has_inf = true;
    }
    if (has_nan) RCLCPP_ERROR(get_node()->get_logger(), "OBSERVATION CONTAINS NaN!");
    if (has_inf) RCLCPP_ERROR(get_node()->get_logger(), "OBSERVATION CONTAINS Inf!");

    first_obs = false;
  }

  // Run policy inference
  vector_t action = policy_->forward(observation);

  // Debug: Print action stats
  static int call_count = 0;
  if (call_count % 50 == 0) {  // Print every 50 calls (1 second at 50Hz)
    RCLCPP_INFO(get_node()->get_logger(), "Action [0:5]: %.3f, %.3f, %.3f, %.3f, %.3f",
                action(0), action(1), action(2), action(3), action(4));
  }
  call_count++;

  // Update last action for next iteration
  proprioObs_->setLastAction(action);

  // Update proprioceptive history buffer (shift and add new observation)
  // Note: We need to compute current proprio observation again to add to history
  vector_t currentProprio(74);

  // Get base rotation first (needed for multiple computations below)
  quaternion_t base_quat = leggedModel()->getBaseRotation();

  // Get angular velocity from generalized velocity (indices 3-5)
  // NOTE: Generalized velocity is in world frame, transform to body frame like IMU
  vector3_t ang_vel_world = leggedModel()->getGeneralizedVelocity().segment<3>(3);
  vector3_t ang_vel_body = base_quat.inverse() * ang_vel_world;
  currentProprio.segment<3>(0) = ang_vel_body * 0.25;

  // Roll, pitch only (2)
  double sinr_cosp = 2.0 * (base_quat.w() * base_quat.x() + base_quat.y() * base_quat.z());
  double cosr_cosp = 1.0 - 2.0 * (base_quat.x() * base_quat.x() + base_quat.y() * base_quat.y());
  double roll = std::atan2(sinr_cosp, cosr_cosp);
  double sinp = 2.0 * (base_quat.w() * base_quat.y() - base_quat.z() * base_quat.x());
  double pitch = std::asin(std::clamp(sinp, -1.0, 1.0));
  currentProprio(3) = roll;
  currentProprio(4) = pitch;

  // Joint positions relative to default (23)
  // Extract 23 DOFs from 29 DOFs (skip wrist joints)
  vector_t joint_pos_29 = leggedModel()->getGeneralizedPosition().tail(29);
  vector_t joint_pos = extract23Dofs(joint_pos_29);
  currentProprio.segment(5, 23) = joint_pos - defaultJointPos_;

  // Joint velocities (23) × 0.05, ankles zeroed
  vector_t joint_vel_29 = leggedModel()->getGeneralizedVelocity().tail(29);
  vector_t joint_vel = extract23Dofs(joint_vel_29);
  vector_t scaled_vel = joint_vel * 0.05;
  scaled_vel(4) = 0.0;   // left_ankle_pitch
  scaled_vel(5) = 0.0;   // left_ankle_roll
  scaled_vel(10) = 0.0;  // right_ankle_pitch
  scaled_vel(11) = 0.0;  // right_ankle_roll
  currentProprio.segment(28, 23) = scaled_vel;

  // Last action (23)
  currentProprio.segment(51, 23) = action;

  // Update history buffer (remove oldest, add newest)
  proprioHistoryBuffer_.pop_front();
  proprioHistoryBuffer_.push_back(currentProprio);

  // Increment motion time
  motionTime_ += controlDt_;

  // Process action (from sim2sim.py:274-280)
  // 1. Clip action to [-10, 10]
  vector_t clipped_action = action.cwiseMax(-10.0).cwiseMin(10.0);

  // 2. Scale by 0.5
  vector_t scaled_action = clipped_action * 0.5;

  // 3. Add to default position (23 DOFs)
  vector_t desired_23 = scaled_action + defaultJointPos_;

  // 4. Expand to 29 DOFs for simulation (insert zeros for wrist joints)
  desiredPosition_ = expand23DofsTo29(desired_23);

  // Debug: Check if actions are reasonable
  static int debug_count = 0;
  if (debug_count % 50 == 0) {
    RCLCPP_INFO(get_node()->get_logger(), "Raw action range: [%.3f, %.3f]",
                action.minCoeff(), action.maxCoeff());
    RCLCPP_INFO(get_node()->get_logger(), "Target position range: [%.3f, %.3f]",
                desiredPosition_.minCoeff(), desiredPosition_.maxCoeff());
  }
  debug_count++;
}

vector_t MotionTrackingController::assembleGeneralistObservation() {
  vector_t obs(2154);

  // Set current motion time for sampling future frames
  motionRefObs_->setCurrentTime(motionTime_);

  // Set model for all observation terms
  motionRefObs_->setModel(leggedModel());
  proprioObs_->setModel(leggedModel());
  historyObs_->setModel(leggedModel());

  // [0:600] - Motion reference (20 frames × 30 dims)
  obs.segment<600>(0) = motionRefObs_->getValue();

  // [600:674] - Proprioceptive state (74 dims)
  obs.segment<74>(600) = proprioObs_->getValue();

  // [674:2154] - History (last 20 obs_prop × 74 dims)
  obs.segment<1480>(674) = historyObs_->getValue();

  return obs;
}

vector_t MotionTrackingController::extract23Dofs(const vector_t& dofs_29) const {
  // Extract 23 DOFs from 29 DOFs by skipping wrist joints
  // Simulation has 29 joints, policy expects 23 joints (no wrists)
  // Wrist joints are at indices: 19-21 (left wrist), 26-28 (right wrist)
  vector_t dofs_23(23);

  // Copy first 19 joints (legs + torso + left arm up to elbow)
  dofs_23.segment(0, 19) = dofs_29.segment(0, 19);

  // Skip left wrist (indices 19-21), copy right arm (indices 22-25)
  dofs_23.segment(19, 4) = dofs_29.segment(22, 4);

  // Right wrist (indices 26-28) are at the end, so we're done
  return dofs_23;
}

vector_t MotionTrackingController::expand23DofsTo29(const vector_t& dofs_23) const {
  // Expand 23 DOFs to 29 DOFs by inserting zeros for wrist joints
  // Policy outputs 23 joints, but simulation needs 29 joints
  // Insert zeros at indices: 19-21 (left wrist), 26-28 (right wrist)
  vector_t dofs_29 = vector_t::Zero(29);

  // Copy first 19 joints (legs + torso + left arm up to elbow)
  dofs_29.segment(0, 19) = dofs_23.segment(0, 19);

  // Wrist joints [19-21] stay at zero

  // Copy right arm (indices 22-25 in 29-DOF space, indices 19-22 in 23-DOF space)
  dofs_29.segment(22, 4) = dofs_23.segment(19, 4);

  // Wrist joints [26-28] stay at zero

  return dofs_29;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)
