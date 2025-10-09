//
// Created for generalist motion tracking
//

#include "motion_tracking_controller/TorchPolicy.h"

#include <stdexcept>
#include <iostream>

namespace legged {

TorchPolicy::TorchPolicy(const std::string& modelPath)
    : modelPath_(modelPath),
      device_(torch::kCPU) {  // Use CPU for robot deployment
  std::cout << "[TorchPolicy] Model path: " << modelPath_ << std::endl;
}

void TorchPolicy::init() {
  std::cout << "[TorchPolicy] Loading model..." << std::endl;

  try {
    // Load the model
    module_ = torch::jit::load(modelPath_);
    module_.to(device_);
    module_.eval();  // Set to evaluation mode

    std::cout << "[TorchPolicy] Model loaded successfully" << std::endl;

    // Infer input/output sizes by running a dummy forward pass
    // For generalist policy: input is 2154, output is 23 (G1)
    observationSize_ = 2154;
    actionSize_ = 23;

    std::cout << "[TorchPolicy] Observation size: " << observationSize_ << std::endl;
    std::cout << "[TorchPolicy] Action size: " << actionSize_ << std::endl;

    // Initialize last action
    lastAction_ = vector_t::Zero(actionSize_);

  } catch (const c10::Error& e) {
    throw std::runtime_error("[TorchPolicy] Failed to load model: " + std::string(e.what()));
  }
}

void TorchPolicy::reset() {
  lastAction_ = vector_t::Zero(actionSize_);
  std::cout << "[TorchPolicy] Reset" << std::endl;
}

vector_t TorchPolicy::forward(const vector_t& observations) {
  try {
    // Convert observations to torch tensor
    torch::Tensor obs_tensor = eigenToTorch(observations);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(obs_tensor);

    torch::Tensor output_tensor = module_.forward(inputs).toTensor();

    // Convert back to Eigen
    vector_t actions = torchToEigen(output_tensor);

    // Store last action
    lastAction_ = actions;

    return actions;

  } catch (const c10::Error& e) {
    std::cerr << "[TorchPolicy] Forward pass failed: " << e.what() << std::endl;
    return lastAction_;  // Return last valid action on error
  }
}

torch::Tensor TorchPolicy::eigenToTorch(const vector_t& vec) {
  // Create tensor from Eigen vector
  // Shape: [1, vec.size()] (batch size 1)
  std::vector<float> data(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    data[i] = static_cast<float>(vec(i));
  }

  torch::Tensor tensor = torch::from_blob(
      data.data(),
      {1, static_cast<long>(vec.size())},
      torch::kFloat32
  ).clone();  // Clone to own the data

  return tensor.to(device_);
}

vector_t TorchPolicy::torchToEigen(const torch::Tensor& tensor) {
  // Convert torch tensor back to Eigen vector
  // Assumes tensor shape is [1, action_size]
  torch::Tensor cpu_tensor = tensor.to(torch::kCPU);

  auto accessor = cpu_tensor.accessor<float, 2>();
  size_t size = accessor.size(1);

  vector_t vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec(i) = static_cast<scalar_t>(accessor[0][i]);
  }

  return vec;
}

}  // namespace legged
