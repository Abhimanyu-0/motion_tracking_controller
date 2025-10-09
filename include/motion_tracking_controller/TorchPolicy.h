//
// Created for generalist motion tracking
//

#pragma once

#include <legged_rl_controllers/Policy.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>

namespace legged {

/**
 * @brief Policy implementation using PyTorch JIT (TorchScript)
 *
 * Loads and runs .pt models exported from PyTorch.
 * Compatible with models traced via torch.jit.trace() or torch.jit.script().
 */
class TorchPolicy : public Policy {
 public:
  using SharedPtr = std::shared_ptr<TorchPolicy>;

  /**
   * @brief Constructor
   * @param modelPath Path to .pt model file
   */
  explicit TorchPolicy(const std::string& modelPath);

  ~TorchPolicy() override = default;

  void init() override;
  void reset() override;
  vector_t forward(const vector_t& observations) override;

  size_t getObservationSize() const override { return observationSize_; }
  size_t getActionSize() const override { return actionSize_; }
  vector_t getLastAction() override { return lastAction_; }

 private:
  std::string modelPath_;
  torch::jit::script::Module module_;
  torch::Device device_;

  vector_t lastAction_;
  size_t observationSize_{0};
  size_t actionSize_{0};

  /**
   * @brief Convert Eigen vector to Torch tensor
   */
  torch::Tensor eigenToTorch(const vector_t& vec);

  /**
   * @brief Convert Torch tensor to Eigen vector
   */
  vector_t torchToEigen(const torch::Tensor& tensor);
};

}  // namespace legged
