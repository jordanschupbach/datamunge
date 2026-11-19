#pragma once

/// @file distillation.hpp
/// @brief Temperature-scaled teacher and student objectives.

#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Blend label cross-entropy with temperature-scaled teacher cross-entropy.
///
/// The teacher is detached. Soft targets come from softmax(teacher / temperature).
/// Their cross-entropy uses student logits divided by the same temperature and
/// has a temperature-squared multiplier. Each row is averaged equally.
/// @param student CPU Float32 or Float64 logits [rows, classes].
/// @param labels Same-shape target probabilities, typically one-hot.
/// @param teacher Same-shape teacher logits with the student's dtype and device.
/// @param temperature Positive finite softmax temperature, default two.
/// @param soft_weight Fraction of the objective assigned to the soft term, default one half.
/// @return Scalar loss differentiable with respect to the student.
/// @throws std::invalid_argument Invalid tensors, temperature, or weight.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto zero = n::zeros({1, 2});
/// const auto loss = n::distillation_loss(zero, n::tensor({1., 0.}, {1, 2}), zero);
/// std::println("{:.6f}", loss.item()); // 1.732868
/// @endcode
/// @par References
/// - Hinton, Vinyals, and Dean (2015), Distilling the Knowledge in a Neural Network,
///   https://arxiv.org/abs/1503.02531
/// @see softmax_cross_entropy(), mse()
[[nodiscard]] inline Tensor distillation_loss(
    const Tensor& student, const Tensor& labels, const Tensor& teacher,
    double temperature = 2., double soft_weight = .5) {
  if (!student.defined() || !labels.defined() || !teacher.defined() ||
      student.dim() != 2 || student.numel() == 0 || labels.shape() != student.shape() ||
      teacher.shape() != student.shape() || student.device().backend != compute::Backend::CPU ||
      labels.device().backend != compute::Backend::CPU || teacher.device().backend != compute::Backend::CPU ||
      (student.dtype() != DType::Float32 && student.dtype() != DType::Float64) ||
      labels.dtype() != student.dtype() || teacher.dtype() != student.dtype() ||
      !(temperature > 0.) || !std::isfinite(temperature * temperature) ||
      !(soft_weight >= 0. && soft_weight <= 1.))
    throw std::invalid_argument("distillation_loss: incompatible tensors or loss settings");
  const auto hard = softmax_cross_entropy(student, labels.detach());
  if (soft_weight == 0.) return hard;
  const auto target = softmax(teacher.detach() / temperature);
  const auto soft = softmax_cross_entropy(student / temperature, target);
  return (1. - soft_weight) * hard + (soft_weight * temperature * temperature) * soft;
}

} // namespace datamunge::nn
