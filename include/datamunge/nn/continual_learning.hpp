#pragma once

/// @file continual_learning.hpp
/// @brief Importance weighting and reservoir selection for continual learning.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Estimate the diagonal empirical Fisher of a binary logistic model.
/// @param x Design matrix [rows,features].
/// @param y Observed binary labels [rows,1].
/// @param weights Logistic coefficients [features,1].
/// @return Mean squared per-example log-likelihood gradients [features,1].
/// @throws std::invalid_argument If dimensions are incompatible or empty.
/// @par Example
/// @code
/// auto fisher=dm::nn::logistic_empirical_fisher(dm::nn::ones({2,3}),
///     dm::nn::zeros({2,1}),dm::nn::zeros({3,1}));
/// @endcode
/// @see consolidation_penalty()
inline Tensor logistic_empirical_fisher(const Tensor& x,const Tensor& y,const Tensor& weights) {
  if (!x.defined() || !y.defined() || !weights.defined() || x.dim()!=2 ||
      x.size(0)<1 || x.size(1)<1 || y.shape()!=Shape{x.size(0),1} ||
      weights.shape()!=Shape{x.size(1),1})
    throw std::invalid_argument("Invalid logistic Fisher dimensions");
  return reshape(mean(square(x*(y-sigmoid(matmul(x,weights)))),{0}),weights.shape());
}

/// @brief Penalize parameter changes in proportion to per-weight importance.
/// @param weights Current weights.
/// @param anchor Prior weights of the same shape.
/// @param importance Nonnegative diagonal importance of the same shape.
/// @param strength Finite nonnegative penalty multiplier.
/// @return Scalar equal to half the weighted squared displacement times strength.
/// @throws std::invalid_argument For incompatible shapes or multiplier.
/// @par Example
/// @code
/// auto penalty=dm::nn::consolidation_penalty(dm::nn::ones({2}),dm::nn::zeros({2}),
///     dm::nn::ones({2}),3);
/// @endcode
/// @par References
/// - Kirkpatrick et al. (2017), Overcoming catastrophic forgetting. doi:10.1073/pnas.1611835114
/// @see synaptic_importance(), logistic_empirical_fisher()
inline Tensor consolidation_penalty(const Tensor& weights,const Tensor& anchor,
                                   const Tensor& importance,double strength) {
  if (!weights.defined() || !anchor.defined() || !importance.defined() ||
      weights.shape()!=anchor.shape() || weights.shape()!=importance.shape() ||
      !std::isfinite(strength) || strength<0)
    throw std::invalid_argument("Invalid consolidation penalty arguments");
  return (.5*strength)*sum(importance*square(weights-anchor));
}

/// @brief Estimate synaptic importance from an optimization trajectory.
/// @param path Weights before and after each step [steps+1,features].
/// @param gradients Loss gradients before each step [steps,features].
/// @param damping Positive offset in the displacement denominator, default 0.1.
/// @return Nonnegative importance [features].
/// @throws std::invalid_argument For incompatible shapes or invalid damping.
/// @par Example
/// @code
/// auto importance=dm::nn::synaptic_importance(dm::nn::zeros({2,3}),dm::nn::zeros({1,3}));
/// @endcode
/// @par References
/// - Zenke et al. (2017), Continual Learning Through Synaptic Intelligence. https://proceedings.mlr.press/v70/zenke17a.html
/// @see consolidation_penalty()
inline Tensor synaptic_importance(const Tensor& path,const Tensor& gradients,double damping=.1) {
  if (!path.defined() || !gradients.defined() || path.dim()!=2 || gradients.dim()!=2 ||
      gradients.size(0)<1 || gradients.size(1)<1 ||
      path.shape()!=Shape{gradients.size(0)+1,gradients.size(1)} ||
      !std::isfinite(damping) || damping<=0)
    throw std::invalid_argument("Invalid synaptic importance trajectory");
  const auto steps=gradients.size(0);
  const auto delta=slice_axis(path,0,1,steps+1)-slice_axis(path,0,0,steps);
  const auto total=select(path,0,steps)-select(path,0,0);
  return relu(-sum(gradients*delta,{0})/(square(total)+damping));
}

/// @brief Sample row IDs uniformly without replacement from a stream of known length.
/// @param count Number of records in the stream.
/// @param capacity Maximum retained records.
/// @param seed Seed for replacement draws.
/// @return Reservoir slot contents, in slot order.
/// @throws std::invalid_argument For negative counts or capacity.
/// @par Example
/// @code
/// auto rows=dm::nn::reservoir_indices(384,32,31255);
/// @endcode
/// @par References
/// - Vitter (1985), Random Sampling with a Reservoir. doi:10.1145/3147.3165
inline std::vector<std::int64_t> reservoir_indices(int count,int capacity,unsigned seed) {
  if (count<0 || capacity<0) throw std::invalid_argument("Invalid reservoir size");
  const int kept=std::min(count,capacity);
  std::vector<std::int64_t> rows(kept);
  std::iota(rows.begin(),rows.end(),0);
  std::mt19937 rng(seed);
  for (int i=kept;i<count;++i) {
    const int slot=std::uniform_int_distribution<int>(0,i)(rng);
    if (slot<kept) rows[slot]=i;
  }
  return rows;
}

/// @brief Distill cached binary teacher probabilities at a common temperature.
/// @param student Unscaled student logits.
/// @param teacher Cached probabilities from sigmoid(teacher_logits / temperature),
/// matching student shape and dtype.
/// @param temperature Finite positive temperature, default two.
/// @return Mean binary cross-entropy multiplied by temperature squared.
/// @throws std::invalid_argument For incompatible tensors or temperature.
/// @par Example
/// @code
/// auto loss=dm::nn::binary_distillation_loss(dm::nn::zeros({2,1}),dm::nn::full({2,1},.5));
/// @endcode
/// @par References
/// - Li and Hoiem (2018), Learning without Forgetting. doi:10.1109/TPAMI.2017.2773087
/// @see sigmoid_cross_entropy()
inline Tensor binary_distillation_loss(const Tensor& student,const Tensor& teacher,
                                       double temperature=2) {
  if (!student.defined() || !teacher.defined() || student.numel()<1 ||
      student.shape()!=teacher.shape() || student.dtype()!=teacher.dtype() ||
      (student.dtype()!=DType::Float32 && student.dtype()!=DType::Float64) ||
      !std::isfinite(temperature*temperature) || !(temperature>0))
    throw std::invalid_argument("Invalid binary distillation inputs");
  return temperature*temperature*sigmoid_cross_entropy(student/temperature,teacher.detach());
}

}  // namespace datamunge::nn
