#pragma once
/// @file series_classification.hpp
/// @brief Compact InceptionTime and ROCKET components for univariate classification.

#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Classify histories using one residual group of three multiscale convolution blocks.
/// @param store CPU parameter store matching signal dtype.
/// @param name Parameter prefix.
/// @param signal Floating CPU tensor [batch,time,1].
/// @param classes Positive output width.
/// @return Differentiable unnormalized scores [batch,classes].
/// @throws std::invalid_argument Invalid input or stored shape.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto y = dm::nn::inception_time(p, "model", dm::nn::ones({2,64,1}), 3);
/// std::println("{}", y.size(1)); // 3
/// @endcode
/// @par References
/// - Fawaz et al., InceptionTime, https://arxiv.org/abs/1909.04939
/// @see conv1d(), layer_norm()
[[nodiscard]] inline Tensor inception_time(ParameterStore& store, const std::string& name,
                                            const Tensor& signal, std::int64_t classes) {
  if (!signal.defined() || signal.dim() != 3 || signal.size(0) < 1 || signal.size(1) < 1 ||
      signal.size(2) != 1 || signal.dtype() != store.dtype() ||
      signal.device().backend != compute::Backend::CPU || classes < 1)
    throw std::invalid_argument("inception_time: invalid series or classes");
  auto state = signal;
  for (int block = 0; block < 3; ++block) {
    const auto prefix = name + "/inception" + std::to_string(block);
    const auto reduced = block == 0 ? state : conv1d(store, prefix + "/reduce", state, 8, 1);
    std::vector<Tensor> branches;
    for (int kernel : {9, 19, 39})
      branches.push_back(conv1d(store, prefix + "/conv" + std::to_string(kernel), reduced, 6, kernel));
    branches.push_back(conv1d(store, prefix + "/pool", max_pool1d(state, 3, 1, Padding::Same), 6, 1));
    state = relu(layer_norm(store, prefix + "/norm", concat(branches, 2)));
  }
  const auto shortcut = conv1d(store, name + "/skip", signal, 24, 1);
  return dense(store, name + "/head", mean(relu(state + shortcut), {1}), classes);
}

/// @brief One frozen ROCKET filter.
/// @par Example
/// @code
/// auto bank = dm::nn::rocket_bank(64, 1, 44201);
/// std::println("{}", bank.size()); // 1
/// @endcode
/// @see rocket_bank(), rocket_features()
struct RocketKernel {
  std::vector<float> taps; ///< Correlation taps before dilation.
  std::int64_t dilation; ///< Positive tap spacing.
  bool same; ///< Symmetric zero padding if true, valid output otherwise.
  float bias; ///< Offset added before computing features.
};

/// @brief Generate seeded zero-mean Gaussian kernels with random bias, spacing, and padding.
/// @param length Signal length, at least 11.
/// @param count Positive kernel count.
/// @param seed MT19937 initialization seed.
/// @return Reusable, ordered bank of frozen kernels.
/// @throws std::invalid_argument Invalid length or count.
/// @par Example
/// @code
/// auto bank = dm::nn::rocket_bank(64, 2, 44201);
/// std::println("{}", bank.size()); // 2
/// @endcode
/// @par References
/// - Dempster et al., ROCKET, https://arxiv.org/abs/1910.13051
/// @see rocket_features()
[[nodiscard]] inline std::vector<RocketKernel> rocket_bank(
    std::int64_t length, std::int64_t count, unsigned seed) {
  if (length < 11 || count < 1) throw std::invalid_argument("rocket_bank: invalid length or count");
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> unit(0, 1), offset(-1, 1);
  std::vector<RocketKernel> bank;
  bank.reserve(count);
  for (std::int64_t i = 0; i < count; ++i) {
    const int taps = 7 + 2 * (rng() % 3);
    const auto limit = (length - 1) / (taps - 1);
    const auto dilation = static_cast<std::int64_t>(std::pow(limit, unit(rng)));
    const bool same = rng() % 2;
    std::vector<float> weights(taps);
    float center = 0;
    for (auto& value : weights) {
      value = normal(rng);
      center += value / taps;
    }
    for (auto& value : weights) value -= center;
    bank.push_back({std::move(weights), dilation, same, offset(rng)});
  }
  return bank;
}

/// @brief Extract maximum response and proportion positive for each frozen ROCKET kernel.
/// @param signal Floating CPU tensor [batch,time,1].
/// @param bank Nonempty ordered kernel bank with each dilated span no greater than time.
/// @return Features [batch,2 * kernels] alternating maximum and proportion positive.
/// @throws std::invalid_argument Invalid series or kernel configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto features = n::rocket_features(n::ones({2,64,1}), n::rocket_bank(64, 2, 44201));
/// std::println("{}", features.size(1)); // 4
/// @endcode
/// @see rocket_bank(), conv1d()
[[nodiscard]] inline Tensor rocket_features(const Tensor& signal, const std::vector<RocketKernel>& bank) {
  if (!signal.defined() || signal.dim() != 3 || signal.size(0) < 1 || signal.size(1) < 1 ||
      signal.size(2) != 1 || signal.device().backend != compute::Backend::CPU || bank.empty() ||
      (signal.dtype() != DType::Float32 && signal.dtype() != DType::Float64))
    throw std::invalid_argument("rocket_features: invalid signal or bank");
  std::vector<Tensor> features;
  for (const auto& kernel : bank) {
    if (kernel.taps.empty() || kernel.dilation < 1 ||
        kernel.taps.size() - 1 > static_cast<std::size_t>((signal.size(1) - 1) / kernel.dilation))
      throw std::invalid_argument("rocket_features: invalid kernel span");
    const auto taps = from_vector(kernel.taps, {static_cast<std::int64_t>(kernel.taps.size()), 1, 1});
    const auto response = conv1d(signal, taps.to(signal.dtype()), 1,
                                 kernel.same ? Padding::Same : Padding::Valid, kernel.dilation) + kernel.bias;
    features.push_back(max(response, {1}));
    features.push_back(mean(greater(response, 0.).to(signal.dtype()), {1}));
  }
  return concat(features, 1);
}

/// @brief Fit a ridge head with a final, unpenalized intercept row.
/// @param features Floating CPU training features [rows,features].
/// @param labels Floating CPU training targets [rows,classes].
/// @param penalty Finite nonnegative squared-norm penalty, default one.
/// @return Float64 coefficients [features + 1,classes], with intercept last.
/// @throws std::invalid_argument Invalid dimensions, dtypes, or penalty.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto beta = n::rocket_ridge(n::ones({4,2}), n::ones({4,1}));
/// std::println("{}", beta.size(0)); // 3
/// @endcode
/// @see rocket_features(), lstsq()
[[nodiscard]] inline Tensor rocket_ridge(const Tensor& features, const Tensor& labels, double penalty = 1.) {
  if (!features.defined() || !labels.defined() || features.dim() != 2 || labels.dim() != 2 ||
      features.size(0) < 1 || features.size(1) < 1 || labels.size(1) < 1 || labels.size(0) != features.size(0) ||
      features.device().backend != compute::Backend::CPU || labels.device().backend != compute::Backend::CPU ||
      (features.dtype() != DType::Float32 && features.dtype() != DType::Float64) ||
      (labels.dtype() != DType::Float32 && labels.dtype() != DType::Float64) ||
      !(penalty >= 0 && std::isfinite(penalty)))
    throw std::invalid_argument("rocket_ridge: invalid fitting data or penalty");
  const auto rows = features.size(0), cols = features.size(1);
  constexpr auto dtype = DType::Float64;
  const auto design = concat({features.to(dtype), ones({rows, 1}, dtype)}, 1);
  const auto regularizer = concat({eye(cols, dtype) * std::sqrt(penalty), zeros({cols, 1}, dtype)}, 1);
  return lstsq(concat({design, regularizer}, 0),
               concat({labels.to(dtype), zeros({cols, labels.size(1)}, dtype)}, 0), 0., false);
}

}  // namespace datamunge::nn
