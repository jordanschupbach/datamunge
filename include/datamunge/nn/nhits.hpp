#pragma once

/// @file nhits.hpp
/// @brief Multi-rate residual forecasting with hierarchical linear interpolation.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Specify the pooling width and forecast resolution of one N-HiTS stack.
/// @par Example
/// @code
/// const dm::nn::NHitsScale coarse{6, 2};
/// std::println("{}", coarse.knots); // 2
/// @endcode
/// @see nhits()
struct NHitsScale {
  std::int64_t pool_width; ///< Positive nonoverlapping pooling width that divides the lookback.
  std::int64_t knots; ///< Positive number of forecast knots, at most the horizon.
};

/// @brief Trace the residual history and the learned forecast at one resolution.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto result = dm::nn::nhits(p, "series", dm::nn::ones({1, 24}), 6, {{6, 2}});
/// std::println("{}", result.stacks.front().pooled.size(1)); // 4
/// @endcode
/// @see nhits()
struct NHitsStack {
  Tensor residual; ///< Input to the stack [batch, lookback].
  Tensor pooled; ///< Max-pooled residual [batch, lookback / pool_width].
  Tensor knots; ///< Learned coefficients [batch, knots].
  Tensor forecast; ///< Linearly interpolated contribution [batch, horizon].
  Tensor backcast; ///< Removed history [batch, lookback], undefined for the final stack.
};

/// @brief Additive forecast and ordered N-HiTS stack traces.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto result = dm::nn::nhits(p, "series", dm::nn::ones({1, 24}), 6, {{6, 2}});
/// std::println("{}", result.forecast.size(1)); // 6
/// @endcode
/// @see nhits()
struct NHitsOutput {
  Tensor forecast; ///< Sum of stack contributions [batch, horizon].
  std::vector<NHitsStack> stacks; ///< Traces in the supplied scale order.
};

/// @brief Fit a direct forecast from residual stacks at multiple resolutions.
///
/// Each stack pools the current residual, applies two ReLU layers, and
/// predicts uniformly spaced knots. Linear interpolation includes both
/// endpoints. A single knot gives a constant contribution. Nonfinal stacks
/// remove a learned full-resolution backcast from the residual.
/// @param store CPU parameter store matching the history dtype.
/// @param name Parameter prefix for this architecture.
/// @param history CPU Float32 or Float64 matrix [batch, lookback].
/// @param horizon Positive forecast length.
/// @param scales Nonempty list of pooling widths and knot counts in stack order.
/// @param width Positive hidden layer width, default 32.
/// @return Differentiable forecasts and stack traces.
/// @throws std::invalid_argument Invalid shapes, scales, dtype, or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto result = dm::nn::nhits(p, "series", dm::nn::ones({1, 24}), 6, {{6, 2}, {1, 6}});
/// std::println("{}", result.stacks.size()); // 2
/// @endcode
/// @par References
/// - Challu et al., N-HiTS, https://arxiv.org/abs/2201.12886
/// @see dense(), max()
[[nodiscard]] inline NHitsOutput nhits(
    ParameterStore& store, const std::string& name, const Tensor& history,
    std::int64_t horizon, const std::vector<NHitsScale>& scales, std::int64_t width = 32) {
  if (!history.defined() || history.dim() != 2 || history.size(0) < 1 || history.size(1) < 1 ||
      history.device().backend != compute::Backend::CPU || history.dtype() != store.dtype() ||
      horizon < 1 || width < 1 || scales.empty())
    throw std::invalid_argument("nhits: invalid history or architecture");
  const auto rows = history.size(0), lookback = history.size(1);
  for (const auto& [pool, knots] : scales)
    if (pool < 1 || lookback % pool != 0 || knots < 1 || knots > horizon)
      throw std::invalid_argument("nhits: invalid pool width or knot count");

  NHitsOutput result{zeros({rows, horizon}, history.dtype()), {}};
  auto residual = history;
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const auto [pool, knots] = scales[i];
    const auto prefix = name + "/stack" + std::to_string(i);
    const auto pooled = max(residual.reshape({rows, lookback / pool, pool}), {2});
    const auto hidden = dense(store, prefix + "/fc1", pooled, width, Activation::Relu);
    const auto features = dense(store, prefix + "/fc2", hidden, width, Activation::Relu);
    const auto coefficients = dense(store, prefix + "/knots", features, knots);
    auto basis = zeros({knots, horizon}, history.dtype());
    /// @brief Fill the linear interpolation matrix in the computation dtype.
    const auto interpolate = [&]<class T>(T* weights) {
      for (std::int64_t lead = 0; lead < horizon; ++lead) {
        const auto position = double(lead) * (knots - 1) / std::max<std::int64_t>(horizon - 1, 1);
        const auto left = static_cast<std::int64_t>(position);
        const auto right = std::min(left + 1, knots - 1);
        weights[left * horizon + lead] = static_cast<T>(1 - (position - left));
        weights[right * horizon + lead] += static_cast<T>(position - left);
      }
    };
    if (history.dtype() == DType::Float32)
      interpolate(basis.data_ptr<float>());
    else
      interpolate(basis.data_ptr<double>());
    const auto forecast = matmul(coefficients, basis);
    result.forecast = result.forecast + forecast;
    result.stacks.push_back({residual, pooled, coefficients, forecast, {}});
    if (i + 1 < scales.size()) {
      result.stacks.back().backcast = dense(store, prefix + "/backcast", features, lookback);
      residual = residual - result.stacks.back().backcast;
    }
  }
  return result;
}

}  // namespace datamunge::nn
