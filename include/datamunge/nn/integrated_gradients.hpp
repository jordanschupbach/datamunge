#pragma once
/// @file integrated_gradients.hpp
/// @brief Tensor straight-line midpoint integration of fixed-target ordinary derivatives.
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own midpoint integrated attributions,mean derivatives and every evaluated path coordinate.
///
/// Owning tensors retain all sampled derivatives in step-major/case-minor order.
/// No learned state or optimizer;baseline and fixed target choices define the quantity.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 1, 2, 1}), base = n::zeros(x.shape());
/// const auto out = m::IntegratedGradients(x, base, n::zeros({2}, n::DType::Int32), 16, 8,
///                                         [](const n::Tensor &points, const n::Tensor &) {
///                                           return n::ones(points.shape());
///                                         });
/// @endcode
/// @par References
/// - Sundararajan,Taly and Yan(2017),Axiomatic Attribution for Deep Networks, https://arxiv.org/abs/1703.01365. This implementation uses straight-line MIDPOINT quadrature,fixed explicit classes,Float64path arithmetic followed byFloat32model evaluation. Finite-node completeness is approximate at nonlinear/piecewise-linear boundaries and depends on the chosen baseline.
/// @see IntegratedGradients(),VanillaInputGradient()
struct IntegratedResult {
  Tensor attribution;   ///< Float32input-shaped(x-baseline)*mean path derivative.
  Tensor mean_gradient; ///< Float32input-shaped mean of all sampled derivatives.
  Tensor path_gradient; ///< Float32[steps,N,...]all actual midpoint input derivatives.
  Tensor nodes;         ///< Float64[steps]nodes(k+.5)/steps.
  int steps = 0;        ///< Positive actual quadrature count,0for an unpopulated result.
};
/// @brief Represent an ordinary fixed-target input derivative callable.
///
/// Callable(Float32[count,...]points,Int32[count]fixedclasses)->Float32same-shape
/// ordinary derivatives. Rows must be independent;use SUM selected scores,not
/// batchmean scaling. Callable captures own model lifetime;must not update weights.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::InputGradientFunction gradient = [](const n::Tensor &points, const n::Tensor &) {
///   return n::ones(points.shape());
/// };
/// @endcode
/// @par References
/// - Sundararajan,Taly and Yan(2017),Axiomatic Attribution for Deep Networks, https://arxiv.org/abs/1703.01365. This implementation uses straight-line MIDPOINT quadrature,fixed explicit classes,Float64path arithmetic followed byFloat32model evaluation. Finite-node completeness is approximate at nonlinear/piecewise-linear boundaries and depends on the chosen baseline.
/// @see IntegratedGradients(),VanillaInputGradient()
using InputGradientFunction = std::function<Tensor(const Tensor &, const Tensor &)>;
/// @brief Integrate fixed-target ordinary derivatives along the baseline-to-input straight line.
///
/// All path arithmetic uses Float64tensors,thenFloat32model evaluation.
/// Point(k,b)=Float32(baseline_b+alpha_k*(input_b-baseline_b));case IDs/
/// callback chunk bounds are metadata only. Accumulate derivative means inDouble;
/// round mean/attribution toFloat32only at output. Discrete target IDs stay fixed,
/// source input/baseline and model weights unchanged. Keeps allsteps*Nderivatives;
/// quadrature and baseline dependence remain explicit,no optimizer hidden.
///
/// @param input Nonempty Float32[N,...]observed input,batch rank>=2.
/// @param baseline Matching Float32input-shaped baseline observations.
/// @param targets Int32[N]fixed explicit categories,passed unchanged to callback.
/// @param steps Positive midpoint quadrature count.
/// @param batch Positive maximum path-point callback batch.
/// @param gradient Callable returning ordinary Float32input derivatives for independent batch rows.
/// @return Attributions,mean derivatives,all step-major path derivatives and exact Double midpoint nodes.
/// @throws std::invalid_argument For invalid dtype/rank/shape/empty batches/steps,nonfinite data,missing callback or invalid derivative outputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 1, 2, 1}), base = n::zeros(x.shape());
/// const auto out = m::IntegratedGradients(x, base, n::zeros({2}, n::DType::Int32), 16, 8,
///                                         [](const n::Tensor &points, const n::Tensor &) {
///                                           return n::ones(points.shape());
///                                         });
/// @endcode
/// @par References
/// - Sundararajan,Taly and Yan(2017),Axiomatic Attribution for Deep Networks, https://arxiv.org/abs/1703.01365. This implementation uses straight-line MIDPOINT quadrature,fixed explicit classes,Float64path arithmetic followed byFloat32model evaluation. Finite-node completeness is approximate at nonlinear/piecewise-linear boundaries and depends on the chosen baseline.
/// @see IntegratedGradients(),VanillaInputGradient()
inline IntegratedResult IntegratedGradients(const Tensor &input, const Tensor &baseline,
                                            const Tensor &targets, int steps, int batch,
                                            const InputGradientFunction &gradient) {
  if (input.dtype() != nn::DType::Float32 || baseline.dtype() != input.dtype() ||
      input.shape() != baseline.shape() || input.dim() < 2 || input.size(0) < 1 ||
      input.numel() < 1 || targets.dtype() != nn::DType::Int32 ||
      targets.shape() != nn::Shape{input.size(0)} || steps < 1 || batch < 1 || !gradient ||
      !nn::all(nn::is_finite(input)).item<bool>() ||
      !nn::all(nn::is_finite(baseline)).item<bool>())
    throw std::invalid_argument("matching finite Float32input/baseline,fixedInt32targets and "
                                "positive quadrature config required");
  const auto count = input.size(0);
  const auto nodes = (nn::arange(0, steps, 1, nn::DType::Float64) + .5) / double(steps);
  nn::Shape alpha_shape(input.dim() + 1, 1);
  alpha_shape[0] = steps;
  const auto first = nn::expand_dims(baseline.to(nn::DType::Float64), 0),
             difference = nn::expand_dims(
                 input.to(nn::DType::Float64) - baseline.to(nn::DType::Float64), 0);
  auto shape = input.shape();
  shape[0] *= steps;
  const auto points =
      (first + nodes.reshape(alpha_shape) * difference).to(nn::DType::Float32).reshape(shape);
  const auto classes = nn::tile(targets, {steps});
  std::vector<Tensor> parts;
  for (std::int64_t start = 0; start < shape[0]; start += batch) {
    const auto end = std::min<std::int64_t>(start + batch, shape[0]);
    const auto x = nn::slice_axis(points, 0, start, end);
    const auto g = gradient(x, nn::slice_axis(classes, 0, start, end));
    if (g.shape() != x.shape() || g.dtype() != nn::DType::Float32 ||
        !nn::all(nn::is_finite(g)).item<bool>())
      throw std::invalid_argument(
          "finite aligned ordinary derivative required at every path point");
    parts.push_back(g.detach());
  }
  auto trace_shape = input.shape();
  trace_shape.insert(trace_shape.begin(), steps);
  const auto trace = nn::concat(parts, 0).reshape(trace_shape);
  const auto average = nn::mean(trace.to(nn::DType::Float64), {0});
  return {((input.to(nn::DType::Float64) - baseline.to(nn::DType::Float64)) * average)
              .to(nn::DType::Float32),
          average.to(nn::DType::Float32), trace, nodes, steps};
}
} // namespace datamunge::nn::models
