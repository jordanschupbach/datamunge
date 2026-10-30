#pragma once
/// @file dream.hpp
/// @brief Tensor half-pixel octaves and bounded RMS-normalized input optimization.
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/params.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Resize a Float32NHWC octave with explicit Double tensor half-pixel interpolation.
///
/// Coordinates=(destination+.5)*source/target-.5,clamped0..source-1.
/// All coordinate/floor/weight/gather/interpolation arithmetic usesDoubletensors,
/// thenFloat32output. This deliberately differs from image_ops Float32coordinate
/// plans so the retained source Double octave convention is reproduced.
/// The interpolation keeps its live input graph;no learned preprocessing.
///
/// @param x Nonempty finite Float32[N,H,W,C]image tensor.
/// @param height Positive target height.
/// @param width Positive target width.
/// @return Float32[N,height,width,C]half-pixel bilinear image,clamped source-coordinate borders.
/// @throws std::invalid_argument For invalid rank/dtype/empty axes,nonfinite image or nonpositive target sizes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::DreamResize(n::ones({1, 2, 2, 1}), 3, 3);
/// @endcode
/// @par References
/// - Mordvintsev,Olah and Tyka(2015),Inceptionism:Going Deeper into Neural Networks, https://research.google/blog/inceptionism-going-deeper-into-neural-networks/. Repository teaching variant uses frozen MNIST feature means,RMS-normalized projected TensorSGD input steps and monotonic12trial backtracking,not an ImageNet Inception dream or classifier retraining.
/// @see DreamAscent(),DreamResize()
inline Tensor DreamResize(const Tensor &x, int height, int width) {
  if (x.dtype() != nn::DType::Float32 || x.dim() != 4 || x.numel() < 1 || height < 1 ||
      width < 1 || !nn::all(nn::is_finite(x)).item<bool>())
    throw std::invalid_argument("finiteFloat32NHWCoctave and positive target required");
  const auto fy = nn::clip((nn::arange(0, height, 1, nn::DType::Float64) + .5) *
                                   double(x.size(1)) / double(height) -
                               .5,
                           0., double(x.size(1) - 1)),
             fx = nn::clip((nn::arange(0, width, 1, nn::DType::Float64) + .5) *
                                   double(x.size(2)) / double(width) -
                               .5,
                           0., double(x.size(2) - 1));
  const auto y0 = nn::floor(fy).to(nn::DType::Int64), x0 = nn::floor(fx).to(nn::DType::Int64),
             y1 = nn::minimum(y0 + 1, nn::full(y0.shape(), x.size(1) - 1, nn::DType::Int64)),
             x1 = nn::minimum(x0 + 1, nn::full(x0.shape(), x.size(2) - 1, nn::DType::Int64));
  const auto wide = x.to(nn::DType::Float64), tl = nn::gather(nn::gather(wide, y0, 1), x0, 2),
             tr = nn::gather(nn::gather(wide, y0, 1), x1, 2),
             bl = nn::gather(nn::gather(wide, y1, 1), x0, 2),
             br = nn::gather(nn::gather(wide, y1, 1), x1, 2);
  const auto dy = (fy - y0.to(nn::DType::Float64)).reshape({1, height, 1, 1}),
             dx = (fx - x0.to(nn::DType::Float64)).reshape({1, 1, width, 1});
  return ((1. - dy) * ((1. - dx) * tl + dx * tr) + dy * ((1. - dx) * bl + dx * br))
      .to(nn::DType::Float32);
}
/// @brief Own one scalar feature objective and its ordinary Float32input derivative.
///
/// Scalar score is line-search control metadata;the image derivative remains
/// a tensor. The callback must use the same objective for every candidate.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto out = m::DreamAscent(n::full({1, 2, 2, 1}, .6), .02, [](const n::Tensor &x) {
///   auto p = x.detach().clone().requires_grad_();
///   const auto score = -n::mean(n::square(p - .7));
///   return m::DreamEvaluation{score.item(), n::grad(score, {p})[0].detach()};
/// });
/// @endcode
/// @par References
/// - Mordvintsev,Olah and Tyka(2015),Inceptionism:Going Deeper into Neural Networks, https://research.google/blog/inceptionism-going-deeper-into-neural-networks/. Repository teaching variant uses frozen MNIST feature means,RMS-normalized projected TensorSGD input steps and monotonic12trial backtracking,not an ImageNet Inception dream or classifier retraining.
/// @see DreamAscent(),DreamResize()
struct DreamEvaluation {
  double score;    ///< Finite current scalar feature objective,presentation/control only.
  Tensor gradient; ///< Float32input-shaped ordinary objective derivative.
};
/// @brief Represent an ordinary frozen-model feature objective and derivative callback.
///
/// Callable(Float32image)->DreamEvaluation. Captures own model lifetime,
/// performs no learned-weight update,and evaluates all line-search candidates
/// with exactly the same objective. A zero/no-op callback is valid.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::DreamFunction f = [](const n::Tensor &x) {
///   return m::DreamEvaluation{n::mean(x).item(), n::ones(x.shape()) / double(x.numel())};
/// };
/// @endcode
/// @par References
/// - Mordvintsev,Olah and Tyka(2015),Inceptionism:Going Deeper into Neural Networks, https://research.google/blog/inceptionism-going-deeper-into-neural-networks/. Repository teaching variant uses frozen MNIST feature means,RMS-normalized projected TensorSGD input steps and monotonic12trial backtracking,not an ImageNet Inception dream or classifier retraining.
/// @see DreamAscent(),DreamResize()
using DreamFunction = std::function<DreamEvaluation(const Tensor &)>;
/// @brief Own one bounded input-ascent result and its explicit line-search diagnostics.
///
/// The returned image is a Float32copy/proposal;original image and frozen model
/// remain unchanged. Failed line search is an explicit unchanged-image no-op.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto out = m::DreamAscent(n::full({1, 2, 2, 1}, .6), .02, [](const n::Tensor &x) {
///   auto p = x.detach().clone().requires_grad_();
///   const auto score = -n::mean(n::square(p - .7));
///   return m::DreamEvaluation{score.item(), n::grad(score, {p})[0].detach()};
/// });
/// @endcode
/// @par References
/// - Mordvintsev,Olah and Tyka(2015),Inceptionism:Going Deeper into Neural Networks, https://research.google/blog/inceptionism-going-deeper-into-neural-networks/. Repository teaching variant uses frozen MNIST feature means,RMS-normalized projected TensorSGD input steps and monotonic12trial backtracking,not an ImageNet Inception dream or classifier retraining.
/// @see DreamAscent(),DreamResize()
struct DreamStep {
  Tensor image;       ///< Accepted Float32input image,or unchanged copy if no step accepted.
  Tensor gradient;    ///< Actual initial ordinary objective derivative.
  double before = 0;  ///< Initial scalar objective.
  double after = 0;   ///< Accepted scalar objective,or before for a no-op.
  double rms = 0;     ///< Double tensor RMS of all initial gradient coordinates.
  double step = 0;    ///< Accepted nonnegative learning rate,0for no-op.
  int backtracks = 0; ///< Halving count0..11for accepted step,12for exhausted search.
};
/// @brief Take one RMS-normalized projected TensorSGD input step with explicit monotonic backtracking.
///
/// Doubletensor RMS,one Doubleimage leaf in a local ParameterStore,plain
/// TensorSGD on negative normalized ascent derivative. Try at most12rates,
/// halving after rejection;each proposal clips to[0,1]then roundsFloat32.
/// Restore the original Doubleleaf before each trial. Accept score>=before,
/// otherwise explicitno-op;zeroRMS/rate no division. This is one inspectable
/// step helper,not hidden classifier fitting;reports show the image optimizer
/// and full octave/iteration/line-search loops explicitly.
///
/// @param image Finite Float32input image in[0,1],unchanged by this function.
/// @param learning_rate Finite nonnegative initial rate,zero returns unchanged copy.
/// @param evaluate Fixed ordinary objective/derivative callback,does not update frozen weights.
/// @return Accepted candidate/gradient/scores/RMS/rate/backtracks,or unchanged no-op.
/// @throws std::invalid_argument For invalid rate/image/bounds/callback score/gradient or nonfinite candidate score.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto out = m::DreamAscent(n::full({1, 2, 2, 1}, .6), .02, [](const n::Tensor &x) {
///   auto p = x.detach().clone().requires_grad_();
///   const auto score = -n::mean(n::square(p - .7));
///   return m::DreamEvaluation{score.item(), n::grad(score, {p})[0].detach()};
/// });
/// @endcode
/// @par References
/// - Mordvintsev,Olah and Tyka(2015),Inceptionism:Going Deeper into Neural Networks, https://research.google/blog/inceptionism-going-deeper-into-neural-networks/. Repository teaching variant uses frozen MNIST feature means,RMS-normalized projected TensorSGD input steps and monotonic12trial backtracking,not an ImageNet Inception dream or classifier retraining.
/// @see DreamAscent(),DreamResize()
inline DreamStep DreamAscent(const Tensor &image, double learning_rate,
                             const DreamFunction &evaluate) {
  if (!std::isfinite(learning_rate) || learning_rate < 0 ||
      image.dtype() != nn::DType::Float32 || image.numel() < 1 || !evaluate ||
      !nn::all(nn::is_finite(image)).item<bool>() || nn::min(image).item() < 0 ||
      nn::max(image).item() > 1)
    throw std::invalid_argument("finite boundedFloat32input and nonnegative rate required");
  const auto e = evaluate(image);
  if (!std::isfinite(e.score) || e.gradient.dtype() != nn::DType::Float32 ||
      e.gradient.shape() != image.shape() || !nn::all(nn::is_finite(e.gradient)).item<bool>())
    throw std::invalid_argument("finite alignedordinaryfeaturederivative required");
  DreamStep out;
  out.image = image.clone();
  out.gradient = e.gradient;
  out.before = out.after = e.score;
  const auto wide = e.gradient.to(nn::DType::Float64),
             rms = nn::sqrt(nn::mean(nn::square(wide)));
  out.rms = rms.item();
  if (learning_rate == 0 || out.rms == 0)
    return out;
  nn::ParameterStore input(0, nn::DType::Float64);
  input.get("image", image.shape(), nn::init::zeros());
  input.set("image", image.to(nn::DType::Float64));
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer sgd(input, learning_rate, config);
  double rate = learning_rate;
  const auto direction = -wide / (rms + 1e-12);
  for (int attempt = 0; attempt < 12; ++attempt) {
    input.set("image", image.to(nn::DType::Float64));
    sgd.set_learning_rate(rate);
    sgd.step({direction});
    const auto candidate = nn::clip(input.at("image"), 0., 1.).to(nn::DType::Float32);
    const double score = evaluate(candidate).score;
    if (!std::isfinite(score))
      throw std::invalid_argument("finitecandidate objective required");
    if (score >= e.score) {
      out.image = candidate;
      out.after = score;
      out.step = rate;
      out.backtracks = attempt;
      return out;
    }
    rate *= .5;
  }
  out.backtracks = 12;
  return out;
}
} // namespace datamunge::nn::models
