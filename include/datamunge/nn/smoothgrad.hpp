#pragma once
/// @file smoothgrad.hpp
/// @brief Tensor explicit-noise signed gradient estimates and MonteCarlo summaries.
#include "datamunge/nn/integrated_gradients.hpp"
#include <cmath>
#include <random>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own signed SmoothGrad estimates,distinct magnitude summaries and every noisy derivative.
///
/// Owning tensors retain explicit noise,noisy inputs and gradients. These are
/// finite-bank summaries at fixed targets;standard error estimates MonteCarlo
/// uncertainty per coordinate,not explanation correctness.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 3});
/// const auto noise = m::SmoothGradNoise(x.shape(), 8, 1171);
/// const auto result = m::SmoothGrad(x, n::zeros({2}, n::DType::Int32), noise, .1f, false, 4,
///                                   [](const n::Tensor &p, const n::Tensor &) {
///                                     return n::ones(p.shape());
///                                   });
/// @endcode
/// @par References
/// - Smilkov,Thorat,Kim,Viegas and Wattenberg(2017),SmoothGrad:removing noise by adding noise, https://arxiv.org/abs/1706.03825. Signed MEANordinary gradients is the primary estimator;meanabsolute/meansquare are explicitly different summaries. Unclipped and clipped input perturbations define different sampling distributions;finite noise banks need not have zero sample mean.
/// @see SmoothGrad(),VanillaInputGradient()
struct SmoothGradResult {
  Tensor mean;            ///< Float32input-shaped SIGNEDsample mean ordinary gradient.
  Tensor mean_absolute;   ///< Float32mean absolute gradient,different from abs(mean).
  Tensor mean_square;     ///< Float32mean squared gradient,different estimator.
  Tensor sample_variance; ///< Float32unbiasedsample variance,n1returns0.
  Tensor standard_error;  ///< Float32sqrt(sample_variance/samples).
  Tensor standard_noise;  ///< Original Float64[samples,N,...]explicit standardized bank.
  Tensor
      noisy_input; ///< Float32[samples,N,...]actual perturbed inputs,clipped only ifrequested.
  Tensor
      sample_gradient; ///< Float32[samples,N,...]every actual fixed-target ordinary derivative.
};
/// @brief Generate a reproducible explicit Double Gaussian bank using tensor Box-Muller transforms.
///
/// Host RNG draws are ingress metadata;Float64tensors compute midpoint-uniform
/// scaling,log/sqrt/cos/sin Box-Muller pairs. Seeded raw draws follow source
/// (u+.5)/2^32,two successive uniforms per pair;odd final element takes cosine.
/// No global RNG or learned state is consumed.
///
/// @param input_shape Nonempty input shape of rank>=2,including positive batch axis.
/// @param samples Positive number of noisy replicas.
/// @param seed Seed for source-compatible mt19937raw draws.
/// @return Float64[samples,input_shape...]standard-normal bank.
/// @throws std::invalid_argument For empty/nonpositive shape or sample count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto noise = m::SmoothGradNoise({2, 3}, 32, 1171);
/// @endcode
/// @par References
/// - Smilkov,Thorat,Kim,Viegas and Wattenberg(2017),SmoothGrad:removing noise by adding noise, https://arxiv.org/abs/1706.03825. Signed MEANordinary gradients is the primary estimator;meanabsolute/meansquare are explicitly different summaries. Unclipped and clipped input perturbations define different sampling distributions;finite noise banks need not have zero sample mean.
/// @see SmoothGrad(),VanillaInputGradient()
inline Tensor SmoothGradNoise(const nn::Shape &input_shape, int samples, std::uint32_t seed) {
  if (samples < 1 || input_shape.size() < 2 ||
      std::any_of(
          input_shape.begin(), input_shape.end(),
          [](auto x) {
            return x < 1;
          }))
    throw std::invalid_argument("positive noise shape and samples required");
  auto shape = input_shape;
  shape.insert(shape.begin(), samples);
  const auto count = nn::shape_numel(shape), pairs = (count + 1) / 2;
  std::mt19937 generator(seed);
  std::vector<std::int64_t> raw(pairs * 2);
  for (auto &value : raw)
    value = generator();
  const auto uniform =
      (nn::from_vector(raw, {pairs, 2}).to(nn::DType::Float64) + .5) / 4294967296.;
  const auto u = nn::select(uniform, 1, 0),
             angle = nn::select(uniform, 1, 1) * 6.283185307179586476925286766559;
  const auto radius = nn::sqrt(nn::log(u) * -2.);
  const auto bank =
      nn::stack({radius * nn::cos(angle), radius * nn::sin(angle)}, 1).reshape({pairs * 2});
  return nn::slice_axis(bank, 0, 0, count).reshape(shape);
}
/// @brief Estimate signed ordinary fixed-target gradients under an explicit Gaussian perturbation bank.
///
/// Point(k,b)=Float32(Double(input_b)+Double(sigma)*noise_kb),optionally
/// clipped AFTERFloat32rounding. Callback computes ordinary model derivatives at
/// these points,not derivative through clipping. No target reclassification.
/// Doubletensor reductions form mean,meanabs,meansquare and variance,thenFloat32
/// outputs. Variance=max(0,(sum(g*g)-sum(g)^2/samples)/(samples-1));n1returns0.
/// Sigma0recovers the ordinary derivative;finite-bank meannoise need not be0.
/// No model training/optimizer/RNG hidden;all samples are supplied explicitly.
///
/// @param input Nonempty finite Float32[N,...]original observations,rank>=2.
/// @param targets Int32[N]fixed categories,reused at every noisy point.
/// @param standard_noise Finite Float64[samples,N,...]explicit standard-normal bank.
/// @param sigma Finite nonnegative Float32noise scale,source.15default in report.
/// @param clipped Iftrue,clip perturbed Float32inputs to[0,1];false preserves off-domain points.
/// @param batch Positive maximum noisy-point callback batch.
/// @param gradient Ordinary fixed-target independent-row derivative callable.
/// @return Signed mean,distinct magnitude means,unbiased variance,SE,bank,noisy points and all derivatives.
/// @throws std::invalid_argument For invalid dtype/shape/configuration,nonfinite inputs/noise or invalid callback derivatives.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 3});
/// const auto noise = m::SmoothGradNoise(x.shape(), 8, 1171);
/// const auto result = m::SmoothGrad(x, n::zeros({2}, n::DType::Int32), noise, .1f, false, 4,
///                                   [](const n::Tensor &p, const n::Tensor &) {
///                                     return n::ones(p.shape());
///                                   });
/// @endcode
/// @par References
/// - Smilkov,Thorat,Kim,Viegas and Wattenberg(2017),SmoothGrad:removing noise by adding noise, https://arxiv.org/abs/1706.03825. Signed MEANordinary gradients is the primary estimator;meanabsolute/meansquare are explicitly different summaries. Unclipped and clipped input perturbations define different sampling distributions;finite noise banks need not have zero sample mean.
/// @see SmoothGrad(),VanillaInputGradient()
inline SmoothGradResult SmoothGrad(const Tensor &input, const Tensor &targets,
                                   const Tensor &standard_noise, float sigma, bool clipped,
                                   int batch, const InputGradientFunction &gradient) {
  if (input.dtype() != nn::DType::Float32 || input.dim() < 2 || input.numel() < 1 ||
      input.size(0) < 1 || standard_noise.dtype() != nn::DType::Float64 ||
      standard_noise.dim() != input.dim() + 1 || !std::isfinite(sigma) || sigma < 0 ||
      batch < 1 || !gradient || !nn::all(nn::is_finite(input)).item<bool>() ||
      !nn::all(nn::is_finite(standard_noise)).item<bool>())
    throw std::invalid_argument("finite aligned SmoothGrad configuration required");
  const auto samples = standard_noise.size(0);
  auto expected = input.shape();
  expected.insert(expected.begin(), samples);
  if (samples < 1 || standard_noise.shape() != expected ||
      targets.dtype() != nn::DType::Int32 || targets.shape() != nn::Shape{input.size(0)})
    throw std::invalid_argument("explicit noise bank/fixed target geometry required");
  auto noisy =
      (nn::expand_dims(input.to(nn::DType::Float64), 0) + standard_noise * double(sigma))
          .to(nn::DType::Float32);
  if (clipped)
    noisy = nn::clip(noisy, 0., 1.);
  auto flat_shape = input.shape();
  flat_shape[0] *= samples;
  const auto points = noisy.reshape(flat_shape), classes = nn::tile(targets, {samples});
  std::vector<Tensor> parts;
  for (std::int64_t start = 0; start < flat_shape[0]; start += batch) {
    const auto end = std::min<std::int64_t>(start + batch, flat_shape[0]);
    const auto x = nn::slice_axis(points, 0, start, end),
               g = gradient(x, nn::slice_axis(classes, 0, start, end));
    if (g.shape() != x.shape() || g.dtype() != nn::DType::Float32 ||
        !nn::all(nn::is_finite(g)).item<bool>())
      throw std::invalid_argument("finite ordinary derivatives at all noisy points required");
    parts.push_back(g.detach());
  }
  const auto trace = nn::concat(parts, 0).reshape(expected),
             wide = trace.to(nn::DType::Float64);
  const auto sums = nn::sum(wide, {0}), squares = nn::sum(nn::square(wide), {0});
  const auto mean = sums / double(samples), absolute = nn::mean(nn::abs(wide), {0}),
             square = squares / double(samples);
  const auto variance =
      samples > 1
          ? nn::maximum((squares - nn::square(sums) / double(samples)) / double(samples - 1),
                        nn::zeros(input.shape(), nn::DType::Float64))
          : nn::zeros(input.shape(), nn::DType::Float64);
  return {mean.to(nn::DType::Float32),
          absolute.to(nn::DType::Float32),
          square.to(nn::DType::Float32),
          variance.to(nn::DType::Float32),
          nn::sqrt(variance / double(samples)).to(nn::DType::Float32),
          standard_noise,
          noisy,
          trace};
}
} // namespace datamunge::nn::models
