#pragma once

/// @file volume_rendering.hpp
/// @brief Differentiable emission-absorption compositing along sampled rays.

#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace datamunge::nn {

/// @brief Composites emission and absorption from front to back along each ray.
///
/// A sample of density sigma and length step transmits exp(-sigma * step)
/// of the incident light. The rest contributes its color to the composite.
/// The terminal transmittance multiplies the uniform background. Use a
/// nonnegative density, for example the output of softplus. Gradients flow
/// through both density and color. Float32 and Float64 CPU tensors are supported.
/// @param density Nonempty tensor [rays, samples, 1] of nonnegative densities.
/// @param color Tensor [rays, samples, channels] with the same floating dtype.
/// @param step Positive finite distance represented by each sample.
/// @param background Finite background intensity for each channel, default zero.
/// @return Pair of rendered colors [rays, channels] and terminal transmittance [rays, 1].
/// @throws std::invalid_argument Incompatible tensors or invalid step or background.
///
/// @par Example
/// @code
/// const auto density = dm::nn::zeros({1, 3, 1});
/// const auto color = dm::nn::ones({1, 3, 3});
/// const auto [rendered, through] = dm::nn::composite_rays(density, color, 0.5, 0.25);
/// std::println("{} {}", dm::nn::mean(rendered).item(), through.item()); // 0.25 1
/// @endcode
/// @par References
/// - Mildenhall et al. (2020), NeRF, https://arxiv.org/abs/2003.08934, equation 3.
/// @see fourier_features(), softplus()
[[nodiscard]] inline std::pair<Tensor, Tensor> composite_rays(
    const Tensor& density, const Tensor& color, double step, double background = 0.) {
  if (!density.defined() || !color.defined() || density.dim() != 3 ||
      color.dim() != 3 || density.numel() == 0 || color.numel() == 0 ||
      density.size(2) != 1 || density.size(0) != color.size(0) ||
      density.size(1) != color.size(1) || density.dtype() != color.dtype() ||
      (density.dtype() != DType::Float32 && density.dtype() != DType::Float64) ||
      density.device().backend != compute::Backend::CPU ||
      color.device().backend != compute::Backend::CPU ||
      !(step > 0.) || !std::isfinite(step) || !std::isfinite(background))
    throw std::invalid_argument("composite_rays: incompatible CPU tensors or invalid rendering parameters");

  const auto optical_depth = density * (-step);
  const auto survival = exp(optical_depth);
  const auto opacity = -expm1(optical_depth);
  auto through = full({density.size(0), 1}, 1., density.dtype(), density.context());
  auto rendered = zeros({color.size(0), color.size(2)}, color.dtype(), color.context());
  for (std::int64_t i = 0; i < density.size(1); ++i) {
    rendered = rendered + through * select(opacity, 1, i) * select(color, 1, i);
    through = through * select(survival, 1, i);
  }
  return {rendered + through * background, through};
}

} // namespace datamunge::nn
