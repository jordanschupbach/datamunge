#pragma once
/// @file gan_data.hpp
/// @brief Explicit ordered Gaussian-ring samples for small generative-model experiments.
#include "datamunge/nn/autoencoder_models.hpp"
#include "datamunge/nn/gan_models.hpp"

namespace datamunge::nn::models {
/// @brief Generate ordered Gaussian components on a two-dimensional ring with explicit noise seed.
///
/// Row i uses component i%components at angle 2*pi*(i%components)/components,
/// then adds independent deviation*GaussianNoise coordinates. Centres, noise and
/// addition are computed in Double, using the source expression order; only the
/// final coordinates cast to dtype. Defaults reproduce the compact BiGAN fitting
/// distribution (eight components, radius 2, standard deviation .08), not a real
/// dataset. Rows cycle deterministically through components; there is no random
/// component assignment or shuffling. Counts need not divide components, so the
/// first rows get any remainder. Zero radius/deviation and one component are valid.
///
/// Requires positive rows/components and representable rows*2, finite nonnegative radius
/// and deviation, and Float32/64 dtype. Every result is finite, owning, contiguous
/// CPU[rows,2], without gradients or retained RNG state; repeated calls with equal
/// arguments repeat bits. No fitting, train/test partition, external assets or
/// shared mutable state. O(rows) work/storage. Standard allocation failures can
/// propagate; finite Double or Float32 overflow is rejected, without clipping.
/// @param rows Positive sample count, with representable rows*2.
/// @param seed Explicit uint32 Gaussian seed, default 5507; no global RNG is consumed.
/// @param components Positive component count, default 8; component i%components.
/// @param radius Finite nonnegative ring radius in coordinate units, default 2.
/// @param deviation Finite nonnegative per-coordinate Gaussian standard deviation, default .08.
/// @param dtype Float32 default, or Float64; only final coordinates are cast.
/// @return Owning detached finite CPU[rows,2] ordered point coordinates.
/// @throws std::invalid_argument For invalid dimensions, options, dtype or computed nonfinite
/// coordinates.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto points=dm::nn::models::GaussianRingPoints(4,5507,4,2.,0.);
/// // Four ordered centres are approximately(2,0),(0,2),(-2,0),(0,-2).
/// @endcode
/// @par References
/// - The migrated compact BiGAN experiment's ordered ring, not a published benchmark.
///   GaussianNoise() supplies the canonical seeded Box-Muller stream.
/// @see GaussianNoise(), BidirectionalGan()
inline Tensor GaussianRingPoints(
    std::int64_t  rows,
    std::uint32_t seed       = 5507,
    std::int64_t  components = 8,
    double        radius     = 2.,
    double        deviation  = .08,
    DType         dtype      = DType::Float32) {
  if (rows <= 0 || rows > std::numeric_limits<std::int64_t>::max() / 2 || components <= 0 ||
      !std::isfinite(radius) || radius < 0 || !std::isfinite(deviation) || deviation < 0 ||
      (dtype != DType::Float32 && dtype != DType::Float64))
    throw std::invalid_argument("Invalid Gaussian ring dimensions/options");
  const auto          noise = GaussianNoise({rows, 2}, 1, seed).to_vector<double>();
  std::vector<double> points(static_cast<std::size_t>(rows * 2));
  for (std::int64_t i = 0; i < rows; ++i)
    for (int j = 0; j < 2; ++j) {
      const double theta = 2 * std::acos(-1.) * double(i % components) / double(components);
      points[i * 2 + j] =
          radius * (j ? std::sin(theta) : std::cos(theta)) + deviation * noise[i * 2 + j];
    }
  auto out = cast(from_vector(points, {rows, 2}), dtype);
  gan_detail::Finite(out);
  return out;
}
} // namespace datamunge::nn::models
