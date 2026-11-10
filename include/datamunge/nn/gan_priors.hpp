#pragma once
/// @file gan_priors.hpp
/// @brief Explicit locally seeded Box-Muller rejection priors with complete proposal audits.
#include <cmath>
#include <limits>
#include <random>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Own accepted Gaussian coordinates and every tested Double proposal in stream order.
///
/// z[rows,features] has requested Float32/64 dtype; proposals[K] is Float64,
/// including rejected values and the last accepted value, without an unused
/// trailing sine. Both fields are detached owning tensors, independent of
/// records/other draws and global RNG. Thresholding precedes final dtype rounding.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto prior=dm::nn::models::GaussianRejectionPrior(2,12,6907,.5);
/// // prior.z is[2,12]; proposals retains all consumed accepted/rejected candidates.
/// @endcode
/// @par References
/// - Box and Muller (1958), A Note on the Generation of Random Normal Deviates,
///   https://doi.org/10.1214/aoms/1177706645 .
/// - Brock et al. (2018/2019), https://arxiv.org/abs/1809.11096 , truncation by rejection.
/// @see GaussianRejectionPrior(), GaussianNoise()
struct GaussianRejectionDraw {
  Tensor z;         ///< Owning detached accepted coordinates[rows,features], one final dtype cast.
  Tensor proposals; ///< Owning detached Float64[K] tested candidates, including rejected draws.
};

/// @brief Draw an explicit locally seeded Gaussian bank by rejecting candidates outside an absolute
/// limit.
///
/// Local std::mt19937(seed) draws midpoint uniforms (word+.5)/2^32 in pairs.
/// Box-Muller emits cosine first, caches sine, and tests each Double candidate
/// against abs(candidate)<=limit; accepted values receive one final Float32/64
/// cast. Default positive infinity performs no rejection. This is rejection,
/// not clipping/rescaling; a finite threshold is applied before output rounding,
/// so an arbitrary nonrepresentable limit may be exceeded slightly by final casting.
/// Every tested candidate is audited, including rejects and the final accepted
/// value; no unused sine is appended. No global generator, fitting or input graph.
/// Positive rows/features, Float32/64 dtype and positive non-NaN limit required.
/// The source work guard requires tested proposals <1000*rows*features, including
/// after the final acceptance; exhausting it throws without returning a partial
/// bank. Counts and products must fit int64. Work/storage O(K+rows*features),
/// K below that bound. Owning outputs are detached; concurrent calls are independent.
/// @param rows Positive code-bank row count.
/// @param features Positive coordinate width,12 for the compact BigGAN.
/// @param seed Explicit uint32 local MT seed, with no other stream affected.
/// @param limit Positive absolute Double threshold, default infinity.
/// @param dtype Accepted-code dtype Float32(default) or Float64; audit always Float64.
/// @return Owning GaussianRejectionDraw with accepted z and full tested proposal stream.
/// @throws std::invalid_argument For invalid dimensions/products, limit or dtype.
/// @throws std::runtime_error When the strict source proposal budget is exhausted.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto a=dm::nn::models::GaussianRejectionPrior(3,12,6907,1.);
/// const auto b=dm::nn::models::GaussianRejectionPrior(3,12,6907,1.);
/// // a/b z and proposal fields are bit exact, without another RNG stream consumed.
/// @endcode
/// @par References
/// - Box and Muller (1958), https://doi.org/10.1214/aoms/1177706645 .
/// - Brock, Donahue and Simonyan (2018/2019), Large Scale GAN Training for High
///   Fidelity Natural Image Synthesis, https://arxiv.org/abs/1809.11096 .
///   Midpoint uniforms, MT order, final casting and work guard follow this source recipe.
/// @see GaussianRejectionDraw, GaussianNoise(), BigGanGenerate()
inline GaussianRejectionDraw GaussianRejectionPrior(
    std::int64_t  rows,
    std::int64_t  features,
    std::uint32_t seed,
    double        limit = std::numeric_limits<double>::infinity(),
    DType         dtype = DType::Float32) {
  const auto top = std::numeric_limits<std::int64_t>::max();
  if (rows <= 0 || features <= 0 || rows > top / features || rows * features > top / 1000 ||
      !(limit > 0) || std::isnan(limit) || (dtype != DType::Float32 && dtype != DType::Float64))
    throw std::invalid_argument("Invalid Gaussian rejection bank geometry, threshold or dtype");
  const auto          count = rows * features, budget = count * 1000;
  std::mt19937        rng(seed);
  std::vector<double> proposals, accepted;
  accepted.reserve(static_cast<std::size_t>(count));
  bool   spare  = false;
  double cached = 0;
  while (accepted.size() < static_cast<std::size_t>(count)) {
    double candidate;
    if (spare) {
      candidate = cached;
      spare     = false;
    } else {
      const double u = (double(rng()) + .5) / 4294967296., v = (double(rng()) + .5) / 4294967296.;
      const double radius = std::sqrt(-2 * std::log(u)),
                   angle  = 6.283185307179586476925286766559 * v;
      candidate           = radius * std::cos(angle);
      cached              = radius * std::sin(angle);
      spare               = true;
    }
    proposals.push_back(candidate);
    if (std::abs(candidate) <= limit)
      accepted.push_back(candidate);
    if (proposals.size() >= static_cast<std::size_t>(budget))
      throw std::runtime_error("Gaussian rejection proposal budget exhausted");
  }
  return {
      cast(from_vector(accepted, {rows, features}), dtype),
      from_vector(proposals, {static_cast<std::int64_t>(proposals.size())})};
}
} // namespace datamunge::nn::models
