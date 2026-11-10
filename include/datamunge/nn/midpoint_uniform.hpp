#pragma once
/// @file midpoint_uniform.hpp
/// @brief Locally seeded raw-word midpoint uniform banks, independent of global RNG state.
#include <limits>
#include <random>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {
/// @brief Draw a deterministic local-MT19937 midpoint uniform matrix with one final dtype cast.
///
/// Each coordinate consumes one std::mt19937(seed) word and computes
/// (double(word)+.5)/2^32 in row-major order, then casts once to Float32(default)
/// or Float64. Double values lie strictly inside(0,1); Float32 rounding can
/// produce1 at the upper boundary. No rejection, clipping, global stream,
/// parameter store, fitting or retained RNG state. Positive rows/columns with
/// representable product required. Owning contiguous detached CPU[rows,columns];
/// repeat calls with equal arguments repeat bits and concurrent calls are independent.
/// Work/storage O(rows*columns). Standard allocation failures may propagate.
/// @param rows Positive output row count.
/// @param columns Positive coordinate width, default1 for interpolation weights.
/// @param seed Explicit uint32 seed, default6601; no other stream is consumed.
/// @param dtype Float32(default) or Float64 output, after Double midpoint arithmetic.
/// @return Owning detached CPU uniform matrix in the requested dtype.
/// @throws std::invalid_argument For nonpositive/overflowing dimensions or unsupported dtype.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto a=dm::nn::MidpointUniform(3,1,6601);
/// const auto b=dm::nn::MidpointUniform(3,1,6601);
/// // a/b are bit exact; Float64 coordinates remain strictly inside(0,1).
/// @endcode
/// @par References
/// - Matsumoto and Nishimura (1998), Mersenne Twister, https://doi.org/10.1145/272991.272995 .
///   The migrated WGAN recipe specifies the raw-word midpoint transform and final cast.
/// @see uniform(), models::GaussianNoise(), models::WassersteinGan()
inline Tensor MidpointUniform(
    std::int64_t  rows,
    std::int64_t  columns = 1,
    std::uint32_t seed    = 6601,
    DType         dtype   = DType::Float32) {
  if (rows <= 0 || columns <= 0 || rows > std::numeric_limits<std::int64_t>::max() / columns ||
      (dtype != DType::Float32 && dtype != DType::Float64))
    throw std::invalid_argument("Invalid midpoint uniform geometry or dtype");
  std::mt19937        rng(seed);
  std::vector<double> values(static_cast<std::size_t>(rows * columns));
  for (auto& value : values)
    value = (double(rng()) + .5) / 4294967296.;
  return cast(from_vector(values, {rows, columns}), dtype);
}
} // namespace datamunge::nn
