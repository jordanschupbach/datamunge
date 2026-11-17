#pragma once
/// @file continuous_actions.hpp
/// @brief Source-rounded action-box scaling and caller-stream Gaussian noise packing.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/tensor.hpp>
#include <datamunge/rl/environments/interface.hpp>
#include <random>
/// @brief Explicit action conversion and caller-owned noise streams for native continuous control.
namespace datamunge::rl::continuous {

/// @brief Clamp normalized actions to[-1,1] and apply the source Float32 affine box map.
///
/// This transforms an action value only, not its log-density. No environment step or RNG is
/// invoked.
///
/// @param unit Finite normalized actions, one per continuous dimension; out-of-range entries are
/// clamped.
/// @param space Continuous action descriptor with matching dim/low/high sizes and finite increasing
/// bounds.
/// @return Owning native Action with low+(clamped_unit+1)*.5*(high-low); inputs unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto space=e::ActionSpace::Continuous({-2},{2});auto
/// action=dm::rl::continuous::ScaleAction({.5},space);
/// @endcode
/// @par References
/// - Kapturowski et al. (2019), Recurrent Experience Replay,
/// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see NormalNoise()
inline environments::Action ScaleAction(
    const std::vector<float>&        unit,
    const environments::ActionSpace& space) {
  if (space.kind != environments::ActionSpace::kContinuous || space.dim < 1 ||
      unit.size() != size_t(space.dim) || space.low.size() != unit.size() ||
      space.high.size() != unit.size())
    throw std::invalid_argument("aligned continuous action box required");
  for (size_t i = 0; i < unit.size(); ++i)
    if (!std::isfinite(unit[i]) || !std::isfinite(space.low[i]) || !std::isfinite(space.high[i]) ||
        space.high[i] <= space.low[i] || !std::isfinite(space.high[i] - space.low[i]))
      throw std::invalid_argument("finite increasing action box and finite unit actions required");
  std::vector<float> a(unit.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const float u = std::max(-1.f, std::min(1.f, unit[i]));
    a[i]          = space.low[i] + (u + 1) * 0.5f * (space.high[i] - space.low[i]);
  }
  return environments::Action(a);
}

/// @brief Pack a source-order Float32 standard-normal sample matrix from a caller engine.
///
/// Distribution caching is local to this call, exactly as in the source; no RNG is stored.
/// An odd element count may discard a cached paired draw when the distribution is destroyed.
///
/// @param rows Positive row count at most1million.
/// @param cols Positive column count at most1million; product at most64million.
/// @param rng Caller MT19937-64, advanced in row-major order by one fresh
/// normal_distribution<float> per call.
/// @return Owning Float32 CPU[rows,cols], no autograd path to host random draws.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// std::mt19937_64 rng(0);auto noise=dm::rl::continuous::NormalNoise(2,1,rng);
/// @endcode
/// @par References
/// - Kapturowski et al. (2019), Recurrent Experience Replay,
/// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see ScaleAction()
inline nn::Tensor NormalNoise(int rows, int cols, std::mt19937_64& rng) {
  if (rows < 1 || cols < 1 || rows > 1000000 || cols > 1000000 || int64_t(rows) * cols > 64000000)
    throw std::invalid_argument("positive bounded noise geometry required");
  std::vector<float>              v(static_cast<std::size_t>(rows) * cols);
  std::normal_distribution<float> n(0.f, 1.f);
  for (float& x : v)
    x = n(rng);
  return nn::from_vector(v, {rows, cols});
}

} // namespace datamunge::rl::continuous
