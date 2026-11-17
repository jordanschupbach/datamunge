#pragma once
/// @file tile_coding.hpp
/// @brief Exact source two-dimensional offset tile features, without learned weights.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::rl::foundations {

/// @brief Encode two-dimensional state/action pairs using source offset tile grids.
///
/// The object owns only geometry; no weights, RNG, optimizer or training state. Each action has
/// a disjoint block, and every encoding returns exactly one tile per tiling. Offsets
/// are(1,3)*t/tilings, with source Float32 arithmetic, truncation toward zero, then clamping of
/// integer tile coordinates.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// f::TileCoder coder(8,8,3,-1.2f,.6f,-.07f,.07f);auto ids=coder.Active(-.5f,0.f,1);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see TileCoder::Active()
class TileCoder {
 public:
  /// @brief Construct checked source two-dimensional tile geometry.
  ///
  /// @param tilings Grid count in1..1024.
  /// @param size Tiles per axis in1..1024.
  /// @param actions Action count in1..4096; total feature count must fit positive Int32.
  /// @param lo0 Finite first-coordinate lower endpoint.
  /// @param hi0 Strictly greater finite upper endpoint; difference must be finite.
  /// @param lo1 Finite second-coordinate lower endpoint.
  /// @param hi1 Strictly greater finite upper endpoint; difference must be finite.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::TileCoder coder(8,8,3,-1.2f,.6f,-.07f,.07f);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see num_features()
  TileCoder(int tilings, int size, int actions, float lo0, float hi0, float lo1, float hi1)
      : tilings_(tilings), size_(size), actions_(actions), lo_{lo0, lo1}, hi_{hi0, hi1} {
    if (tilings < 1 || size < 1 || actions < 1 || tilings > 1024 || size > 1024 || actions > 4096 ||
        int64_t(tilings) * size * size * actions > std::numeric_limits<int>::max() ||
        !std::isfinite(lo0) || !std::isfinite(hi0) || !std::isfinite(lo1) || !std::isfinite(hi1) ||
        hi0 <= lo0 || hi1 <= lo1 || !std::isfinite(hi0 - lo0) || !std::isfinite(hi1 - lo1))
      throw std::invalid_argument("tile geometry and finite increasing bounds required");
  }

  /// @brief Return the total number of disjoint action/tile features.
  ///
  /// @return Positive Int32 tilings*size*size*actions.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::TileCoder coder(8,8,3,-1.2f,.6f,-.07f,.07f);auto count=coder.num_features();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Active()
  int num_features() const { return tilings_ * size_ * size_ * actions_; }

  /// @brief Return the number of active features per encoded state/action.
  ///
  /// @return Positive offset-grid count.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::TileCoder coder(8,8,3,-1.2f,.6f,-.07f,.07f);auto count=coder.tilings();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Active()
  int tilings() const { return tilings_; }

  /// @brief Return owning native indices for one state/action encoding.
  ///
  /// Coordinates whose scaled values cannot safely convert to Int32 are rejected. The encoder is
  /// not differentiable in state; gradients of gathered model weights remain ordinary native tensor
  /// paths.
  ///
  /// @param s0 Finite first coordinate; finite outside-box values follow source post-offset integer
  /// clamping.
  /// @param s1 Finite second coordinate, with the same rule.
  /// @param action Action index in0..actions-1.
  /// @return Owning Int64 vector of tilings distinct indices in0..num_features()-1.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::TileCoder coder(8,8,3,-1.2f,.6f,-.07f,.07f);auto ids=coder.Active(-.5f,0.f,2);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see nn::gather()
  std::vector<int64_t> Active(float s0, float s1, int action) const {
    if (!std::isfinite(s0) || !std::isfinite(s1) || action < 0 || action >= actions_)
      throw std::invalid_argument("finite tile state and valid action required");
    std::vector<int64_t> idx(tilings_);
    const float          u0 = (s0 - lo_[0]) / (hi_[0] - lo_[0]) * size_;
    const float          u1 = (s1 - lo_[1]) / (hi_[1] - lo_[1]) * size_;
    if (!std::isfinite(u0) || !std::isfinite(u1) ||
        double(u0) < double(std::numeric_limits<int>::min()) + 4 ||
        double(u1) < double(std::numeric_limits<int>::min()) + 4 ||
        double(u0) > double(std::numeric_limits<int>::max()) - 4 ||
        double(u1) > double(std::numeric_limits<int>::max()) - 4)
      throw std::invalid_argument("tile coordinate exceeds integer range");
    for (int t = 0; t < tilings_; ++t) {
      // Asymmetric offsets (1, 3) * t / tilings, as recommended for 2-D.
      const int c0 = std::min(size_ - 1, std::max(0, static_cast<int>(u0 + 1.f * t / tilings_)));
      const int c1 = std::min(size_ - 1, std::max(0, static_cast<int>(u1 + 3.f * t / tilings_)));
      idx[t]       = ((action * tilings_ + t) * size_ + c0) * size_ + c1;
    }
    return idx;
  }

 private:
  int   tilings_; ///< Number of offset grids.
  int   size_;    ///< Tiles per axis per grid.
  int   actions_; ///< Disjoint action block count.
  float lo_[2];   ///< Finite state-box lower endpoints.
  float hi_[2];   ///< Strictly greater finite upper endpoints.
};

} // namespace datamunge::rl::foundations
