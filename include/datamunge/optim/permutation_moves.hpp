#pragma once

/// @file
/// @brief Feasible 2-opt and 3-opt moves for symmetric permutation tours.

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::optim {

/// @brief Half-open reversal of one segment in a symmetric cycle.
/// @par Example
/// @code
/// const dm::optim::TwoOptMove move{1, 3};
/// @endcode
/// @see sample_two_opt(), apply_two_opt()
struct TwoOptMove {
  std::size_t first; ///< First index of the reversed segment, greater than zero.
  std::size_t last; ///< Index after the reversed segment, at most the tour size.
};

/// @brief Sample two nonadjacent cut edges uniformly in a symmetric cycle.
/// @param size Number of positions in the cycle.
/// @param rng Borrowed random engine, advanced while sampling.
/// @return Feasible reversal or an empty optional for cycles smaller than four.
/// @par Example
/// @code
/// std::mt19937_64 rng(42);
/// const auto move = dm::optim::sample_two_opt(4, rng);
/// std::print("{}\n", move.has_value()); // true
/// @endcode
/// @par References
/// - Croes (1958). https://doi.org/10.1287/opre.6.6.791
/// @see apply_two_opt(), neighborhood_search()
inline std::optional<TwoOptMove> sample_two_opt(std::size_t size, std::mt19937_64& rng) {
  if (size < 4) return {};
  std::uniform_int_distribution<std::size_t> cut(1, size);
  for (;;) {
    auto first = cut(rng), last = cut(rng);
    if (first > last) std::swap(first, last);
    if (last - first >= 2 && last - first <= size - 2) return TwoOptMove{first, last};
  }
}

/// @brief Reverse the selected segment, preserving the city at index zero.
/// @param tour Mutable permutation representing a cycle with implicit closure.
/// @param move Valid cut positions from sample_two_opt() or an equivalent move.
/// @throws std::invalid_argument Invalid or adjacent cuts.
/// @par Example
/// @code
/// std::vector<std::size_t> tour{0, 1, 2, 3};
/// dm::optim::apply_two_opt(tour, {1, 3});
/// std::print("{}\n", tour[1]); // 2
/// @endcode
/// @see sample_two_opt()
inline void apply_two_opt(std::vector<std::size_t>& tour, const TwoOptMove& move) {
  if (tour.size() < 4 || move.first == 0 || move.last > tour.size() ||
      move.first >= move.last || move.last - move.first < 2 || move.last - move.first > tour.size() - 2) {
    throw std::invalid_argument("2-opt requires two nonadjacent edges");
  }
  std::reverse(tour.begin() + move.first, tour.begin() + move.last);
}

/// @brief Reconnection of two adjacent interior segments after three edge cuts.
/// @par Example
/// @code
/// const auto pattern = dm::optim::ThreeOptPattern::exchange;
/// @endcode
/// @see ThreeOptMove
enum class ThreeOptPattern {
  reverse_both,    ///< Reverse each segment in place.
  exchange,        ///< Exchange the two segments.
  exchange_first,  ///< Reverse the first segment and exchange the segments.
  exchange_second  ///< Reverse the second segment and exchange the segments.
};

/// @brief Three nonadjacent cuts and one genuine three-edge reconnection.
/// @par Example
/// @code
/// const dm::optim::ThreeOptMove move{1, 3, 5, dm::optim::ThreeOptPattern::exchange};
/// @endcode
/// @see sample_three_opt(), apply_three_opt()
struct ThreeOptMove {
  std::size_t first; ///< Lower boundary of the first segment.
  std::size_t middle; ///< Boundary shared by the two segments.
  std::size_t last; ///< Upper boundary of the second segment.
  ThreeOptPattern pattern; ///< Segment reconnection rule.
};

/// @brief Sample three pairwise nonadjacent edges and one of four reconnections.
/// @param size Number of positions in the cycle.
/// @param rng Borrowed random engine, advanced while sampling.
/// @return Feasible move or an empty optional for cycles smaller than six.
/// @par Example
/// @code
/// std::mt19937_64 rng(42);
/// const auto move = dm::optim::sample_three_opt(6, rng);
/// std::print("{}\n", move.has_value()); // true
/// @endcode
/// @par References
/// - Lin (1965). https://doi.org/10.1002/j.1538-7305.1965.tb04146.x
/// @see apply_three_opt(), neighborhood_search()
inline std::optional<ThreeOptMove> sample_three_opt(std::size_t size, std::mt19937_64& rng) {
  if (size < 6) return {};
  std::uniform_int_distribution<std::size_t> cut(1, size);
  for (;;) {
    std::array cuts{cut(rng), cut(rng), cut(rng)};
    std::ranges::sort(cuts);
    if (cuts[1] - cuts[0] >= 2 && cuts[2] - cuts[1] >= 2 && cuts[2] - cuts[0] <= size - 2) {
      const auto pattern = std::uniform_int_distribution<int>(0, 3)(rng);
      return ThreeOptMove{cuts[0], cuts[1], cuts[2], static_cast<ThreeOptPattern>(pattern)};
    }
  }
}

/// @brief Apply one reconnection, preserving the city at index zero.
/// @param tour Mutable permutation representing a symmetric cycle.
/// @param move Typed reconnection from sample_three_opt() or an equivalent move.
/// @throws std::invalid_argument Invalid cuts or unknown reconnection pattern.
/// @par Example
/// @code
/// std::vector<std::size_t> tour{0, 1, 2, 3, 4, 5};
/// dm::optim::apply_three_opt(tour, {1, 3, 5, dm::optim::ThreeOptPattern::exchange});
/// std::print("{}\n", tour[1]); // 3
/// @endcode
/// @see sample_three_opt()
inline void apply_three_opt(std::vector<std::size_t>& tour, const ThreeOptMove& move) {
  if (tour.size() < 6 || move.first == 0 || move.last > tour.size() ||
      move.first >= move.middle || move.middle >= move.last ||
      move.middle - move.first < 2 || move.last - move.middle < 2 || move.last - move.first > tour.size() - 2) {
    throw std::invalid_argument("3-opt requires three pairwise nonadjacent edges");
  }
  const auto first = tour.begin() + move.first;
  const auto middle = tour.begin() + move.middle;
  const auto last = tour.begin() + move.last;
  switch (move.pattern) {
    case ThreeOptPattern::reverse_both:
      std::reverse(first, middle);
      std::reverse(middle, last);
      return;
    case ThreeOptPattern::exchange:
      break;
    case ThreeOptPattern::exchange_first:
      std::reverse(first, middle);
      break;
    case ThreeOptPattern::exchange_second:
      std::reverse(middle, last);
      break;
    default:
      throw std::invalid_argument("Unknown 3-opt reconnection");
  }
  std::rotate(first, middle, last);
}

} // namespace datamunge::optim
