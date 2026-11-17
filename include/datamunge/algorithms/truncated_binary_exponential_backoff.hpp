#pragma once
/// @file truncated_binary_exponential_backoff.hpp
/// @brief Checked truncated binary exponential backoff operations and producer diagnostics.
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Return the capped binary contention window.
/// Pure bounded arithmetic; no RNG, clock, frame state or synchronization. A uniform real u implies uniform slots; finite PRNG mapping may be biased.
/// @param collisions Nonnegative collision count.
/// @return 2^min(collisions,10).
/// @throws std::invalid_argument Negative collisions or invalid fraction where supplied.
/// @par Example
/// @code
/// auto w=dm::algorithms::contention_window(3); //8
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline int contention_window(int collisions) {
  if (collisions < 0)
    throw std::invalid_argument("negative collisions");
  return 1 << std::min(collisions, 10);
}
/// @brief Return the largest candidate slot.
/// Pure bounded arithmetic; no RNG, clock, frame state or synchronization. A uniform real u implies uniform slots; finite PRNG mapping may be biased.
/// @param collisions Nonnegative collision count.
/// @return Window minus1.
/// @throws std::invalid_argument Negative collisions or invalid fraction where supplied.
/// @par Example
/// @code
/// auto w=dm::algorithms::max_backoff_slots(3); //7
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline int max_backoff_slots(int collisions) {
  return contention_window(collisions) - 1;
}
/// @brief Test the16-collision abandonment policy.
/// Pure bounded arithmetic; no RNG, clock, frame state or synchronization. A uniform real u implies uniform slots; finite PRNG mapping may be biased.
/// @param collisions Nonnegative collision count.
/// @return True at16 or later.
/// @throws std::invalid_argument Negative collisions or invalid fraction where supplied.
/// @par Example
/// @code
/// bool stop=dm::algorithms::backoff_give_up(16);
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline bool backoff_give_up(int collisions) {
  (void)contention_window(collisions);
  return collisions >= 16;
}
/// @brief Map a caller random fraction to a candidate slot.
/// Pure bounded arithmetic; no RNG, clock, frame state or synchronization. A uniform real u implies uniform slots; finite PRNG mapping may be biased.
/// @param collisions Nonnegative collision count.
/// @param u Finite caller-supplied fraction in [0,1).
/// @return floor(u*window); no retry is authorized at or after16.
/// @throws std::invalid_argument Negative collisions or invalid fraction where supplied.
/// @par Example
/// @code
/// auto slot=dm::algorithms::backoff_slot(3,0.5); //4
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline int backoff_slot(int collisions, double u) {
  int w = contention_window(collisions);
  if (!std::isfinite(u) || u < 0 || u >= 1)
    throw std::invalid_argument("u outside [0,1)");
  return static_cast<int>(u * w);
}

} // namespace datamunge::algorithms
