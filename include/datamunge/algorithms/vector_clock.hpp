#pragma once
/// @file vector_clock.hpp
/// @brief Checked native vector clock and actual producing diagnostics.
#include <algorithm>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Mutable vector timestamp with nonnegative signed integer counters.
/// Counters start0. Caller serializes mutations; distinct clocks/calls share no state.
/// @param n Process count, default0; at most1000000.
/// @throws std::length_error Process count exceeds1000000.
/// @par Example
/// @code
/// dm::algorithms::VectorClock clock(3);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VectorClock {
  std::vector<int> v; ///< Owned process counters; public edits must preserve nonnegativity.

  /// @brief Construct a zero vector clock.
  /// @param n Process count<=1000000, default0.
  /// @throws std::length_error Too many processes.
  /// @par Example
  /// @code
  /// dm::algorithms::VectorClock c(2);
  /// @endcode
  /// @par References
  /// - The accompanying report derives the implemented recurrence.
  /// @see Other operations in this header.
  explicit VectorClock(std::size_t n = 0) {
    if (n > 1000000)
      throw std::length_error("clock size");
    v.assign(n, 0);
  }
};
/// @brief Validate clock counters before an operation.
/// @param c Clock with nonnegative entries.
/// @throws std::invalid_argument A counter is negative.
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
/// @par Example
/// @code
/// dm::algorithms::VectorClock c(2);dm::algorithms::vc_validate(c);
/// @endcode
inline void vc_validate(const VectorClock& c) {
  for (int x : c.v)
    if (x < 0)
      throw std::invalid_argument("negative clock counter");
}
/// @brief Increment the local process counter atomically with respect to validation.
/// O(number of processes) validation/work; no internal locking. Causality interpretation assumes the complete event/send/receive protocol.
/// @param c Mutable clock, unchanged on failure.
/// @param pid In-range local process ID.
/// @throws std::invalid_argument Invalid counter, shape or process ID where applicable.
/// @throws std::overflow_error Own counter would exceed INT_MAX.
/// @par Example
/// @code
/// dm::algorithms::VectorClock c(2);dm::algorithms::vc_event(c,0);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline void vc_event(VectorClock& c, std::size_t pid) {
  vc_validate(c);
  if (pid >= c.v.size())
    throw std::invalid_argument("clock pid");
  if (c.v[pid] == INT_MAX)
    throw std::overflow_error("clock overflow");
  ++c.v[pid];
}
/// @brief Increment a send event and copy its timestamp.
/// O(number of processes) validation/work; no internal locking. Causality interpretation assumes the complete event/send/receive protocol.
/// @param c Mutable local clock.
/// @param pid In-range sending process ID.
/// @return Owned post-increment message timestamp.
/// @throws std::invalid_argument Invalid counter, shape or process ID where applicable.
/// @throws std::overflow_error Own counter would exceed INT_MAX.
/// @par Example
/// @code
/// dm::algorithms::VectorClock c(2);auto message=dm::algorithms::vc_send(c,0);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline VectorClock vc_send(VectorClock& c, std::size_t pid) {
  vc_event(c, pid);
  return c;
}
/// @brief Merge componentwise maxima then increment the receive event.
/// O(number of processes) validation/work; no internal locking. Causality interpretation assumes the complete event/send/receive protocol.
/// @param c Mutable receiver clock, unchanged on failure.
/// @param pid In-range receiving process ID.
/// @param message Borrowed equal-size nonnegative message clock.
/// @throws std::invalid_argument Invalid counter, shape or process ID where applicable.
/// @throws std::overflow_error Own counter would exceed INT_MAX.
/// @par Example
/// @code
/// dm::algorithms::VectorClock c(2),m(2);dm::algorithms::vc_receive(c,1,m);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline void vc_receive(VectorClock& c, std::size_t pid, const VectorClock& message) {
  vc_validate(c);
  vc_validate(message);
  if (c.v.size() != message.v.size() || pid >= c.v.size())
    throw std::invalid_argument("clock shape/pid");
  if (std::max(c.v[pid], message.v[pid]) == INT_MAX)
    throw std::overflow_error("clock overflow");
  for (std::size_t i = 0; i < c.v.size(); ++i)
    c.v[i] = std::max(c.v[i], message.v[i]);
  ++c.v[pid];
}
/// @brief Compare vector timestamps in the strict causal partial order.
/// O(number of processes) validation/work; no internal locking. Causality interpretation assumes the complete event/send/receive protocol.
/// @param a First nonnegative clock.
/// @param b Second same-size nonnegative clock.
/// @return True if a<=b componentwise and at least one strict inequality.
/// @throws std::invalid_argument Invalid counter, shape or process ID where applicable.
/// @par Example
/// @code
/// dm::algorithms::VectorClock a(2),b(2);dm::algorithms::vc_event(b,0);bool before=dm::algorithms::vc_happens_before(a,b);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline bool vc_happens_before(const VectorClock& a, const VectorClock& b) {
  vc_validate(a);
  vc_validate(b);
  if (a.v.size() != b.v.size())
    throw std::invalid_argument("clock shape");
  bool strict = false;
  for (std::size_t i = 0; i < a.v.size(); ++i) {
    if (a.v[i] > b.v[i])
      return false;
    if (a.v[i] < b.v[i])
      strict = true;
  }
  return strict;
}
/// @brief Detect incomparable distinct vector timestamps.
/// O(number of processes) validation/work; no internal locking. Causality interpretation assumes the complete event/send/receive protocol.
/// @param a First nonnegative clock.
/// @param b Second same-size nonnegative clock.
/// @return True only for distinct incomparable vectors; equal stamps are not concurrent.
/// @throws std::invalid_argument Invalid counter, shape or process ID where applicable.
/// @par Example
/// @code
/// dm::algorithms::VectorClock a(2),b(2);dm::algorithms::vc_event(a,0);dm::algorithms::vc_event(b,1);bool concurrent=dm::algorithms::vc_concurrent(a,b);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline bool vc_concurrent(const VectorClock& a, const VectorClock& b) {
  return !vc_happens_before(a, b) && !vc_happens_before(b, a) && a.v != b.v;
}

} // namespace datamunge::algorithms
