#pragma once
/// @file lamport_clock.hpp
/// @brief Checked scalar logical clocks with actual local/send/receive transition evidence.
#include <cstdint>
namespace datamunge::algorithms {
/// @brief Actual semantic operation of one local logical-clock transition.
/// @par Example
/// @code
/// dm::algorithms::LamportClock c;const auto e=c.send_step();
/// std::print("{}\n",e.kind==dm::algorithms::LamportEventKind::Send); // true
/// @endcode
/// @see LamportEvent
enum class LamportEventKind {
  Local,  ///< Actual local/internal event increment.
  Send,   ///< Actual send event increment; after is the stamp to attach.
  Receive ///< Actual receive max(local,message)+1 update.
};
/// @brief Owned actual before/message/after values of one producing clock update.
/// message_time is the caller's actual nonnegative stamp for Receive, zero
/// placeholder otherwise. No process/message identity, network validation,
/// physical duration or happened-before relation is inferred by this record.
/// @par Example
/// @code
/// dm::algorithms::LamportClock c;const auto e=c.receive_step(5);
/// std::print("{} {} {}\n",e.before,e.message_time,e.after); // 0 5 6
/// @endcode
/// @see LamportClock
struct LamportEvent {
  LamportEventKind kind{LamportEventKind::Local}; ///< Actual chosen local API operation.
  std::int64_t     before{0};       ///< Actual local nonnegative clock before this update.
  std::int64_t     message_time{0}; ///< Actual incoming stamp or zero placeholder for Local/Send.
  std::int64_t     after{0};        ///< Actual committed local clock after this update.
};
/// @brief Own one process's nonnegative signed64 scalar Lamport logical clock.
/// Starts0. Internal/send events increment once; receive commits
/// max(current,message_stamp)+1. Scalar order respects happened-before when the
/// caller records every local event and attaches actual send stamps to receives.
/// Converse false: ordered timestamps need not imply causality. This class does
/// not authenticate messages, model a network, synchronize threads or produce a
/// unique total order across processes. Pair with process ids externally if needed.
/// All transitions O(1), no allocation; invalid/overflow failures leave state
/// unchanged. Copies own values, independent objects share no mutable state.
/// @par Example
/// @code
/// dm::algorithms::LamportClock a,b;const auto stamp=a.send();
/// std::print("{} {}\n",stamp,b.receive(stamp)); // 1 2
/// @endcode
/// @par References
/// - Lamport(1978), Time, Clocks, and the Ordering of Events in a Distributed System, Communications of the ACM21(7),558–565, https://doi.org/10.1145/359545.359563.
/// @see LamportEvent, receive_step()
class LamportClock {
 public:
  /// @brief Construct a zero local logical clock without allocation/shared state.
  /// @par Example
  /// @code
  /// const dm::algorithms::LamportClock c;std::print("{}\n",c.time()); // 0
  /// @endcode
  LamportClock() = default;
  /// @brief Inspect the current nonnegative logical time without mutation.
  /// @return Owned current signed64 value, initially0, no allocation.
  std::int64_t time() const noexcept { return time_; }
  /// @brief Commit one local/internal event through the shared checked transition.
  /// @return Actual new local value, previous+1.
  /// @throws std::overflow_error If current time is INT64_MAX; state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::LamportClock c;std::print("{}\n",c.local_event()); // 1
  /// @endcode
  /// @see local_step()
  std::int64_t local_event();
  /// @brief Commit one send event and return the actual stamp to attach to its message.
  /// @return Actual new local value, previous+1.
  /// @throws std::overflow_error If current time is INT64_MAX; state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::LamportClock c;std::print("{}\n",c.send()); // 1
  /// @endcode
  /// @see send_step(), receive()
  std::int64_t send();
  /// @brief Commit max(current,incoming)+1 through the shared checked receive transition.
  /// Require incoming>=0; no message-origin/causality validation. Old/out-of-order
  /// nonnegative stamps still advance past the current value. Failure is atomic.
  /// @param message_time Actual nonnegative signed64 received timestamp.
  /// @return Actual new local value, strictly greater than current and incoming.
  /// @throws std::invalid_argument On negative incoming timestamp, state unchanged.
  /// @throws std::overflow_error If current/incoming maximum is INT64_MAX, state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::LamportClock c;std::print("{}\n",c.receive(5)); // 6
  /// @endcode
  /// @see receive_step(), send()
  std::int64_t receive(std::int64_t message_time);
  /// @brief Commit one local event and capture its actual before/after values once.
  /// Shares local_event() kernel, no replay/caller references/allocation.
  /// @return Owned actual local transition, message_time0 placeholder.
  /// @throws std::overflow_error At INT64_MAX, state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::LamportClock c;const auto e=c.local_step();
  /// std::print("{} {}\n",e.before,e.after); // 0 1
  /// @endcode
  /// @see local_event(), LamportEvent
  LamportEvent local_step();
  /// @brief Commit one send and capture actual before/after plus Send category.
  /// Shares send() kernel, no replay/caller references/allocation.
  /// @return Owned actual send transition; attach after to the message.
  /// @throws std::overflow_error At INT64_MAX, state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::LamportClock c;std::print("{}\n",c.send_step().after); // 1
  /// @endcode
  /// @see send(), LamportEvent
  LamportEvent send_step();
  /// @brief Commit one receive and capture its actual original stamp and local state boundary.
  /// Same complete stamp/overflow validation and atomicity as receive().
  /// No message/network inference, replay or allocation.
  /// @param message_time Actual nonnegative received timestamp.
  /// @return Owned actual receive transition.
  /// @throws std::invalid_argument On negative incoming stamp, state unchanged.
  /// @throws std::overflow_error At maximum current/incoming stamp, state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::LamportClock c;const auto e=c.receive_step(5);
  /// std::print("{} {}\n",e.message_time,e.after); // 5 6
  /// @endcode
  /// @see receive(), LamportEvent
  LamportEvent receive_step(std::int64_t message_time);

 private:
  std::int64_t time_{0}; ///< Actual owned local logical time, always in[0,INT64_MAX].

  /// @brief Validate and commit one actual O(1) semantic clock transition.
  /// @param kind Actual Local/Send/Receive operation.
  /// @param message Actual received stamp or zero placeholder for Local/Send.
  /// @return Owned actual pre/post update values.
  /// @throws std::invalid_argument On negative Receive stamp before mutation.
  /// @throws std::overflow_error On max(current,message)+1 overflow before mutation.
  LamportEvent step(LamportEventKind kind, std::int64_t message);
};
} // namespace datamunge::algorithms
