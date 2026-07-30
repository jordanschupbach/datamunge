#pragma once

/// \file lamport_clock.hpp
/// \brief Lamport logical clocks: a scalar timestamp that respects the happened-before
///        relation in a distributed system (Lamport 1978).
///
/// Distributed processes have no shared physical clock, yet we need a consistent notion
/// of "before". Lamport's *logical clock* is a per-process integer counter with two
/// rules: increment before every local event or send, and on receiving a message set the
/// clock to \f$\max(\text{local}, \text{message stamp}) + 1\f$. The resulting timestamps
/// satisfy the *clock condition*: if event \f$a\f$ *happened before* \f$b\f$ (Lamport's
/// partial order -- program order within a process, or a send before its receive), then
/// \f$C(a) < C(b)\f$. The converse does not hold (equal-or-ordered timestamps do not imply
/// causality), which is what vector clocks fix. Lamport clocks are the foundation of
/// distributed mutual exclusion, snapshots, and totally-ordered multicast.

#include <algorithm>
#include <cstdint>

namespace datamunge::algorithms {

/// A single process's Lamport logical clock.
class LamportClock {
 public:
    /// The current logical time.
    std::int64_t time() const { return time_; }

    /// A local (internal) event: increment and return the new time.
    std::int64_t local_event() { return ++time_; }

    /// Prepare to send a message: increment and return the timestamp to attach.
    std::int64_t send() { return ++time_; }

    /// Receive a message stamped \p message_time: advance past it, then tick.
    std::int64_t receive(std::int64_t message_time) {
        time_ = std::max(time_, message_time) + 1;
        return time_;
    }

 private:
    std::int64_t time_ = 0;
};

}  // namespace datamunge::algorithms
