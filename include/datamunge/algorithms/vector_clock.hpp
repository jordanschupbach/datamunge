#pragma once

/// \file vector_clock.hpp
/// \brief Vector clocks: timestamps that capture causality exactly, so concurrency is
///        detectable (Fidge 1988; Mattern 1988).
///
/// A Lamport scalar clock guarantees \f$a\to b \Rightarrow C(a)<C(b)\f$ but not the
/// converse, so it cannot tell causally-ordered events from concurrent ones. A *vector
/// clock* fixes this: each of \f$n\f$ processes carries an \f$n\f$-entry vector. A process
/// increments its own entry on each event/send; on receive it takes the component-wise
/// maximum with the message's vector, then increments its own entry. Then, comparing two
/// stamps componentwise,
/// \f[
///   a \to b \iff V(a) < V(b)\ (\text{every component} \le,\ \text{at least one} <),
/// \f]
/// and two events are *concurrent* iff neither vector dominates the other. Vector clocks
/// thus detect causality *and* concurrency exactly -- the basis of conflict detection in
/// eventually-consistent stores (Dynamo, Riak) and of consistent-snapshot algorithms.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// A vector clock over \p n processes (all entries start at 0).
struct VectorClock {
    std::vector<int> v;
    explicit VectorClock(std::size_t n = 0) : v(n, 0) {}
};

/// A local event at process \p pid: increment its own component.
inline void vc_event(VectorClock& c, std::size_t pid) { ++c.v[pid]; }

/// Prepare to send from \p pid: increment and return the vector to attach.
inline VectorClock vc_send(VectorClock& c, std::size_t pid) {
    ++c.v[pid];
    return c;
}

/// Receive \p message at \p pid: componentwise max, then increment own entry.
inline void vc_receive(VectorClock& c, std::size_t pid, const VectorClock& message) {
    for (std::size_t i = 0; i < c.v.size(); ++i) c.v[i] = std::max(c.v[i], message.v[i]);
    ++c.v[pid];
}

/// True iff \p a happened-before \p b (a <= b componentwise and a != b).
inline bool vc_happens_before(const VectorClock& a, const VectorClock& b) {
    bool strictly_less = false;
    for (std::size_t i = 0; i < a.v.size(); ++i) {
        if (a.v[i] > b.v[i]) return false;
        if (a.v[i] < b.v[i]) strictly_less = true;
    }
    return strictly_less;
}

/// True iff \p a and \p b are concurrent (neither happened-before the other).
inline bool vc_concurrent(const VectorClock& a, const VectorClock& b) {
    return !vc_happens_before(a, b) && !vc_happens_before(b, a) && a.v != b.v;
}

}  // namespace datamunge::algorithms
