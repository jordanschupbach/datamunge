#pragma once

/// \file exponential_backoff.hpp
/// \brief Exponential backoff retry delays, with capping and jitter.
///
/// When many clients retry a failed operation (a network request, a lock acquisition), retrying
/// on a fixed schedule makes them collide again and again. *Exponential backoff* spreads them
/// out by doubling the wait after each failure: attempt \f$k\f$ waits \f$\min(\text{base}\cdot
/// 2^k, \text{cap})\f$. Adding *jitter* -- randomizing within that window -- breaks up the
/// synchronized "thundering herd" that pure doubling still leaves. This module computes the
/// backoff delays and a jittered variant.

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// \brief Backoff delay before retry attempt \c k (0-indexed): min(base * 2^k, cap).
inline double exponential_backoff_delay(int k, double base, double cap) {
    double d = base;
    for (int i = 0; i < k && d < cap; ++i) d *= 2.0;
    return d < cap ? d : cap;
}

/// \brief The capped exponential backoff delays for attempts 0..n-1.
inline std::vector<double> exponential_backoff_sequence(int n, double base, double cap) {
    std::vector<double> s;
    s.reserve(n);
    for (int k = 0; k < n; ++k) s.push_back(exponential_backoff_delay(k, base, cap));
    return s;
}

/// \brief "Full jitter" delay: a uniform sample in [0, min(base*2^k, cap)].
///
/// \param u  a uniform value in [0, 1) (supplied by the caller for reproducibility).
inline double exponential_backoff_jittered(int k, double base, double cap, double u) {
    return u * exponential_backoff_delay(k, base, cap);
}

}  // namespace datamunge::algorithms
