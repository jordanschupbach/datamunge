#pragma once

/// \file truncated_binary_exponential_backoff.hpp
/// \brief Truncated binary exponential backoff (Ethernet CSMA/CD collision resolution).
///
/// Classic Ethernet resolves collisions with *truncated binary exponential backoff*. After the
/// \f$c\f$-th consecutive collision on a frame, a station waits a random number of slot times
/// drawn uniformly from \f$\{0,1,\dots,2^{\min(c,10)}-1\}\f$ before retransmitting -- the
/// *contention window* doubles per collision, so heavier contention spreads stations over a
/// wider interval. The window is *truncated* at \f$c=10\f$ (window \f$2^{10}=1024\f$), and after
/// 16 collisions the frame is abandoned. This module gives the contention window and slot bounds.

#include <cstdint>

namespace datamunge::algorithms {

/// \brief Contention window after \c collisions collisions: \f$2^{\min(c,10)}\f$ slot times.
inline int contention_window(int collisions) {
    int exp = collisions < 10 ? collisions : 10;   // truncation at 10
    return 1 << exp;
}

/// \brief Largest random slot count a station may wait after \c collisions collisions
///        (uniform in [0, window-1]).
inline int max_backoff_slots(int collisions) { return contention_window(collisions) - 1; }

/// \brief Whether the frame should be abandoned (Ethernet gives up after 16 collisions).
inline bool backoff_give_up(int collisions) { return collisions >= 16; }

/// \brief A chosen backoff slot given a uniform value \c u in [0,1) (for reproducibility):
///        floor(u * window).
inline int backoff_slot(int collisions, double u) {
    int w = contention_window(collisions);
    int s = static_cast<int>(u * w);
    return s < w ? s : w - 1;
}

}  // namespace datamunge::algorithms
