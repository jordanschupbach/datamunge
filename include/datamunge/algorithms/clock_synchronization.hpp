#pragma once

/// \file clock_synchronization.hpp
/// \brief Two classic physical-clock synchronization algorithms: Cristian's algorithm
///        (sync to a time server) and the Berkeley algorithm (internal averaging).
///
/// Physical clocks drift, so distributed systems periodically resynchronize them.
///   - *Cristian's algorithm* (1989): a client asks a time server for the time and, since
///     the reply took a round trip \f$RTT\f$, estimates the true time as
///     \f$T_{\text{server}} + RTT/2\f$ (assuming symmetric latency). The remaining
///     uncertainty is bounded by \f$RTT/2 - \text{min\_latency}\f$.
///   - *The Berkeley algorithm* (1989): there is no authoritative server; a master polls
///     everyone's clocks, computes the *average* (optionally discarding outliers), and
///     sends each node the *adjustment* needed to reach that average. It keeps the group
///     internally consistent even if none of them is externally accurate.
/// Both are foundational to NTP-style time keeping.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief Cristian's estimate of the true time given a server reading and the round-trip time.
inline double cristian_estimate(double server_time, double round_trip_time) {
    return server_time + round_trip_time / 2.0;
}

/// \brief Cristian's accuracy bound: the estimate is within +/- this of the true time.
///
/// With a known minimum one-way latency, the error is bounded by \f$RTT/2 - \text{min\_latency}\f$.
inline double cristian_error_bound(double round_trip_time, double min_one_way_latency = 0.0) {
    const double b = round_trip_time / 2.0 - min_one_way_latency;
    return b > 0.0 ? b : 0.0;
}

/// Result of a Berkeley synchronization round.
struct BerkeleyResult {
    double              synchronized_time = 0.0;  ///< The agreed (average) time.
    std::vector<double> adjustments;              ///< Per-node correction to reach the average.
};

/// \brief Berkeley clock synchronization: average the clocks and return per-node adjustments.
///
/// \param clocks         Each node's current clock reading (index 0.. ; the master is just
///                       one of them for averaging purposes).
/// \param outlier_margin Clocks farther than this from the master are excluded from the
///                       average (0 disables outlier rejection).
/// \param master         Index of the master node (used only for outlier rejection).
inline BerkeleyResult berkeley_sync(const std::vector<double>& clocks, double outlier_margin = 0.0,
                                    std::size_t master = 0) {
    BerkeleyResult r;
    if (clocks.empty()) return r;
    double      sum = 0.0;
    std::size_t n   = 0;
    for (std::size_t i = 0; i < clocks.size(); ++i) {
        if (outlier_margin > 0.0 && std::fabs(clocks[i] - clocks[master]) > outlier_margin) continue;
        sum += clocks[i];
        ++n;
    }
    r.synchronized_time = n > 0 ? sum / static_cast<double>(n) : clocks[master];
    r.adjustments.resize(clocks.size());
    for (std::size_t i = 0; i < clocks.size(); ++i) r.adjustments[i] = r.synchronized_time - clocks[i];
    return r;
}

}  // namespace datamunge::algorithms
