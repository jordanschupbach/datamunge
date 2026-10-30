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
    std::vector<int>    included;                 ///< 1 for the clocks used in the average, 0 for those rejected.
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
    r.included.assign(clocks.size(), 0);
    for (std::size_t i = 0; i < clocks.size(); ++i) {
        if (outlier_margin > 0.0 && std::fabs(clocks[i] - clocks[master]) > outlier_margin) continue;
        sum += clocks[i];
        r.included[i] = 1;
        ++n;
    }
    r.synchronized_time = n > 0 ? sum / static_cast<double>(n) : clocks[master];
    r.adjustments.resize(clocks.size());
    for (std::size_t i = 0; i < clocks.size(); ++i) r.adjustments[i] = r.synchronized_time - clocks[i];
    return r;
}

/// \brief Berkeley round with Gusella and Zatti's fault-tolerant average: the largest group of
///        readings that fit in an interval of width @p window is averaged, and every clock,
///        including the rejected ones, is told how far it is from that average.
///
/// A clock is rejected when it cannot join the largest agreeing group, so a minority of broken
/// clocks cannot pull the others. Ties between equally large groups go to the one with the
/// smaller spread.
/// \param readings  each node's clock reading (or its offset from a common reference).
/// \param window    width of the agreeing group; 0 or less averages every reading.
/// \return the average, the adjustments and the 0/1 inclusion flags.
BerkeleyResult berkeley_sync_window(const std::vector<double>& readings, double window);

/// \brief Frequency error of a tuning-fork quartz crystal at @p temperature_c, in parts per
///        million: \f$-\beta(T-T_0)^2\f$, the parabolic law of 32.768 kHz watch crystals.
/// \param temperature_c temperature in degrees Celsius.
/// \param turnover_c    turnover temperature \f$T_0\f$ where the crystal is exact (typically 25 C).
/// \param coefficient   parabolic coefficient \f$\beta\f$ in ppm per degree squared (typically 0.034).
/// \return the rate error in ppm; negative means the clock runs slow.
double crystal_drift_ppm(double temperature_c, double turnover_c = 25.0, double coefficient = 0.034);

/// Settings for simulate_berkeley_network().
struct BerkeleyNetworkOptions {
    double      round_interval = 3600.0;  ///< Seconds between synchronization rounds.
    double      min_delay      = 0.002;   ///< Smallest one-way message delay, in seconds.
    double      max_delay      = 0.050;   ///< Largest one-way message delay, in seconds.
    double      window         = 1.0;     ///< Fault-tolerant averaging window in seconds (0: plain average).
    std::size_t master         = 0;       ///< Index of the polling node.
    bool        synchronize    = true;    ///< false lets the clocks drift freely, for comparison.
    unsigned    seed           = 1;       ///< Seed for the random message delays.
};

/// Trace of simulate_berkeley_network(): offsets from true time at every round.
struct BerkeleyNetworkTrace {
    std::vector<double>              time_days;  ///< True time of each round, in days.
    std::vector<std::vector<double>> before;     ///< before[i][k]: offset of node i, in seconds, just before round k.
    std::vector<std::vector<double>> after;      ///< after[i][k]: offset of node i just after round k.
    std::vector<double>              skew_before;  ///< Spread (max - min) of the included clocks just before each round.
    std::vector<double>              skew_after;   ///< Spread of the included clocks just after each round.
    std::vector<std::size_t>         rejected;     ///< Clocks left out of each round's average.
    std::vector<std::vector<int>>    included;     ///< included[i][k]: 1 if node i was in round k's average.
};

/// \brief Simulates Berkeley synchronization of drifting clocks over many days.
///
/// Node \f$i\f$ runs fast by `drift_ppm[i][d]` parts per million on day \f$d\f$, so its offset from
/// true time grows by that rate times the round interval between rounds. In each round the master
/// polls every node: the request takes \f$a\f$ and the reply \f$b\f$ seconds, both uniform on
/// [min_delay, max_delay]. The master estimates the node's clock as its reply plus half the round
/// trip, which is wrong by \f$(a-b)/2\f$. It then forms berkeley_sync_window() of the estimates and
/// every node applies its adjustment.
/// \param drift_ppm per node, the daily rate error in ppm (positive: fast); all rows the same length.
/// \param options   round interval, delays, window, master, seed.
/// \return offsets before and after every round, and the spread of the agreeing clocks.
/// \throws std::invalid_argument for empty or ragged input, a bad master index or bad delays.
BerkeleyNetworkTrace simulate_berkeley_network(const std::vector<std::vector<double>>& drift_ppm,
                                               const BerkeleyNetworkOptions&            options = {});

}  // namespace datamunge::algorithms
