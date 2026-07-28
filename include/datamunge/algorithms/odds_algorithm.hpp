#pragma once

// Bruss's odds algorithm for optimal stopping: given n independent events whose
// success probabilities are p_1, ..., p_n (revealed one at a time), find the
// strategy that maximises the probability of stopping on the LAST success, and
// the value of that probability.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

struct OddsResult {
    std::size_t threshold{0};       // 1-based index s: start accepting the first success from here
    double      win_probability{0}; // probability of stopping on the last success
};

// `p[i]` is the success probability of event i (0-based), each in [0, 1].
// The optimal rule is: let nothing before index `threshold` count, then stop at
// the first success at or after it.
inline OddsResult odds_algorithm(const std::vector<double>& p) {
    const std::size_t n = p.size();
    OddsResult        r;
    if (n == 0) return r;

    // Sum the odds r_i = p_i / (1 - p_i) from the last event backwards until the
    // running sum reaches 1; that index is the optimal threshold.
    double sum = 0.0, prod = 1.0;
    std::size_t s = n; // 1-based; default "never stop before the end"
    for (std::size_t i = n; i-- > 0;) {
        const double q  = 1.0 - p[i];
        const double odds = q > 0 ? p[i] / q : 0.0;
        sum += odds;
        prod *= q;
        s = i + 1; // 1-based index of event i
        if (sum >= 1.0) break;
    }
    r.threshold       = s;
    r.win_probability = prod * sum; // Q_s * R_s
    return r;
}

} // namespace datamunge::algorithms
