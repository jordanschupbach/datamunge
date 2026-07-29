// Wang-Landau algorithm (2001): estimate the density of states g(E) -- the number
// of configurations at each energy -- by a random walk that flattens its own
// energy histogram. Ordinary Monte Carlo samples configurations with Boltzmann
// weight and barely visits rare energies; Wang-Landau instead accepts moves with
// probability min(1, g(E_old)/g(E_new)) using a running estimate of g, so it
// spends equal time at every energy. Each visit multiplies g(E) by a modification
// factor f; once the histogram is flat, f is reduced (f -> sqrt(f)) and the walk
// repeats, converging g to the true density of states as f -> 1.
//
// This implementation uses the toy system of N Ising-like spins whose energy is
// the number of up-spins, for which the exact density of states is the binomial
// coefficient g(E) = C(N, E) -- a clean target to verify against.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
struct WlRng {
    std::uint64_t s;
    explicit WlRng(std::uint64_t seed) : s(seed ? seed : 0x100000001B3ULL) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double        uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    std::size_t   below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};
} // namespace detail

// Estimate log g(E) for E = 0..N (energy = number of up-spins). Returns the
// log-density normalized so that log g(0) = 0; the exact answer is
// log C(N,E) = log(N! / (E!(N-E)!)). `flatness` is the histogram flatness
// threshold, `log_f_min` the stopping modification factor.
inline std::vector<double> wang_landau_dos(int N, double flatness = 0.85,
                                           double log_f_min = 1e-6, std::uint64_t seed = 1) {
    const int              bins = N + 1;
    std::vector<double>    logg(bins, 0.0);
    std::vector<long long> hist(bins, 0);
    std::vector<char>      spin(N, 0); // all down => E = 0
    int                    E = 0;
    detail::WlRng          rng(seed);

    double     log_f  = 1.0;
    const long check  = static_cast<long>(bins) * 1000; // moves between flatness checks
    while (log_f > log_f_min) {
        for (int b = 0; b < bins; ++b) hist[b] = 0;
        bool flat = false;
        while (!flat) {
            for (long step = 0; step < check; ++step) {
                const std::size_t i  = rng.below(static_cast<std::size_t>(N));
                const int         En = E + (spin[i] ? -1 : 1); // flipping one spin moves E by +-1
                // Accept with min(1, g(E)/g(En)) = min(1, exp(logg[E]-logg[En])).
                if (logg[E] - logg[En] >= 0.0 || rng.uniform() < std::exp(logg[E] - logg[En])) {
                    spin[i] ^= 1;
                    E = En;
                }
                logg[E] += log_f;
                ++hist[E];
            }
            // Flatness: every bin visited and min >= flatness * mean.
            long long mn = hist[0], tot = 0;
            for (int b = 0; b < bins; ++b) { if (hist[b] < mn) mn = hist[b]; tot += hist[b]; }
            const double mean = static_cast<double>(tot) / bins;
            flat              = (mn > 0) && (static_cast<double>(mn) >= flatness * mean);
        }
        log_f *= 0.5; // f -> sqrt(f) in log space
    }

    const double base = logg[0];
    for (double& v : logg) v -= base; // normalize so log g(0) = 0
    return logg;
}

} // namespace datamunge::algorithms
