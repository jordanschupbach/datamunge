#pragma once

// Random-restart hill climbing: a simple but effective global optimizer for a
// multimodal function. Plain hill climbing walks uphill from a starting point and
// stops at the first local optimum -- which may be far from the best. Random
// restart wraps it in an outer loop: run hill climbing from many *random* starting
// points and keep the best summit found. With enough restarts the search samples
// every basin of attraction, so the global optimum is found with high probability,
// while each individual climb stays cheap and derivative-free.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

struct HillClimbResult {
    std::vector<double> x;
    double              value{0};
};

namespace detail {
struct HcRng {
    std::uint64_t s;
    explicit HcRng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double        uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};
} // namespace detail

// Maximize f over the box [lo_i, hi_i] by random-restart hill climbing.
// `f` is any callable double(const std::vector<double>&). Each climb perturbs one
// coordinate at a time by +/- step, shrinking step when stuck.
template <class F>
HillClimbResult random_restart_hill_climbing(F f, const std::vector<double>& lo,
                                             const std::vector<double>& hi, int restarts = 50,
                                             std::uint64_t seed = 1) {
    const std::size_t d = lo.size();
    detail::HcRng     rng(seed);
    HillClimbResult   best;
    bool              have_best = false;

    for (int r = 0; r < restarts; ++r) {
        std::vector<double> x(d);
        for (std::size_t i = 0; i < d; ++i) x[i] = lo[i] + (hi[i] - lo[i]) * rng.uniform();
        double fx   = f(x);
        double step = 0.0;
        for (std::size_t i = 0; i < d; ++i) step = std::max(step, 0.1 * (hi[i] - lo[i]));

        while (step > 1e-9) {
            bool improved = false;
            for (std::size_t i = 0; i < d; ++i) {
                for (double s : {step, -step}) {
                    std::vector<double> y = x;
                    y[i] += s;
                    if (y[i] < lo[i] || y[i] > hi[i]) continue;
                    const double fy = f(y);
                    if (fy > fx) { x = y; fx = fy; improved = true; }
                }
            }
            if (!improved) step *= 0.5; // no uphill move at this scale -> refine
        }

        if (!have_best || fx > best.value) { best.x = x; best.value = fx; have_best = true; }
    }
    return best;
}

} // namespace datamunge::algorithms
