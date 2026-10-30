/// @file approximate_counting.cpp
/// @brief Morris counter updates, the exact variance and the exact exponent distribution.

#include <datamunge/algorithms/approximate_counting.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::algorithms {

MorrisCounter::MorrisCounter(double base) : base_(base) {
    if (!(base > 1.0)) throw std::invalid_argument("MorrisCounter: base must be greater than 1");
}

void MorrisCounter::increment(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(rng) < std::pow(base_, -exponent_)) exponent_ += 1.0;
}

void MorrisCounter::add(std::uint64_t events, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    while (events > 0) {
        const double p = std::pow(base_, -exponent_);
        std::uint64_t wait = 1;  // events up to and including the next increment
        if (p < 1.0) {
            const double u = 1.0 - unit(rng);  // in (0, 1]
            const double k = std::ceil(std::log(u) / std::log1p(-p));
            wait = k < 1.0 ? 1 : (k >= static_cast<double>(events) ? events + 1 : static_cast<std::uint64_t>(k));
        }
        if (wait > events) return;
        events -= wait;
        exponent_ += 1.0;
    }
}

int MorrisCounter::register_bits() const {
    int bits = 0;
    for (auto c = static_cast<std::uint64_t>(exponent_); c > 0; c >>= 1) ++bits;
    return bits;
}

double MorrisCounter::estimate() const {
    return base_ == 2.0 ? std::pow(2.0, exponent_) - 1.0 : (std::pow(base_, exponent_) - 1.0) / (base_ - 1.0);
}

double morris_variance(double n, double base) { return (base - 1.0) * n * (n - 1.0) / 2.0; }

std::vector<double> morris_exponent_pmf(std::uint64_t n, double base, std::size_t max_level) {
    if (!(base > 1.0)) throw std::invalid_argument("morris_exponent_pmf: base must be greater than 1");
    std::vector<double> p(max_level + 1, 0.0), up(max_level + 1);
    for (std::size_t c = 0; c <= max_level; ++c) up[c] = std::pow(base, -static_cast<double>(c));
    p[0] = 1.0;
    for (std::uint64_t e = 0; e < n; ++e) {
        // Before event e the exponent is at most e; update from the top so each level moves once.
        const std::size_t hi = e < max_level ? static_cast<std::size_t>(e) : max_level;
        for (std::size_t c = hi + 1; c-- > 0;) {
            if (c == max_level) continue;  // the last entry absorbs everything above it
            const double moved = p[c] * up[c];
            p[c + 1] += moved;
            p[c] -= moved;
        }
    }
    return p;
}

}  // namespace datamunge::algorithms
