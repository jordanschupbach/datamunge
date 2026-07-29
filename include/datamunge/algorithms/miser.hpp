#pragma once

// MISER: recursive stratified-sampling Monte Carlo integration (Press & Farrar,
// 1990). Plain Monte Carlo spends samples uniformly and its error falls only as
// 1/sqrt(N); MISER instead spends more samples where the integrand varies most.
// At each level it uses a small pilot sample to find the coordinate whose
// midpoint bisection most reduces the combined variance, splits the region there,
// allocates the remaining samples to the two halves in proportion to their
// estimated standard deviations, and recurses. Concentrating effort on the
// high-variance sub-regions can cut the error dramatically for peaked integrands.

#include <cmath>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

struct MiserResult {
    double value{0};    // estimate of the integral
    double variance{0}; // estimate of the estimator's variance
};

namespace detail {

struct MiserRng { // deterministic xorshift, in [0,1)
    std::uint64_t s;
    explicit MiserRng(std::uint64_t seed) : s(seed ? seed : 0x2545F4914F6CDD1DULL) {}
    double next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (s >> 11) * (1.0 / 9007199254740992.0);
    }
};

template <class F>
MiserResult miser_rec(F& f, std::vector<double>& a, std::vector<double>& b, long calls,
                      MiserRng& rng, std::vector<double>& pt) {
    const std::size_t dim = a.size();
    double            vol = 1.0;
    for (std::size_t i = 0; i < dim; ++i) vol *= (b[i] - a[i]);

    const long MIN_CALLS = 32;
    const long PRE_MIN   = 16;
    if (calls < MIN_CALLS * 2) {
        // Plain Monte Carlo on this region.
        double sum = 0.0, sum2 = 0.0;
        const long n = calls < 2 ? 2 : calls;
        for (long k = 0; k < n; ++k) {
            for (std::size_t i = 0; i < dim; ++i) pt[i] = a[i] + (b[i] - a[i]) * rng.next();
            const double fx = f(pt);
            sum += fx; sum2 += fx * fx;
        }
        const double mean = sum / n;
        const double var  = (sum2 / n - mean * mean) / n; // variance of the mean
        return {vol * mean, vol * vol * (var > 0 ? var : 0)};
    }

    // Pilot sample: estimate, per dimension, the variance in each midpoint half.
    long pre = static_cast<long>(calls * 0.1);
    if (pre < PRE_MIN) pre = PRE_MIN;
    std::vector<double> mid(dim);
    for (std::size_t i = 0; i < dim; ++i) mid[i] = 0.5 * (a[i] + b[i]);

    std::vector<double> sumL(dim, 0), sumL2(dim, 0), sumR(dim, 0), sumR2(dim, 0);
    std::vector<long>   cntL(dim, 0), cntR(dim, 0);
    for (long k = 0; k < pre; ++k) {
        for (std::size_t i = 0; i < dim; ++i) pt[i] = a[i] + (b[i] - a[i]) * rng.next();
        const double fx = f(pt);
        for (std::size_t i = 0; i < dim; ++i) {
            if (pt[i] < mid[i]) { sumL[i] += fx; sumL2[i] += fx * fx; ++cntL[i]; }
            else                { sumR[i] += fx; sumR2[i] += fx * fx; ++cntR[i]; }
        }
    }

    // Choose the bisection dimension minimizing sigma_left + sigma_right.
    std::size_t best_dim = 0;
    double      best_sum = 1e300;
    double      bestSL = 0, bestSR = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        auto sd = [](double s, double s2, long c) {
            if (c < 2) return 0.0;
            const double m = s / c, v = s2 / c - m * m;
            return v > 0 ? std::sqrt(v) : 0.0;
        };
        const double sL = sd(sumL[i], sumL2[i], cntL[i]);
        const double sR = sd(sumR[i], sumR2[i], cntR[i]);
        if (sL + sR < best_sum) { best_sum = sL + sR; best_dim = i; bestSL = sL; bestSR = sR; }
    }

    // Allocate remaining samples proportional to the sub-region std deviations.
    const long remain = calls - pre;
    const double denom = bestSL + bestSR;
    double frac = denom > 0 ? bestSL / denom : 0.5;
    long nL = static_cast<long>(remain * frac);
    if (nL < MIN_CALLS) nL = MIN_CALLS;
    if (nL > remain - MIN_CALLS) nL = remain - MIN_CALLS;
    const long nR = remain - nL;

    const double save = b[best_dim];
    b[best_dim] = mid[best_dim];
    const MiserResult left = miser_rec(f, a, b, nL, rng, pt);
    b[best_dim] = save;

    const double savea = a[best_dim];
    a[best_dim] = mid[best_dim];
    const MiserResult right = miser_rec(f, a, b, nR, rng, pt);
    a[best_dim] = savea;

    return {left.value + right.value, left.variance + right.variance};
}

} // namespace detail

// Integrate f over the box [a_i, b_i] using MISER with a budget of `calls` samples.
template <class F>
MiserResult miser_integrate(F f, std::vector<double> a, std::vector<double> b, long calls,
                            std::uint64_t seed = 1) {
    detail::MiserRng    rng(seed);
    std::vector<double> pt(a.size());
    return detail::miser_rec(f, a, b, calls, rng, pt);
}

} // namespace datamunge::algorithms
