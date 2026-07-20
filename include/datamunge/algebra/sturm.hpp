#pragma once

#include <datamunge/algebra/polynomial.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace datamunge::algebra {

/// @brief Builds the Sturm sequence of p: p0 = p, p1 = p', and p_(i+1) = -rem(p_(i-1), p_i)
///        until reaching a constant (or zero) polynomial. Used to count/isolate the real roots
///        of p in an interval via sign changes (Sturm's theorem). p should be square-free for
///        the sign-change count to equal the number of DISTINCT real roots exactly; callers
///        with a polynomial that may have repeated roots should first run
///        square_free_factorization() and apply Sturm's theorem to each factor.
[[nodiscard]] inline std::vector<Polynomial> sturm_sequence(const Polynomial& p) {
    std::vector<Polynomial> seq;
    seq.push_back(p);
    seq.push_back(p.derivative());
    while (!seq.back().is_zero() && seq.back().degree() > 0) {
        const Polynomial& a = seq[seq.size() - 2];
        const Polynomial& b = seq.back();
        auto [q, r] = a.divmod(b);
        (void)q;
        seq.push_back(r.negate());
    }
    return seq;
}

namespace detail {

[[nodiscard]] inline int sturm_sign_changes(const std::vector<Polynomial>& seq, double x) {
    int changes = 0;
    int prev_sign = 0;
    for (const auto& poly : seq) {
        const double v = poly.evaluate(x);
        if (v == 0.0) continue;
        const int s = v > 0 ? 1 : -1;
        if (prev_sign != 0 && s != prev_sign) ++changes;
        prev_sign = s;
    }
    return changes;
}

[[nodiscard]] inline double cauchy_root_bound(const Polynomial& p) {
    const int n = p.degree();
    if (n <= 0) return 0.0;
    const double lead = std::fabs(p.coefficient(n));
    double max_ratio = 0.0;
    for (int i = 0; i < n; ++i) max_ratio = std::max(max_ratio, std::fabs(p.coefficient(i)) / lead);
    return 1.0 + max_ratio;
}

} // namespace detail

/// @brief The number of DISTINCT real roots of p in the interval (a, b], via Sturm's theorem
///        (the difference in sign-change count between the sequence evaluated at a and at b).
[[nodiscard]] inline int sturm_root_count(const Polynomial& p, double a, double b) {
    const auto seq = sturm_sequence(p);
    return detail::sturm_sign_changes(seq, a) - detail::sturm_sign_changes(seq, b);
}

/// @brief Isolating intervals for every distinct real root of p: parallel lower/upper bound
///        vectors where each (lower[i], upper[i]) contains exactly one root. Found by bisecting
///        a starting interval (a Cauchy bound on root magnitude) driven by sturm_root_count().
struct RealRootIntervals {
    std::vector<double> lower;
    std::vector<double> upper;
};

[[nodiscard]] inline RealRootIntervals isolate_real_roots(const Polynomial& p) {
    RealRootIntervals result;
    if (p.degree() <= 0) return result;

    const double bound = detail::cauchy_root_bound(p);
    std::vector<std::pair<double, double>> stack;
    stack.emplace_back(-bound, bound);
    while (!stack.empty()) {
        auto [lo, hi] = stack.back();
        stack.pop_back();
        const int count = sturm_root_count(p, lo, hi);
        if (count == 0) continue;
        if (count == 1) {
            result.lower.push_back(lo);
            result.upper.push_back(hi);
            continue;
        }
        const double mid = (lo + hi) / 2.0;
        stack.emplace_back(lo, mid);
        stack.emplace_back(mid, hi);
    }
    return result;
}

/// @brief Refines a single isolating interval [lo, hi] (known to contain exactly one root) to
///        within `tolerance` via sign-based bisection (cheaper than re-running Sturm's theorem
///        once a root is already isolated to one per interval).
[[nodiscard]] inline double refine_root(const Polynomial& p, double lo, double hi, double tolerance = 1e-10) {
    double flo = p.evaluate(lo);
    while (hi - lo > tolerance) {
        const double mid = (lo + hi) / 2.0;
        const double fmid = p.evaluate(mid);
        if (fmid == 0.0) return mid;
        if ((fmid > 0) == (flo > 0)) {
            lo = mid;
            flo = fmid;
        } else {
            hi = mid;
        }
    }
    return (lo + hi) / 2.0;
}

/// @brief All distinct real roots of p, isolated via Sturm's theorem then refined by
///        bisection to within `tolerance`.
[[nodiscard]] inline std::vector<double> real_roots(const Polynomial& p, double tolerance = 1e-10) {
    const auto intervals = isolate_real_roots(p);
    std::vector<double> roots;
    roots.reserve(intervals.lower.size());
    for (std::size_t i = 0; i < intervals.lower.size(); ++i)
        roots.push_back(refine_root(p, intervals.lower[i], intervals.upper[i], tolerance));
    return roots;
}

} // namespace datamunge::algebra
