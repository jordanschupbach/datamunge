#include <datamunge/algorithms/ternary_search.hpp>

#include <functional>

namespace datamunge::algorithms {

namespace {

// Shared ternary loop; `keep_left_when` decides, from f(m1) and f(m2), whether the peak/valley lies
// in [lo, m2] (drop the right third) or in [m1, hi] (drop the left third).
TernaryResult ternary(const std::function<double(double)>& f, double lo, double hi, double tol,
                      int max_iter, const std::function<bool(double, double)>& keep_left) {
    TernaryResult r;
    int           it = 0;
    for (; it < max_iter && (hi - lo) > tol; ++it) {
        const double m1 = lo + (hi - lo) / 3.0;
        const double m2 = hi - (hi - lo) / 3.0;
        if (keep_left(f(m1), f(m2)))
            hi = m2; // extremum is in [lo, m2]
        else
            lo = m1; // extremum is in [m1, hi]
    }
    r.x          = 0.5 * (lo + hi);
    r.value      = f(r.x);
    r.iterations = it;
    return r;
}

} // namespace

TernaryResult ternary_search_max(const std::function<double(double)>& f, double lo, double hi,
                                 double tol, int max_iter) {
    // For a maximum, if f(m1) >= f(m2) the peak cannot be in the right third.
    return ternary(f, lo, hi, tol, max_iter, [](double f1, double f2) { return f1 >= f2; });
}

TernaryResult ternary_search_min(const std::function<double(double)>& f, double lo, double hi,
                                 double tol, int max_iter) {
    // For a minimum, if f(m1) <= f(m2) the valley cannot be in the right third.
    return ternary(f, lo, hi, tol, max_iter, [](double f1, double f2) { return f1 <= f2; });
}

} // namespace datamunge::algorithms
