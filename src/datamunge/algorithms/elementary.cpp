#include <datamunge/algorithms/elementary.hpp>

#include <cmath>
#include <vector>

namespace datamunge::algorithms {

double kahan_sum(const std::vector<double>& xs) {
    double sum = 0.0;
    double c   = 0.0; // running compensation for lost low-order bits
    for (double x : xs) {
        const double y = x - c;       // recover what was dropped last time
        const double t = sum + y;     // ... but this addition loses low bits of y
        c              = (t - sum) - y; // (t - sum) recovers the high part; subtracting y leaves the lost part
        sum            = t;
    }
    return sum;
}

double newton_raphson_division(double numerator, double denominator) {
    const double sign = denominator < 0.0 ? -1.0 : 1.0;
    double       d    = std::fabs(denominator);

    // Scale d into [0.5, 1) so a fixed linear seed converges: d = m * 2^e.
    int          e = 0;
    const double m = std::frexp(d, &e); // m in [0.5, 1)

    // Optimal linear initial guess for 1/m on [0.5, 1): x0 = 48/17 - 32/17 * m.
    double x = 48.0 / 17.0 - (32.0 / 17.0) * m;
    // Newton reciprocal iteration x <- x (2 - m x); doubles correct bits each step.
    for (int i = 0; i < 60; ++i) {
        const double xn = x * (2.0 - m * x);
        if (std::fabs(xn - x) <= 1e-17 * std::fabs(xn)) {
            x = xn;
            break;
        }
        x = xn;
    }
    const double recip = sign * std::ldexp(x, -e); // 1/denominator
    return numerator * recip;
}

double nth_root(double a, int n) {
    if (n <= 1) return a;
    if (a == 0.0) return 0.0;

    // Initial guess from the binary exponent: a = m * 2^e  =>  a^(1/n) ~ 2^(e/n).
    int e = 0;
    (void)std::frexp(a, &e);
    double x = std::ldexp(1.0, e / n);
    if (x <= 0.0) x = 1.0;

    // Newton on x^n - a = 0:  x <- ((n-1) x + a / x^(n-1)) / n.
    for (int i = 0; i < 100; ++i) {
        double pow_nm1 = 1.0;
        for (int k = 0; k < n - 1; ++k) pow_nm1 *= x; // x^(n-1)
        const double xn = ((n - 1) * x + a / pow_nm1) / n;
        if (std::fabs(xn - x) <= 1e-15 * std::fabs(xn)) {
            x = xn;
            break;
        }
        x = xn;
    }
    return x;
}

double alpha_max_beta_min(double x, double y) {
    const double ax = std::fabs(x);
    const double ay = std::fabs(y);
    const double hi = ax > ay ? ax : ay;
    const double lo = ax > ay ? ay : ax;
    // Coefficients that minimize the peak relative error (~3.96%).
    constexpr double alpha = 0.960433870103;
    constexpr double beta  = 0.397824734759;
    return alpha * hi + beta * lo;
}

} // namespace datamunge::algorithms
