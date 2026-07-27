#include <datamunge/algorithms/pi_and_cordic.hpp>

#include <cmath>

namespace datamunge::algorithms {

namespace {

constexpr double kPi = 3.14159265358979323846;

// 16^e mod m by fast modular exponentiation (for the BBP digit extraction).
long long modpow16(long long e, long long m) {
    long long r = 1 % m;
    long long b = 16 % m;
    while (e > 0) {
        if (e & 1) r = (r * b) % m;
        b = (b * b) % m;
        e >>= 1;
    }
    return r;
}

// Fractional part of 16^n * sum_{k>=0} 1 / ((8k+j) 16^k): the integer-part terms are handled
// modularly (16^{n-k} mod (8k+j)) and the tail (k>n) is a fast-converging series.
double bbp_series(int j, int n) {
    double s = 0.0;
    for (int k = 0; k <= n; ++k) {
        const long long denom = 8LL * k + j;
        s += static_cast<double>(modpow16(n - k, denom)) / static_cast<double>(denom);
        s -= std::floor(s); // keep only the fractional part
    }
    for (int k = n + 1; k <= n + 24; ++k) {
        const double term = std::pow(16.0, static_cast<double>(n - k)) / static_cast<double>(8 * k + j);
        s += term;
        if (term < 1e-18) break;
    }
    return s - std::floor(s);
}

double factorial(int n) {
    double r = 1.0;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

} // namespace

int bbp_pi_hex_digit(int n) {
    double x = 4.0 * bbp_series(1, n) - 2.0 * bbp_series(4, n) - bbp_series(5, n) - bbp_series(6, n);
    x -= std::floor(x); // reduce to [0,1)
    return static_cast<int>(x * 16.0);
}

double gauss_legendre_pi(int iterations) {
    double a = 1.0;
    double b = 1.0 / std::sqrt(2.0);
    double t = 0.25;
    double p = 1.0;
    for (int i = 0; i < iterations; ++i) {
        const double a_next = 0.5 * (a + b);
        const double b_next = std::sqrt(a * b);
        t -= p * (a - a_next) * (a - a_next);
        a = a_next;
        b = b_next;
        p *= 2.0;
    }
    return (a + b) * (a + b) / (4.0 * t);
}

double chudnovsky_pi(int terms) {
    double sum = 0.0;
    for (int k = 0; k < terms; ++k) {
        const double numerator   = factorial(6 * k) * (13591409.0 + 545140134.0 * k);
        const double denominator = factorial(3 * k) * std::pow(factorial(k), 3.0) *
                                   std::pow(640320.0, 3.0 * k + 1.5);
        double term = 12.0 * numerator / denominator;
        if (k & 1) term = -term;
        sum += term;
    }
    return 1.0 / sum;
}

SinCos cordic_sincos(double theta, int iterations) {
    // Range-reduce theta into [-pi/2, pi/2], within CORDIC's convergence radius (~1.743).
    double z    = std::remainder(theta, 2.0 * kPi); // now in [-pi, pi]
    bool   flip = false;
    if (z > kPi / 2.0) {
        z -= kPi;
        flip = true;
    } else if (z < -kPi / 2.0) {
        z += kPi;
        flip = true;
    }

    double x    = 1.0;
    double y    = 0.0;
    double gain = 1.0;
    double p2   = 1.0; // 2^{-i}
    for (int i = 0; i < iterations; ++i) {
        const double angle = std::atan(p2); // the precomputed arctangent table
        const double d     = (z >= 0.0) ? 1.0 : -1.0;
        const double nx    = x - d * y * p2; // shift-add pseudo-rotation
        const double ny    = y + d * x * p2;
        x = nx;
        y = ny;
        z -= d * angle;
        gain *= 1.0 / std::sqrt(1.0 + p2 * p2);
        p2 *= 0.5;
    }
    SinCos out{y * gain, x * gain};
    if (flip) {
        out.sin = -out.sin;
        out.cos = -out.cos;
    }
    return out;
}

} // namespace datamunge::algorithms
