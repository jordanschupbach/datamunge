/// @file sieve_of_atkin.cpp
/// @brief Sieve of Atkin with operation counts, Eratosthenes counts, li(x) and Riemann's R(x).

#include <datamunge/algorithms/primality.hpp>
#include <datamunge/algorithms/sieve_of_atkin.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief The shared sieve kernel and numeric helpers.
namespace {

/// @brief Largest y >= 0 with y * y <= v.
std::uint64_t isqrt(std::uint64_t v) {
    auto r = static_cast<std::uint64_t>(std::sqrt(static_cast<long double>(v)));
    while (r * r > v) --r;
    while ((r + 1) * (r + 1) <= v) ++r;
    return r;
}

/// @brief Sieve of Atkin over [0, limit] with each form restricted to its region.
/// @param sieve  receives the prime flags for n >= 5 (sized limit + 1).
/// @param work   operation counts, or nullptr.
void atkin_kernel(std::uint64_t limit, std::vector<bool>& sieve, SieveWork* work) {
    sieve.assign(limit + 1, false);
    std::uint64_t points = 0, toggles = 0, removals = 0;
    // 4x^2 + y^2 <= limit
    for (std::uint64_t x = 1; 4 * x * x < limit; ++x) {
        const std::uint64_t ymax = isqrt(limit - 4 * x * x);
        for (std::uint64_t y = 1; y <= ymax; ++y) {
            ++points;
            const std::uint64_t n = 4 * x * x + y * y, r = n % 12;
            if (r == 1 || r == 5) {
                sieve[n] = !sieve[n];
                ++toggles;
            }
        }
    }
    // 3x^2 + y^2 <= limit
    for (std::uint64_t x = 1; 3 * x * x < limit; ++x) {
        const std::uint64_t ymax = isqrt(limit - 3 * x * x);
        for (std::uint64_t y = 1; y <= ymax; ++y) {
            ++points;
            const std::uint64_t n = 3 * x * x + y * y;
            if (n % 12 == 7) {
                sieve[n] = !sieve[n];
                ++toggles;
            }
        }
    }
    // 3x^2 - y^2 <= limit with x > y >= 1: y >= sqrt(3x^2 - limit) when 3x^2 > limit.
    for (std::uint64_t x = 2; 2 * x * x + 2 * x - 1 <= limit; ++x) {  // y = x - 1 gives the smallest value
        const std::uint64_t three = 3 * x * x;
        std::uint64_t       ymin  = 1;
        if (three > limit) {
            ymin = isqrt(three - limit);
            if (ymin * ymin < three - limit) ++ymin;
            if (ymin == 0) ymin = 1;
        }
        for (std::uint64_t y = ymin; y < x; ++y) {
            ++points;
            const std::uint64_t n = three - y * y;
            if (n % 12 == 11) {
                sieve[n] = !sieve[n];
                ++toggles;
            }
        }
    }
    // Clear multiples of squares of primes: a number with a square factor can pass the parity test.
    for (std::uint64_t r = 5; r * r <= limit; ++r)
        if (sieve[r])
            for (std::uint64_t m = r * r; m <= limit; m += r * r) {
                sieve[m] = false;
                ++removals;
            }
    if (work) {
        work->lattice_points  = points;
        work->toggles         = toggles;
        work->square_removals = removals;
    }
}

/// @brief Riemann zeta at s >= 2 by Euler-Maclaurin summation (error well below 1e-15).
double zeta(double s) {
    constexpr int N = 20;
    double        sum = 0.0;
    for (int n = 1; n < N; ++n) sum += std::pow(n, -s);
    const double n = N;
    sum += std::pow(n, 1.0 - s) / (s - 1.0) + 0.5 * std::pow(n, -s) + s * std::pow(n, -s - 1.0) / 12.0 -
           s * (s + 1.0) * (s + 2.0) * std::pow(n, -s - 3.0) / 720.0;
    return sum;
}

}  // namespace

std::vector<std::uint64_t> atkin_primes(std::uint64_t limit) {
    std::vector<std::uint64_t> primes;
    if (limit < 2) return primes;
    std::vector<bool> sieve;
    atkin_kernel(limit, sieve, nullptr);
    primes.push_back(2);
    if (limit >= 3) primes.push_back(3);
    for (std::uint64_t n = 5; n <= limit; ++n)
        if (sieve[n]) primes.push_back(n);
    return primes;
}

std::uint64_t atkin_form_representations(std::uint64_t n, AtkinForm form) {
    std::uint64_t count = 0;
    switch (form) {
        case AtkinForm::FourXSquaredPlusYSquared:
            for (std::uint64_t x = 1; 4 * x * x < n; ++x) {
                const std::uint64_t rest = n - 4 * x * x, y = isqrt(rest);
                count += (y > 0 && y * y == rest);
            }
            break;
        case AtkinForm::ThreeXSquaredPlusYSquared:
            for (std::uint64_t x = 1; 3 * x * x < n; ++x) {
                const std::uint64_t rest = n - 3 * x * x, y = isqrt(rest);
                count += (y > 0 && y * y == rest);
            }
            break;
        case AtkinForm::ThreeXSquaredMinusYSquared:
            // 3x^2 - y^2 = n with x > y >= 1 forces 2x^2 < n + 1, i.e. x <= sqrt(n / 2).
            for (std::uint64_t x = 1; 2 * x * x <= n + 1; ++x) {
                if (3 * x * x <= n) continue;
                const std::uint64_t rest = 3 * x * x - n, y = isqrt(rest);
                count += (y > 0 && y < x && y * y == rest);
            }
            break;
    }
    return count;
}

bool atkin_form_for(std::uint64_t n, AtkinForm& form) {
    switch (n % 12) {
        case 1:
        case 5: form = AtkinForm::FourXSquaredPlusYSquared; return true;
        case 7: form = AtkinForm::ThreeXSquaredPlusYSquared; return true;
        case 11: form = AtkinForm::ThreeXSquaredMinusYSquared; return true;
        default: return false;
    }
}

SieveWork atkin_sieve_work(std::uint64_t limit) {
    SieveWork work;
    work.limit = limit;
    if (limit < 2) return work;
    std::vector<bool> sieve;
    atkin_kernel(limit, sieve, &work);
    work.primes = 1 + (limit >= 3);
    for (std::uint64_t n = 5; n <= limit; ++n) work.primes += sieve[n];
    return work;
}

SieveWork eratosthenes_sieve_work(std::uint64_t limit) {
    SieveWork work;
    work.limit = limit;
    if (limit < 2) return work;
    std::vector<bool> composite(limit + 1, false);
    for (std::uint64_t p = 2; p * p <= limit; ++p) {
        if (composite[p]) continue;
        for (std::uint64_t m = p * p; m <= limit; m += p) {
            composite[m] = true;
            ++work.crossings;
        }
    }
    for (std::uint64_t n = 2; n <= limit; ++n) work.primes += !composite[n];
    return work;
}

double logarithmic_integral(double x) {
    if (!(x > 1.0)) throw std::invalid_argument("logarithmic_integral: x must exceed 1");
    // Ramanujan's series: li(x) = gamma + ln ln x + sqrt(x) sum_n (-1)^(n-1) (ln x)^n / (n! 2^(n-1))
    //                             * sum_{k=0}^{floor((n-1)/2)} 1 / (2k + 1).
    constexpr double gamma = 0.57721566490153286061;
    const double     L     = std::log(x);
    double           sum = 0.0, term = 1.0, inner = 0.0;
    for (int n = 1; n < 400; ++n) {
        term *= L / n;  // (ln x)^n / n!
        if ((n - 1) % 2 == 0) inner += 1.0 / (n);  // adds 1/(2k+1) with 2k+1 = n
        const double t = (n % 2 == 1 ? 1.0 : -1.0) * term / std::ldexp(1.0, n - 1) * inner;
        sum += t;
        if (std::abs(t) < 1e-17 * std::abs(sum) && n > 2 * L) break;
    }
    return gamma + std::log(L) + std::sqrt(x) * sum;
}

double riemann_prime_count(double x) {
    if (!(x >= 1.0)) throw std::invalid_argument("riemann_prime_count: x must be at least 1");
    const double L   = std::log(x);
    double       sum = 1.0, power = 1.0;  // power = (ln x)^k / k!
    for (int k = 1; k < 1000; ++k) {
        power *= L / k;
        const double t = power / (k * zeta(k + 1.0));
        sum += t;
        if (t < 1e-17 * sum && k > L) break;
    }
    return sum;
}

}  // namespace datamunge::algorithms
