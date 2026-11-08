/// @file number_theory_extra.cpp
/// @brief Elementary factorization and a shared exact bounded Fermat kernel.
#include <cstdint>
#include <datamunge/algorithms/integer_arithmetic.hpp>
#include <datamunge/algorithms/number_theory_extra.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Internal modular arithmetic and same-pass Fermat implementation.
namespace {

/// @brief Euclidean greatest common divisor of unsigned inputs.
/// @param a First value.
/// @param b Second value.
/// @return Nonnegative greatest common divisor; zero for both zero.
std::uint64_t gcd_u(std::uint64_t a, std::uint64_t b) {
  while (b) {
    const std::uint64_t t = a % b;
    a                     = b;
    b                     = t;
  }
  return a;
}

/// @brief Multiply modulo a positive modulus using exact wider arithmetic.
/// @param a First factor.
/// @param b Second factor.
/// @param m Positive modulus.
/// @return Product reduced modulo m.
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
  return static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) % m);
}

/// @brief Binary modular powering for the Pollard algorithm.
/// @param a Base.
/// @param e Unsigned exponent.
/// @param m Positive modulus.
/// @return a to e modulo m.
std::uint64_t powmod(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
  std::uint64_t r = 1 % m;
  a %= m;
  while (e) {
    if (e & 1)
      r = mulmod(r, a, m);
    a = mulmod(a, a, m);
    e >>= 1;
  }
  return r;
}

/// @brief Run exact square tests, optionally admitting/storing actual rows.
/// @param n Input at least two, checked here before all fast paths.
/// @param max_trials Positive actual-test budget, checked here.
/// @param trace Optional caller-owned history under construction; null stores none.
/// @param max_entries Optional whole-row quota; zero disables quota.
/// @return Actual bounded outcome; throws on invalid controls or capture failure.
FermatOutcome fermat_kernel(
    std::uint64_t n,
    std::size_t   max_trials,
    FermatTrace*  trace,
    std::size_t   max_entries) {
  if (n < 2 || !max_trials)
    throw std::invalid_argument("Fermat requires n>=2 and a positive trial budget");
  FermatOutcome out;
  out.n               = n;
  out.returned_factor = n;
  if (n % 2 == 0) {
    out.returned_factor = 2;
    out.cofactor        = n / 2;
    out.proper_split    = n > 2;
    out.stop            = FermatStop::Even;
    return out;
  }
  auto a = integer_sqrt_u64(n);
  if (static_cast<unsigned __int128>(a) * a < n)
    ++a;
  out.start_a = a;
  for (;;) {
    if (trace && ((max_entries && trace->rows.size() >= max_entries) ||
                  trace->rows.size() == trace->rows.max_size()))
      throw std::length_error("Fermat capture record limit exceeded");
    const auto difference = static_cast<unsigned __int128>(a) * a - n;
    const auto b          = integer_sqrt_u128(difference);
    const bool square     = static_cast<unsigned __int128>(b) * b == difference;
    if (trace) {
      trace->rows.push_back({a, difference, b, square});
      ++trace->captured_entries;
    }
    ++out.trials;
    out.last_a = a;
    out.last_b = b;
    if (square) {
      const auto f        = a - b;
      out.proper_split    = f > 1 && f < n;
      out.returned_factor = out.proper_split ? f : n;
      out.cofactor        = out.proper_split ? a + b : 1;
      out.stop            = out.proper_split ? FermatStop::Factor : FermatStop::TrivialPair;
      return out;
    }
    if (out.trials == max_trials)
      return out;
    // Every odd input has its trivial pair at a=(n+1)/2<=2^63.
    // It is reached before either the uint64 a or size_t trial count can wrap.
    ++a;
    ++out.increments;
  }
}

} // namespace

std::vector<std::uint64_t> trial_division_factorize(std::uint64_t n) {
  std::vector<std::uint64_t> factors;
  while (n % 2 == 0) {
    factors.push_back(2);
    n /= 2;
  }
  for (std::uint64_t d = 3; d * d <= n; d += 2)
    while (n % d == 0) {
      factors.push_back(d);
      n /= d;
    }
  if (n > 1)
    factors.push_back(n);
  return factors;
}

std::uint64_t fermat_factor(std::uint64_t n) {
  return fermat_factor_work(n).returned_factor;
}
FermatOutcome fermat_factor_work(std::uint64_t n, std::size_t max_trials) {
  return fermat_kernel(n, max_trials, nullptr, 0);
}
FermatTrace fermat_factor_trace(std::uint64_t n, std::size_t max_trials, std::size_t max_entries) {
  FermatTrace trace;
  trace.outcome = fermat_kernel(n, max_trials, &trace, max_entries);
  return trace;
}
std::string_view fermat_stop_name(FermatStop stop) noexcept {
  switch (stop) {
    case FermatStop::Even:
      return "even";
    case FermatStop::Factor:
      return "factor";
    case FermatStop::TrivialPair:
      return "trivial pair";
    case FermatStop::TrialLimit:
      return "trial limit";
  }
  return "unknown";
}

std::uint64_t pollard_p_minus_1(std::uint64_t n, std::uint64_t bound) {
  if (n % 2 == 0)
    return 2;
  std::uint64_t a = 2;
  for (std::uint64_t j = 2; j <= bound; ++j) {
    a = powmod(a, j, n);
    // Test the gcd at every step: we want the *first* j where one prime's order divides the
    // accumulated exponent but the other's does not (a later j may make both smooth -> g == n).
    const std::uint64_t g = gcd_u(a == 0 ? n : a - 1, n);
    if (g > 1 && g < n)
      return g;
    if (g == n)
      return 0; // over-smooth: all factors collapsed at once
  }
  return 0;
}

} // namespace datamunge::algorithms
