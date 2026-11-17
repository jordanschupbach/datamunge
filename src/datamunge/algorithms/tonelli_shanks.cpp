/// @file tonelli_shanks.cpp
/// @brief Actual prime-validated uint64 modular products and two-adic refinement.
#include <algorithm>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <datamunge/algorithms/tonelli_shanks.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private invocation-local exact modular arithmetic.
namespace {
/// @brief Actual checked root-kernel state, excluding external primality work.
struct State {
  std::uint64_t         p;          ///< Validated prime modulus.
  const TonelliOptions& options;    ///< Borrowed invocation policies.
  TonelliRun&           run;        ///< Actual result/counter producer.
  std::size_t           retained{}; ///< Actual retained record/value units.

  /// @brief Count and evaluate one exact unsigned128 modular product.
  std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (run.products == std::numeric_limits<std::size_t>::max() ||
        (options.max_products && run.products >= options.max_products))
      throw std::length_error("Tonelli products");
    ++run.products;
    return static_cast<std::uint64_t>((__uint128_t(a) * b) % p);
  }
  /// @brief Execute actual historical square-and-multiply order.
  std::uint64_t power(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1 % p;
    a %= p;
    while (e) {
      if (e & 1)
        r = mul(r, a);
      a = mul(a, a);
      e >>= 1;
    }
    return r;
  }
  /// @brief Bound actual captured record/scalar units.
  void retain(std::size_t n) {
    auto cap = std::numeric_limits<std::size_t>::max();
    if (retained > cap - n ||
        (options.max_history_values &&
         (retained > options.max_history_values || n > options.max_history_values - retained)))
      throw std::length_error("Tonelli history");
    retained += n;
  }
};
/// @brief Validate primality before shortcuts and execute actual root refinement.
TonelliRun execute(std::uint64_t n, std::uint64_t p, const TonelliOptions& o, bool capture) {
  if (!is_probable_prime(p))
    throw std::invalid_argument("Tonelli prime modulus required");
  TonelliRun r;
  r.reduced = n % p;
  n         = r.reduced;
  State s{p, o, r};
  if (!n) {
    r.result = {true, 0};
    return r;
  }
  if (p == 2) {
    r.result = {true, n};
    return r;
  }
  r.euler = s.power(n, (p - 1) / 2);
  if (r.euler != 1)
    return r;
  /// @brief Canonicalize one actual root to the smaller complementary residue.
  auto finish = [&](std::uint64_t root) { r.result = {true, std::min(root, p - root)}; };
  if (p % 4 == 3) {
    finish(s.power(n, (p + 1) / 4));
    return r;
  }
  std::uint64_t Q = p - 1, S = 0;
  while (!(Q & 1)) {
    Q >>= 1;
    ++S;
  }
  r.odd_part      = Q;
  r.twos          = S;
  std::uint64_t z = 2;
  for (;; ++z) {
    if (r.nonresidue_tests == std::numeric_limits<std::size_t>::max() ||
        (o.max_nonresidue_tests && r.nonresidue_tests >= o.max_nonresidue_tests))
      throw std::length_error("Tonelli auxiliary search");
    ++r.nonresidue_tests;
    auto e = s.power(z, (p - 1) / 2);
    if (capture) {
      s.retain(1);
      r.candidates.push_back({z, e});
    }
    if (e == p - 1)
      break;
  }
  r.nonresidue = z;
  auto M = S, c = s.power(z, Q), t = s.power(n, Q), R = s.power(n, (Q + 1) / 2);
  while (t != 1) {
    std::uint64_t i = 0, tt = t;
    TonelliStep   step;
    step.M_before = M;
    if (capture)
      step.squares.push_back(tt);
    while (tt != 1) {
      tt = s.mul(tt, tt);
      ++i;
      if (capture)
        step.squares.push_back(tt);
      if (i >= M)
        throw std::logic_error("Tonelli refinement invariant");
    }
    auto b = c;
    for (std::uint64_t j = 0; j + 1 < M - i; ++j)
      b = s.mul(b, b);
    auto oldM = M;
    M         = i;
    c         = s.mul(b, b);
    t         = s.mul(t, c);
    R         = s.mul(R, b);
    ++r.refinements;
    if (capture) {
      s.retain(1 + step.squares.size());
      step.M_before   = oldM;
      step.i          = i;
      step.b          = b;
      step.c_after    = c;
      step.t_after    = t;
      step.root_after = R;
      r.steps.push_back(std::move(step));
    }
  }
  finish(R);
  return r;
}
} // namespace
TonelliRun tonelli_shanks_run(std::uint64_t n, std::uint64_t p, const TonelliOptions& options) {
  return execute(n, p, options, false);
}
TonelliRun tonelli_shanks_trace(std::uint64_t n, std::uint64_t p, const TonelliOptions& options) {
  return execute(n, p, options, true);
}
ModSqrt tonelli_shanks(std::uint64_t n, std::uint64_t p) {
  return tonelli_shanks_run(n, p).result;
}
} // namespace datamunge::algorithms
