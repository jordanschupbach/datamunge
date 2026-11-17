/// @file pollard_rho.cpp
/// @brief Exact square-plus-c Brent producer with actual-site histories and cumulative evaluation admission.
#include <algorithm>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <datamunge/algorithms/pollard_rho.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
namespace datamunge::algorithms {
/// @brief Shared exact polynomial and producing split/factorization machinery.
namespace {
/// @brief Exact full-width modular square-plus-constant, including addition carry.
std::uint64_t polynomial(std::uint64_t v, std::uint64_t c, std::uint64_t n) {
  return static_cast<std::uint64_t>((static_cast<unsigned __int128>(mul_mod(v, v, n)) + c) % n);
}
/// @brief Increment selected work without representable counter wrap.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("rho counter range");
  ++n;
}
/// @brief Borrowed actual work/history for one synchronous complete producing call.
struct Context {
  RhoFactorOptions  options;    ///< Owned selected policies.
  RhoFactorWork&    work;       ///< Borrowed caller-owned complete-run counters.
  RhoFactorHistory* history;    ///< Optional caller-owned output; never retained beyond this call.
  std::size_t       records{0}; ///< Actual retained records already admitted.

  /// @brief Admit a historical record before copying its producing values.
  void record() {
    if (records == std::numeric_limits<std::size_t>::max() ||
        (options.max_trace_records && records >= options.max_trace_records))
      throw std::length_error("rho history policy");
    ++records;
  }
  /// @brief Perform one actual admitted polynomial evaluation.
  std::uint64_t evaluate(std::uint64_t v, std::uint64_t c, std::uint64_t n, RhoFactorPhase phase) {
    if (work.evaluations == std::numeric_limits<std::size_t>::max() ||
        (options.max_evaluations && work.evaluations >= options.max_evaluations))
      throw std::length_error("rho evaluation policy");
    if (history)
      record();
    const auto next = polynomial(v, c, n);
    ++work.evaluations;
    if (history)
      history->evaluations.push_back({n, c, v, next, phase});
    return next;
  }
  /// @brief Call canonical deterministic primality and count its actual call boundary.
  bool prime(std::uint64_t n) {
    increment(work.primality_calls);
    return is_probable_prime(n);
  }
  /// @brief Perform and optionally retain an actual normal/fallback gcd boundary.
  std::uint64_t gcd(
      std::uint64_t n,
      std::uint64_t c,
      std::uint64_t x,
      std::uint64_t y,
      std::uint64_t q,
      std::uint64_t steps,
      bool          fallback) {
    if (history)
      record();
    increment(work.gcd_calls);
    const auto d = std::gcd(q, n);
    if (history)
      history->gcds.push_back({n, c, x, y, q, d, steps, fallback});
    return d;
  }
};
/// @brief Original constant/batch/doubling split loop using exact full-width polynomial arithmetic.
std::uint64_t split(std::uint64_t n, Context& ctx) {
  if (n < 2)
    return n;
  if (n % 2 == 0)
    return 2;
  if (ctx.prime(n))
    return n;
  for (std::uint64_t c = 1; c < n; ++c) {
    increment(ctx.work.attempts);
    std::uint64_t x = 2, y = 2, ys = 2, q = 1, d = 1, r = 1;
    do {
      x = y;
      for (std::uint64_t i = 0; i < r; ++i)
        y = ctx.evaluate(y, c, n, RhoFactorPhase::Advance);
      std::uint64_t k = 0;
      while (k < r && d == 1) {
        ys               = y;
        const auto steps = std::min<std::uint64_t>(128, r - k);
        for (std::uint64_t i = 0; i < steps; ++i) {
          y = ctx.evaluate(y, c, n, RhoFactorPhase::Batch);
          q = mul_mod(q, x < y ? y - x : x - y, n);
          increment(ctx.work.batch_products);
        }
        d = ctx.gcd(n, c, x, y, q, steps, false);
        k += steps;
      }
      increment(ctx.work.radius_doublings);
      if (r > std::numeric_limits<std::uint64_t>::max() / 2) {
        if (d == 1)
          throw std::overflow_error("rho radius cannot advance");
        r = 0;
      } else
        r *= 2;
    } while (d == 1);
    if (d == n) {
      do {
        ys = ctx.evaluate(ys, c, n, RhoFactorPhase::Fallback);
        d  = ctx.gcd(n, c, x, ys, x < ys ? ys - x : x - ys, 1, true);
      } while (d == 1);
    }
    if (d > 1 && d < n)
      return d;
  }
  throw std::runtime_error("rho exhausted native constants");
}
/// @brief Recursively split actual odd subproblems under one cumulative context.
void recursive(std::uint64_t n, std::vector<std::uint64_t>& out, Context& ctx) {
  if (n == 1)
    return;
  if (ctx.prime(n)) {
    out.push_back(n);
    return;
  }
  const auto d = split(n, ctx);
  if (d <= 1 || d >= n || n % d)
    throw std::runtime_error("rho proper-split invariant");
  if (ctx.history) {
    ctx.record();
    ctx.history->splits.push_back({n, d});
  }
  recursive(d, out, ctx);
  recursive(n / d, out, ctx);
}
/// @brief Complete native twos-first factorization with optional actual producing histories.
FactorizationTrace complete(std::uint64_t n, RhoFactorOptions options, bool capture) {
  if (n == 0)
    throw std::invalid_argument("factorize: zero has no prime factorization");
  FactorizationTrace t;
  Context            ctx{options, t.work, capture ? &t.history : nullptr};
  while (n % 2 == 0) {
    t.factors.push_back(2);
    n /= 2;
    increment(t.work.peeled_twos);
  }
  recursive(n, t.factors, ctx);
  std::sort(t.factors.begin(), t.factors.end());
  return t;
}
} // namespace
std::uint64_t pollard_rho_factor(std::uint64_t n) {
  RhoFactorWork work;
  Context       ctx{{}, work, nullptr};
  return split(n, ctx);
}
RhoFactorRun pollard_rho_factor_run(std::uint64_t n, RhoFactorOptions options) {
  RhoFactorRun r;
  Context      ctx{options, r.work, nullptr};
  r.factor = split(n, ctx);
  return r;
}
RhoFactorTrace pollard_rho_factor_trace(std::uint64_t n, RhoFactorOptions options) {
  RhoFactorTrace t;
  Context        ctx{options, t.work, &t.history};
  t.factor = split(n, ctx);
  return t;
}
std::vector<std::uint64_t> factorize(std::uint64_t n) {
  return complete(n, {}, false).factors;
}
FactorizationRun factorize_run(std::uint64_t n, RhoFactorOptions options) {
  auto t = complete(n, options, false);
  return {std::move(t.factors), t.work};
}
FactorizationTrace factorize_trace(std::uint64_t n, RhoFactorOptions options) {
  return complete(n, options, true);
}
RhoPolynomialWalk rho_polynomial_walk(
    std::uint64_t n,
    std::uint64_t c,
    std::uint64_t start,
    std::size_t   count,
    std::uint64_t projection,
    std::size_t   cap) {
  if (n < 2)
    throw std::invalid_argument("rho orbit modulus >=2 required");
  if (projection == 0)
    projection = n;
  if (projection < 2 || n % projection)
    throw std::invalid_argument("rho orbit projection must divide modulus");
  if (count > std::numeric_limits<std::size_t>::max() / 2 || (cap && 2 * count > cap))
    throw std::length_error("rho orbit history policy");
  RhoPolynomialWalk                              t;
  std::unordered_map<std::uint64_t, std::size_t> seen;
  std::uint64_t                                  x = start % n;
  c %= n;
  for (std::size_t i = 0; i < count; ++i) {
    t.states.push_back(x);
    const auto residue = x % projection;
    t.projected.push_back(residue);
    if (!t.repeated) {
      const auto [it, inserted] = seen.emplace(residue, i);
      if (!inserted) {
        t.repeated   = true;
        t.cycle_head = it->second;
        t.period     = i - it->second;
      }
    }
    if (i + 1 < count)
      x = polynomial(x, c, n);
  }
  return t;
}
} // namespace datamunge::algorithms
