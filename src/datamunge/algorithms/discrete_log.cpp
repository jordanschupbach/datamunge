/// @file discrete_log.cpp
/// @brief Shared exact fixed-width logarithm producers with actual-site step/digit observation.
#include <algorithm>
#include <datamunge/algorithms/discrete_log.hpp>
#include <datamunge/algorithms/extended_euclidean.hpp>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Shared finite group validation, selected primitive accounting and actual observers.
namespace {
/// @brief One producing run's checked selected-call and retained-record admission.
struct Context {
  DiscreteLogOptions options;    ///< Owned selected policies.
  DiscreteLogWork&   work;       ///< Borrowed work object owned by this synchronous producing call.
  std::size_t        records{0}; ///< Actual retained records admitted so far.

  /// @brief Admit a selected primitive call/check before executing it.
  void action(std::size_t& counter) {
    if (work.operations == std::numeric_limits<std::size_t>::max() ||
        (options.max_operations && work.operations >= options.max_operations))
      throw std::length_error("discrete-log selected operation policy");
    ++work.operations;
    ++counter;
  }
  /// @brief Admit one actual historical record before copying it.
  void record() {
    if (records == std::numeric_limits<std::size_t>::max() ||
        (options.max_trace_records && records >= options.max_trace_records))
      throw std::length_error("discrete-log history policy");
    ++records;
  }
  /// @brief Call the canonical modular power and count its actual call boundary.
  std::uint64_t power(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    action(work.power_calls);
    return mod_pow(a, b, m);
  }
  /// @brief Call canonical exact modular multiplication outside powering internals.
  std::uint64_t product(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    action(work.modular_products);
    return mul_mod(a, b, m);
  }
  /// @brief Call canonical checked modular inverse in the validated signed range.
  std::uint64_t inverse(std::uint64_t a, std::uint64_t m) {
    action(work.inverse_calls);
    const auto value = modular_inverse(static_cast<std::int64_t>(a), static_cast<std::int64_t>(m));
    if (value < 0)
      throw std::runtime_error("discrete-log inverse unexpectedly absent");
    return static_cast<std::uint64_t>(value);
  }
};
/// @brief Validate complete scalar field/order/membership inputs before remainders or walks.
void validate(
    std::uint64_t& g,
    std::uint64_t& h,
    std::uint64_t  p,
    std::uint64_t  order,
    Context&       ctx) {
  if (p < 2 || order == 0 ||
      order > static_cast<std::uint64_t>(std::numeric_limits<long long>::max()) ||
      (p - 1) % order != 0)
    throw std::invalid_argument("discrete-log modulus/order range");
  ctx.action(ctx.work.primality_calls);
  if (!is_probable_prime(p))
    throw std::invalid_argument("discrete-log prime modulus required");
  g %= p;
  h %= p;
  if (g == 0 || h == 0 || ctx.power(g, order, p) != 1 || ctx.power(h, order, p) != 1)
    throw std::invalid_argument("discrete-log subgroup membership");
}
/// @brief Reduced unsigned subtraction without signed narrowing or overflow.
std::uint64_t submod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
  return a >= b ? a - b : m - (b - a);
}
/// @brief Perform one actual original three-way group update and optionally retain it.
void step(
    RhoLogState&             s,
    std::uint64_t            g,
    std::uint64_t            h,
    std::uint64_t            p,
    std::uint64_t            order,
    Context&                 ctx,
    std::vector<RhoLogStep>* history,
    std::size_t              attempt,
    std::size_t              iteration,
    bool                     hare) {
  if (history)
    ctx.record();
  const auto before = s;
  const auto branch = static_cast<unsigned>(s.value % 3);
  switch (branch) {
    case 0:
      s.value = ctx.product(s.value, h, p);
      s.b     = (s.b + 1) % order;
      break;
    case 1:
      s.value = ctx.product(s.value, s.value, p);
      s.a     = (2 * s.a) % order;
      s.b     = (2 * s.b) % order;
      break;
    default:
      s.value = ctx.product(s.value, g, p);
      s.a     = (s.a + 1) % order;
      break;
  }
  ++ctx.work.walk_updates;
  if (history)
    history->push_back({attempt, iteration, hare, branch, before, s});
}
/// @brief One actual initial invariant state from explicit exponents.
RhoLogState initial(
    std::uint64_t g,
    std::uint64_t h,
    std::uint64_t p,
    std::uint64_t order,
    std::uint64_t a,
    std::uint64_t b,
    Context&      ctx) {
  a %= order;
  b %= order;
  const auto ga = ctx.power(g, a, p), hb = ctx.power(h, b, p);
  return {ctx.product(ga, hb, p), a, b};
}
/// @brief Shared native restarted rho producer; all retained updates occur at the producing sites.
RhoLogTrace rho(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions options,
    bool               capture) {
  RhoLogTrace t;
  auto&       r = t.run;
  Context     ctx{options, r.work};
  validate(g, h, p, order, ctx);
  if (h == 1) {
    r.exponent = 0;
    r.found    = true;
    return t;
  }
  std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^ (g * 1000003ULL + h));
  const auto      limit = order > (std::numeric_limits<std::uint64_t>::max() - 64) / 8
                              ? std::numeric_limits<std::uint64_t>::max()
                              : 8 * order + 64;
  for (std::size_t attempt = 0; attempt < options.max_attempts; ++attempt) {
    ++r.work.attempts;
    const auto a = rng() % order, b = rng() % order;
    auto       slow = initial(g, h, p, order, a, b, ctx), fast = slow;
    for (std::uint64_t i = 0; i < limit; ++i) {
      step(slow, g, h, p, order, ctx, capture ? &t.steps : nullptr, attempt, i + 1, false);
      step(fast, g, h, p, order, ctx, capture ? &t.steps : nullptr, attempt, i + 1, true);
      step(fast, g, h, p, order, ctx, capture ? &t.steps : nullptr, attempt, i + 1, true);
      if (slow.value != fast.value)
        continue;
      ++r.work.collisions;
      RhoLogCollision collision{attempt, static_cast<std::size_t>(i + 1), slow, fast};
      collision.numerator   = submod(slow.a, fast.a, order);
      collision.denominator = submod(fast.b, slow.b, order);
      if (collision.denominator) {
        collision.gcd        = std::gcd(collision.denominator, order);
        collision.compatible = collision.numerator % collision.gcd == 0;
        if (collision.compatible) {
          const auto m              = order / collision.gcd;
          collision.reduced_modulus = m;
          const auto rr             = (collision.denominator / collision.gcd) % m,
                     nn             = (collision.numerator / collision.gcd) % m;
          const auto inv            = ctx.inverse(rr, m);
          const auto x0             = ctx.product(nn, inv, m);
          for (std::uint64_t k = 0; k < collision.gcd; ++k) {
            const auto candidate = x0 + k * m;
            ctx.action(r.work.candidate_tests);
            if (ctx.power(g, candidate, p) == h) {
              r.exponent         = static_cast<long long>(candidate);
              r.found            = true;
              collision.resolved = true;
              break;
            }
          }
        }
      }
      if (capture) {
        ctx.record();
        t.collisions.push_back(collision);
      }
      if (r.found)
        return t;
      break;
    }
  }
  return t;
}
/// @brief Factor a positive declared order using the original ascending trial division with bounded checks.
std::vector<PohligFactor> factors(std::uint64_t n, Context& ctx) {
  std::vector<PohligFactor> out;
  for (std::uint64_t d = 2; d <= n / d; ++d) {
    ctx.action(ctx.work.factor_trials);
    if (n % d == 0) {
      unsigned      e     = 0;
      std::uint64_t power = 1;
      do {
        n /= d;
        power *= d;
        ++e;
      } while (n % d == 0);
      out.push_back({d, e, power});
    }
  }
  if (n > 1)
    out.push_back({n, 1, n});
  return out;
}
/// @brief Original linear candidate digit search, counting actual comparisons and modular products.
long long small_log(
    std::uint64_t base,
    std::uint64_t target,
    std::uint64_t p,
    std::uint64_t q,
    Context&      ctx) {
  std::uint64_t cur = 1;
  for (std::uint64_t j = 0; j < q; ++j) {
    ctx.action(ctx.work.small_log_tests);
    if (cur == target)
      return static_cast<long long>(j);
    cur = ctx.product(cur, base, p);
  }
  return -1;
}
/// @brief Shared actual factor/digit/CRT Pohlig-Hellman producer.
PohligHellmanTrace pohlig(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions options,
    bool               capture) {
  PohligHellmanTrace t;
  auto&              r = t.run;
  Context            ctx{options, r.work};
  validate(g, h, p, order, ctx);
  const auto decomposition = factors(order, ctx);
  for (const auto& f : decomposition)
    if (ctx.power(g, order / f.prime, p) == 1)
      throw std::invalid_argument("Pohlig-Hellman requires exact declared base order");
  if (capture)
    t.factors = decomposition;
  std::uint64_t crt_x = 0, crt_mod = 1;
  for (const auto& f : decomposition) {
    const auto    q     = f.prime;
    const auto    gamma = ctx.power(g, order / q, p), ginv = ctx.power(g, order - 1, p);
    std::uint64_t xi = 0, qk = 1;
    for (unsigned k = 0; k < f.exponent; ++k) {
      const auto before       = xi;
      const auto gpow         = ctx.power(ginv, xi, p);
      const auto target       = ctx.power(ctx.product(h, gpow, p), order / (qk * q), p);
      const auto tests_before = r.work.small_log_tests;
      const auto digit        = small_log(gamma, target, p, q, ctx);
      if (digit >= 0)
        xi += static_cast<std::uint64_t>(digit) * qk;
      if (capture) {
        ctx.record();
        t.digits.push_back(
            {q, k, qk, before, target, digit, r.work.small_log_tests - tests_before, xi});
      }
      if (digit < 0)
        return t;
      qk *= q;
    }
    const auto inv        = ctx.inverse(crt_mod % f.power, f.power);
    const auto offset     = submod(xi % f.power, crt_x % f.power, f.power);
    const auto multiplier = ctx.product(offset, inv, f.power);
    const auto after      = static_cast<unsigned __int128>(crt_x) +
                       static_cast<unsigned __int128>(crt_mod) * multiplier;
    const auto combined = static_cast<unsigned __int128>(crt_mod) * f.power;
    if (after >= combined || combined > order)
      throw std::overflow_error("Pohlig-Hellman CRT invariant");
    if (capture) {
      ctx.record();
      t.merges.push_back(
          {f.power,
           xi,
           crt_x,
           crt_mod,
           static_cast<std::uint64_t>(after),
           static_cast<std::uint64_t>(combined)});
    }
    crt_x   = static_cast<std::uint64_t>(after);
    crt_mod = static_cast<std::uint64_t>(combined);
  }
  const auto candidate = crt_x % order;
  if (ctx.power(g, candidate, p) == h) {
    r.exponent = static_cast<long long>(candidate);
    r.found    = true;
  }
  return t;
}
/// @brief Shared first-collision walk; optionally retain the actual cursor updates.
RhoLogCollisionWalk first(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    std::uint64_t      a,
    std::uint64_t      b,
    std::size_t        cap,
    DiscreteLogOptions options,
    bool               capture) {
  RhoLogCollisionWalk t;
  Context             ctx{options, t.work};
  validate(g, h, p, order, ctx);
  t.tortoise = initial(g, h, p, order, a, b, ctx);
  t.hare     = t.tortoise;
  for (std::size_t i = 0; i < cap; ++i) {
    step(t.tortoise, g, h, p, order, ctx, capture ? &t.steps : nullptr, 0, i + 1, false);
    step(t.hare, g, h, p, order, ctx, capture ? &t.steps : nullptr, 0, i + 1, true);
    step(t.hare, g, h, p, order, ctx, capture ? &t.steps : nullptr, 0, i + 1, true);
    ++t.iterations;
    if (t.tortoise.value == t.hare.value) {
      t.collided = true;
      ++t.work.collisions;
      break;
    }
  }
  return t;
}
} // namespace
long long pollard_rho_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order) {
  return rho(g, h, p, order, {}, false).run.exponent;
}
DiscreteLogRun pollard_rho_log_run(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions o) {
  return rho(g, h, p, order, o, false).run;
}
RhoLogTrace pollard_rho_log_trace(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions o) {
  return rho(g, h, p, order, o, true);
}
long long
pohlig_hellman_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order) {
  return pohlig(g, h, p, order, {}, false).run.exponent;
}
DiscreteLogRun pohlig_hellman_run(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions o) {
  return pohlig(g, h, p, order, o, false).run;
}
PohligHellmanTrace pohlig_hellman_trace(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    DiscreteLogOptions o) {
  return pohlig(g, h, p, order, o, true);
}
RhoLogCollisionWalk pollard_rho_log_first_collision(
    std::uint64_t      g,
    std::uint64_t      h,
    std::uint64_t      p,
    std::uint64_t      order,
    std::uint64_t      a,
    std::uint64_t      b,
    std::size_t        cap,
    DiscreteLogOptions o) {
  return first(g, h, p, order, a, b, cap, o, true);
}
std::vector<RhoLogCollisionStudy> pollard_rho_log_collision_study() {
  std::vector<RhoLogCollisionStudy> out;
  for (const auto [p, g] : std::vector<std::pair<std::uint64_t, std::uint64_t>>{
           {1019, 2}, {2039, 7}, {5087, 5}, {10007, 5}, {20707, 5}, {50021, 3}}) {
    std::size_t completed = 0, total = 0, legacy = 0;
    for (std::uint64_t trial = 0; trial < 40; ++trial) {
      const std::uint64_t secret = (1234567ULL * (trial + 1)) % (p - 1);
      const auto          h      = mod_pow(g, secret, p);
      auto                walk   = first(g, h, p, p - 1, 1 + trial % (p - 2), 1, 100000, {}, false);
      completed += walk.collided;
      total += walk.iterations;
      legacy += walk.collided ? walk.iterations - 1 : walk.iterations;
    }
    out.push_back(
        {p,
         g,
         p - 1,
         40,
         completed,
         static_cast<double>(total) / 40,
         static_cast<double>(legacy) / 40});
  }
  return out;
}
} // namespace datamunge::algorithms
