/// @file lenstra_ecm.cpp
/// @brief Bounded affine stage1 ECM using unsigned128 modular products and actual producing evidence.
#include <datamunge/algorithms/lenstra_ecm.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private exact residue arithmetic and same-execution stage1 producer.
namespace ecm_detail {
/// @brief Unsigned64 modular residue/storage type.
using U = std::uint64_t;
/// @brief Exact double-width product type for the entire unsigned64 modulus domain.
using Wide = unsigned __int128;
/// @brief Signed widened Euclidean coefficient type; coefficients remain bounded by the64-bit modulus.
using Signed = __int128;
/// @brief Multiply residues without signed overflow.
/// @param a First residue.
/// @param b Second residue.
/// @param n Positive modulus.
/// @return Exact normalized product residue.
U mul(U a, U b, U n) {
  return static_cast<U>(Wide(a) * b % n);
}
/// @brief Add residues without64-bit wrap.
/// @param a First residue.
/// @param b Second residue.
/// @param n Positive modulus.
/// @return Exact normalized sum residue.
U add(U a, U b, U n) {
  return static_cast<U>((Wide(a) + b) % n);
}
/// @brief Subtract normalized residues without wrap.
/// @param a First normalized residue.
/// @param b Second normalized residue.
/// @param n Positive modulus.
/// @return Exact normalized difference.
U sub(U a, U b, U n) {
  return a >= b ? a - b : n - (b - a);
}
/// @brief Invert a known coprime residue with widened Euclidean coefficients.
/// @param a Actual normalized coprime residue.
/// @param n Positive modulus.
/// @return Exact normalized inverse.
U inverse(U a, U n) {
  Signed r = n, next = a, t = 0, nt = 1;
  while (next) {
    const Signed q = r / next, rr = r - q * next, tt = t - q * nt;
    r    = next;
    next = rr;
    t    = nt;
    nt   = tt;
  }
  if (t < 0)
    t += n;
  return static_cast<U>(t);
}
/// @brief Add a selected count without size_t overflow.
/// @param value Mutable count.
/// @param amount Actual increment, default1.
/// @throws std::length_error On overflow.
void count(std::size_t& value, std::size_t amount = 1) {
  if (amount > std::numeric_limits<std::size_t>::max() - value)
    throw std::length_error("ECM count overflow");
  value += amount;
}
/// @brief Admit one actual logical record before appending capture.
/// @param trace Optional mutable complete capture.
/// @param cap Optional actual-record cap,0 off.
/// @throws std::length_error On actual count or cap limits.
void record(ECMTrace* trace, std::size_t cap) {
  if (!trace)
    return;
  if (cap && trace->captured_entries >= cap)
    throw std::length_error("ECM actual capture quota");
  count(trace->captured_entries);
}
/// @brief Produce one actual affine point addition and optional evaluated-denominator record.
/// @param n Modulus>=2.
/// @param a Normalized curve coefficient.
/// @param p Actual first normalized point.
/// @param q Actual second normalized point.
/// @param work Mutable selected producing work.
/// @param curve Actual curve attempt index.
/// @param trace Optional complete capture.
/// @param cap Actual logical-record cap.
/// @return Owned actual operation including factor evidence or shortcut output.
/// @throws std::length_error On actual counts/capture quota.
/// @throws std::bad_alloc On capture growth.
ECMAddition point_add(
    U           n,
    U           a,
    ECMPoint    p,
    ECMPoint    q,
    ECMWork&    work,
    std::size_t curve,
    ECMTrace*   trace,
    std::size_t cap) {
  count(work.additions);
  ECMAddition e;
  e.curve = curve;
  e.left  = p;
  e.right = q;
  if (p.infinity) {
    e.result = q;
    return e;
  }
  if (q.infinity) {
    e.result = p;
    return e;
  }
  if (p.x == q.x && add(p.y, q.y, n) == 0)
    return e;
  e.evaluated = true;
  if (p.x == q.x && p.y == q.y) {
    e.numerator   = add(mul(3 % n, mul(p.x, p.x, n), n), a, n);
    e.denominator = add(p.y, p.y, n);
  } else {
    e.numerator   = sub(q.y, p.y, n);
    e.denominator = sub(q.x, p.x, n);
  }
  e.gcd = std::gcd(e.denominator, n);
  count(work.denominator_gcds);
  if (e.gcd > 1 && e.gcd < n)
    e.factor = e.gcd;
  else if (e.gcd == 1) {
    e.inverted = true;
    count(work.inversions);
    e.slope   = mul(e.numerator, inverse(e.denominator, n), n);
    const U x = sub(sub(mul(e.slope, e.slope, n), p.x, n), q.x, n);
    e.result  = {x, sub(mul(e.slope, sub(p.x, x, n), n), p.y, n), false};
  }
  if (trace) {
    record(trace, cap);
    trace->denominators.push_back(e);
  }
  return e;
}
/// @brief Perform the historical right-to-left scalar multiplication, including its final doubling.
/// @param p Actual starting point.
/// @param k Actual prime-power scalar.
/// @param a Normalized curve coefficient.
/// @param n Modulus>=2.
/// @param factor Mutable proper factor discovered by an actual denominator.
/// @param work Mutable producing counts.
/// @param curve Actual curve index.
/// @param trace Optional complete capture.
/// @param cap Actual record cap.
/// @return Actual resulting point, or historical partial accumulator on doubling factor exit.
/// @throws std::length_error On count/capture limits.
/// @throws std::bad_alloc On capture growth.
ECMPoint multiply(
    ECMPoint    p,
    U           k,
    U           a,
    U           n,
    U&          factor,
    ECMWork&    work,
    std::size_t curve,
    ECMTrace*   trace,
    std::size_t cap) {
  count(work.multiplications);
  ECMPoint r;
  while (k) {
    if (k & 1) {
      auto e = point_add(n, a, r, p, work, curve, trace, cap);
      r      = e.result;
      factor = e.factor;
      if (factor)
        return r;
    }
    auto e = point_add(n, a, p, p, work, curve, trace, cap);
    p      = e.result;
    factor = e.factor;
    if (factor)
      return r;
    k >>= 1;
  }
  return r;
}
/// @brief Advance the historical xorshift64 state with defined unsigned wrap.
/// @param state Mutable nonzero generator state.
/// @return Actual next random word.
U random(U& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}
/// @brief Validate explicit finite scientific search limits.
/// @param options Caller bound and curve budget.
/// @throws std::invalid_argument On limits outside documented domains.
void validate(ECMOptions options) {
  if (options.bound < 2 || options.bound > 1000000 || options.max_curves > 1000000)
    throw std::invalid_argument("ECM explicit search limits");
}
/// @brief Produce one complete bounded search, optionally observing real curves and slope records.
/// @param n Original unsigned64 input.
/// @param seed Original caller seed.
/// @param options Validated stage1 budget.
/// @param trace Optional initially empty complete capture.
/// @param cap Actual logical-record cap,0 off.
/// @return Owned actual factor/status/work.
/// @throws std::length_error On selected count/capture limits.
/// @throws std::bad_alloc On sieve/capture allocation failure.
ECMAnalysis run(U n, U seed, ECMOptions options, ECMTrace* trace, std::size_t cap) {
  ECMAnalysis out;
  out.n       = n;
  out.seed    = seed;
  out.options = options;
  out.factor  = n;
  // Every early success updates the proper-factor status in the shared finishing path.
  if (n <= 1)
    return out;
  for (U p :
       {2ULL,
        3ULL,
        5ULL,
        7ULL,
        11ULL,
        13ULL,
        17ULL,
        19ULL,
        23ULL,
        29ULL,
        31ULL,
        37ULL,
        41ULL,
        43ULL,
        47ULL}) {
    count(out.work.trial_divisions);
    if (n % p == 0) {
      out.factor = p;
      out.proper = p < n;
      return out;
    }
  }
  if (options.max_curves == 0)
    return out;
  std::vector<U>    powers;
  std::vector<bool> composite(std::size_t(options.bound) + 1, false);
  for (U p = 2; p <= options.bound; ++p)
    if (!composite[p]) {
      U power = p;
      while (power <= options.bound / p)
        power *= p;
      powers.push_back(power);
      if (trace) {
        record(trace, cap);
        trace->prime_powers.push_back(power);
      }
      for (U q = p * p; q <= options.bound; q += p)
        composite[q] = true;
    }
  U rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
  for (std::size_t t = 0; t < options.max_curves; ++t) {
    ECMCurve c;
    c.index   = t;
    c.initial = {random(rng) % n, random(rng) % n, false};
    c.final   = c.initial;
    c.a       = random(rng) % n;
    c.b =
        sub(sub(mul(c.initial.y, c.initial.y, n),
                mul(mul(c.initial.x, c.initial.x, n), c.initial.x, n),
                n),
            mul(c.a, c.initial.x, n),
            n);
    count(out.work.curves);
    const U disc =
        add(mul(4 % n, mul(mul(c.a, c.a, n), c.a, n), n), mul(27 % n, mul(c.b, c.b, n), n), n);
    c.discriminant_gcd = std::gcd(disc, n);
    count(out.work.discriminant_gcds);
    if (c.discriminant_gcd == n)
      count(out.work.singular_curves);
    else if (c.discriminant_gcd > 1)
      c.factor = c.discriminant_gcd;
    else
      for (U power : powers) {
        c.final = multiply(c.final, power, c.a, n, c.factor, out.work, t, trace, cap);
        if (c.factor)
          break;
      }
    if (trace) {
      record(trace, cap);
      trace->curves.push_back(c);
    }
    if (c.factor > 1 && c.factor < n) {
      out.factor = c.factor;
      out.proper = true;
      return out;
    }
  }
  return out;
}
/// @brief Validate a normalized point against the actual supplied curve.
/// @param n Modulus>=2.
/// @param a Normalized coefficient.
/// @param b Normalized coefficient.
/// @param p Finite normalized point or infinity.
/// @throws std::invalid_argument On residue range or off-curve values.
void on_curve(U n, U a, U b, ECMPoint p) {
  if (p.infinity)
    return;
  if (p.x >= n || p.y >= n ||
      mul(p.y, p.y, n) != add(add(mul(mul(p.x, p.x, n), p.x, n), mul(a, p.x, n), n), b, n))
    throw std::invalid_argument("ECM normalized on-curve point required");
}
} // namespace ecm_detail
ECMAddition
ecm_add(std::uint64_t n, std::uint64_t a, std::uint64_t b, ECMPoint left, ECMPoint right) {
  if (n < 2 || a >= n || b >= n)
    throw std::invalid_argument("ECM modulus/coefficient range");
  ecm_detail::on_curve(n, a, b, left);
  ecm_detail::on_curve(n, a, b, right);
  ECMWork work;
  return ecm_detail::point_add(n, a, left, right, work, 0, nullptr, 0);
}
std::uint64_t lenstra_ecm(std::uint64_t n, std::uint64_t seed) {
  return lenstra_ecm_analyze(n, seed).factor;
}
ECMAnalysis lenstra_ecm_analyze(std::uint64_t n, std::uint64_t seed, ECMOptions options) {
  ecm_detail::validate(options);
  return ecm_detail::run(n, seed, options, nullptr, 0);
}
ECMTrace lenstra_ecm_trace(
    std::uint64_t n,
    std::uint64_t seed,
    ECMOptions    options,
    std::size_t   max_entries) {
  ecm_detail::validate(options);
  ECMTrace out;
  out.analysis = ecm_detail::run(n, seed, options, &out, max_entries);
  return out;
}
ECMHasseStudy ecm_hasse_smoothness(std::uint32_t prime, std::uint32_t bound) {
  if (prime < 2 || prime > 1000000 || bound < 2 || bound > 1000000)
    throw std::invalid_argument("ECM small-prime interval domain");
  for (std::uint32_t d = 2; d <= prime / d; ++d)
    if (prime % d == 0)
      throw std::invalid_argument("ECM Hasse prime required");
  std::uint32_t radius = 0;
  while ((radius + 1) * (radius + 1) <= 4 * prime)
    ++radius;
  ECMHasseStudy out;
  out.prime = prime;
  out.bound = bound;
  out.lower = prime + 1 - radius;
  out.upper = prime + 1 + radius;
  for (std::uint32_t k = out.lower; k <= out.upper; ++k) {
    auto remaining = k;
    bool smooth    = true;
    for (std::uint32_t d = 2; d <= remaining / d; ++d)
      while (remaining % d == 0) {
        if (d > bound)
          smooth = false;
        remaining /= d;
      }
    if (remaining > bound)
      smooth = false;
    if (smooth)
      ++out.smooth;
    ++out.total;
  }
  return out;
}
} // namespace datamunge::algorithms
