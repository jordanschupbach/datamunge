/// @file shamir_secret_sharing.cpp
/// @brief One exact checked prime-field evaluation/interpolation producer per operation.
#include <bit>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <datamunge/algorithms/shamir_secret_sharing.hpp>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Check an incremental logical size without wrap.
/// @param used Existing count.
/// @param n Complete next units.
/// @param cap Zero or enabled bound.
void admit(std::size_t used, std::size_t n, std::size_t cap) {
  if (n > std::numeric_limits<std::size_t>::max() - used || (cap && (used > cap || n > cap - used)))
    throw std::length_error("Shamir: capacity/policy");
}
/// @brief Exact overflow-safe modular addition of reduced field values.
/// @param a First residue.
/// @param b Second residue.
/// @param p Positive prime.
/// @return Sum residue.
std::uint64_t add(std::uint64_t a, std::uint64_t b, std::uint64_t p) {
  return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) + b) % p);
}
/// @brief Exact overflow-free modular subtraction of reduced residues.
/// @param a Minuend.
/// @param b Subtrahend.
/// @param p Positive prime.
/// @return Difference residue.
std::uint64_t sub(std::uint64_t a, std::uint64_t b, std::uint64_t p) {
  return a >= b ? a - b : p - (b - a);
}
/// @brief Validate complete prime-field observations before any interpolation.
/// @param shares Borrowed complete points.
/// @param p Candidate modulus.
void validate(const std::vector<SecretShare>& shares, std::uint64_t p) {
  if (!is_probable_prime(p) || shares.empty())
    throw std::invalid_argument("Shamir: prime and nonempty observations required");
  std::set<std::uint64_t> seen;
  for (auto s : shares)
    if (!s.x || s.x >= p || s.y >= p || !seen.insert(s.x).second)
      throw std::invalid_argument("Shamir: invalid/duplicate field observation");
}
/// @brief Count and execute one actual canonical exact modular product.
/// @param a First field residue.
/// @param b Second field residue.
/// @param p Prime.
/// @param used Mutable actual product counter.
/// @param o Policies.
/// @return Actual product residue.
std::uint64_t product(
    std::uint64_t        a,
    std::uint64_t        b,
    std::uint64_t        p,
    std::size_t&         used,
    const ShamirOptions& o) {
  admit(used, 1, o.max_products);
  ++used;
  return mul_mod(a, b, p);
}
/// @brief Admit and delegate one actual inverse powering; add actual reported work once.
/// @param value Nonzero field denominator.
/// @param p Prime.
/// @param used Mutable actual total products.
/// @param inverse_used Mutable actual inverse-only products.
/// @param o Policies.
/// @return Actual Fermat inverse.
std::uint64_t inverse(
    std::uint64_t        value,
    std::uint64_t        p,
    std::size_t&         used,
    std::size_t&         inverse_used,
    const ShamirOptions& o) {
  const auto exponent = p - 2;
  const auto bound = static_cast<std::size_t>(std::bit_width(exponent) + std::popcount(exponent));
  admit(used, bound, o.max_products);
  auto r = mod_pow_trace(value, exponent, p);
  used += r.work.products;
  inverse_used += r.work.products;
  return r.result;
}
/// @brief Evaluate all actual ascending-power polynomial terms once.
/// @param secret Field constant.
/// @param n Complete shares.
/// @param c Borrowed field coefficients.
/// @param p Prime.
/// @param o Policies.
/// @param capture Retain actual coefficients/terms.
/// @return Owned actual split output.
ShamirSplitRun split(
    std::uint64_t                     secret,
    unsigned                          n,
    const std::vector<std::uint64_t>& c,
    std::uint64_t                     p,
    const ShamirOptions&              o,
    bool                              capture) {
  admit(0, n, o.max_shares);
  admit(0, c.size(), o.max_coefficients);
  if (c.size() == std::numeric_limits<std::size_t>::max() || c.size() + 1 > n || n >= p ||
      secret >= p || !is_probable_prime(p))
    throw std::invalid_argument("Shamir: prime/threshold/secret");
  for (auto a : c)
    if (a >= p)
      throw std::invalid_argument("Shamir: coefficient outside field");
  if (n && c.size() > std::numeric_limits<std::size_t>::max() / n)
    throw std::length_error("Shamir: term geometry");
  const auto terms = static_cast<std::size_t>(n) * c.size();
  if (terms > std::numeric_limits<std::size_t>::max() / 2)
    throw std::length_error("Shamir: work geometry");
  admit(0, 2 * terms, o.max_products);
  if (capture)
    admit(c.size(), terms, o.max_trace_entries);
  ShamirSplitRun out;
  out.shares.reserve(n);
  if (capture) {
    out.coefficients = c;
    out.steps.reserve(terms);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t x     = i + 1;
    auto                y     = secret;
    std::uint64_t       power = 1;
    for (std::size_t j = 0; j < c.size(); ++j) {
      power           = product(power, x, p, out.modular_products, o);
      const auto term = product(c[j], power, p, out.modular_products, o), before = y;
      y = add(y, term, p);
      if (capture)
        out.steps.push_back({i, j + 1, power, c[j], before, term, y});
    }
    out.shares.push_back({x, y});
  }
  return out;
}
/// @brief Accumulate actual Lagrange contributions once.
/// @param s Borrowed complete field points.
/// @param p Prime.
/// @param o Policies.
/// @param capture Retain actual contribution records.
/// @return Owned actual interpolation/work.
ShamirCombineRun
combine(const std::vector<SecretShare>& s, std::uint64_t p, const ShamirOptions& o, bool capture) {
  admit(0, s.size(), o.max_shares);
  validate(s, p);
  if (capture)
    admit(0, s.size(), o.max_trace_entries);
  ShamirCombineRun out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    std::uint64_t num = 1, den = 1;
    for (std::size_t j = 0; j < s.size(); ++j)
      if (i != j) {
        num = product(num, s[j].x, p, out.modular_products, o);
        den = product(den, sub(s[j].x, s[i].x, p), p, out.modular_products, o);
      }
    const auto inv   = inverse(den, p, out.modular_products, out.inverse_products, o);
    const auto basis = product(num, inv, p, out.modular_products, o),
               term  = product(s[i].y, basis, p, out.modular_products, o);
    out.secret       = add(out.secret, term, p);
    if (capture)
      out.terms.push_back({i, num, den, inv, basis, term, out.secret});
  }
  return out;
}
} // namespace
ShamirSplitRun shamir_split_run(
    std::uint64_t                     s,
    unsigned                          n,
    const std::vector<std::uint64_t>& c,
    std::uint64_t                     p,
    const ShamirOptions&              o) {
  return split(s, n, c, p, o, false);
}
ShamirSplitRun shamir_split_trace(
    std::uint64_t                     s,
    unsigned                          n,
    const std::vector<std::uint64_t>& c,
    std::uint64_t                     p,
    const ShamirOptions&              o) {
  return split(s, n, c, p, o, true);
}
ShamirCombineRun
shamir_combine_run(const std::vector<SecretShare>& s, std::uint64_t p, const ShamirOptions& o) {
  return combine(s, p, o, false);
}
ShamirCombineRun
shamir_combine_trace(const std::vector<SecretShare>& s, std::uint64_t p, const ShamirOptions& o) {
  return combine(s, p, o, true);
}
std::vector<SecretShare>
shamir_split(std::uint64_t s, unsigned n, const std::vector<std::uint64_t>& c, std::uint64_t p) {
  return shamir_split_run(s, n, c, p).shares;
}
std::uint64_t shamir_combine(const std::vector<SecretShare>& s, std::uint64_t p) {
  return shamir_combine_run(s, p).secret;
}
std::vector<std::uint64_t> shamir_quadratic_completion(
    const std::vector<SecretShare>& s,
    std::uint64_t                   candidate,
    std::uint64_t                   p) {
  if (s.size() != 2 || candidate >= p)
    throw std::invalid_argument("Shamir completion: two observations and field candidate required");
  validate(s, p);
  const auto r1 = mul_mod(sub(s[0].y, candidate, p), mod_pow(s[0].x, p - 2, p), p),
             r2 = mul_mod(sub(s[1].y, candidate, p), mod_pow(s[1].x, p - 2, p), p);
  const auto a2 = mul_mod(sub(r2, r1, p), mod_pow(sub(s[1].x, s[0].x, p), p - 2, p), p),
             a1 = sub(r1, mul_mod(a2, s[0].x, p), p);
  return {a1, a2};
}
} // namespace datamunge::algorithms
