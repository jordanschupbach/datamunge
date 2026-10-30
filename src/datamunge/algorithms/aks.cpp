/// @file aks.cpp
/// @brief AKS primality: perfect powers, the modulus r, ring powers and the step trace.

#include <datamunge/algorithms/aks.hpp>

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Ring arithmetic for AKS.
namespace {

/// @brief 128-bit unsigned products.
using u128 = unsigned __int128;

/// @brief \f$a b \bmod m\f$ without overflow.
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
  return static_cast<std::uint64_t>(static_cast<u128>(a) * b % m);
}

/// @brief Product of two polynomials modulo \f$(x^r-1, n)\f$ (cyclic convolution). For
///        \f$n<2^{32}\f$ the products are summed in 128 bits and reduced once per coefficient.
std::vector<std::uint64_t> cyclic_mul(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b,
                                      std::uint64_t n) {
  const std::size_t r = a.size();
  std::vector<u128> acc(r, 0);
  const bool small = n < (std::uint64_t{1} << 32);
  for (std::size_t i = 0; i < r; ++i) {
    if (a[i] == 0) continue;
    const u128 ai = a[i];
    std::size_t k = i;
    for (std::size_t j = 0; j < r; ++j, ++k) {
      if (k == r) k = 0;
      if (small) acc[k] += ai * b[j];
      else acc[k] = (acc[k] + ai * b[j] % n) % n;
    }
  }
  std::vector<std::uint64_t> out(r);
  for (std::size_t k = 0; k < r; ++k) out[k] = static_cast<std::uint64_t>(acc[k] % n);
  return out;
}

} // namespace

std::uint64_t multiplicative_order(std::uint64_t n, std::uint64_t r) {
  if (r == 0) throw std::invalid_argument("multiplicative_order: r must be positive");
  if (r == 1) return 1;
  if (std::gcd(n % r, r) != 1) return 0;
  std::uint64_t k = 1, cur = n % r;
  while (cur != 1) {
    cur = mulmod(cur, n % r, r);
    ++k;
  }
  return k;
}

std::uint64_t euler_totient(std::uint64_t n) {
  std::uint64_t phi = n, m = n;
  for (std::uint64_t p = 2; p * p <= m; ++p)
    if (m % p == 0) {
      while (m % p == 0) m /= p;
      phi -= phi / p;
    }
  if (m > 1) phi -= phi / m;
  return phi;
}

std::pair<std::uint64_t, int> perfect_power(std::uint64_t n) {
  for (int e = 63; e >= 2; --e) {
    if (n < 4) break;
    // Integer e-th root by rounding the floating estimate and correcting.
    auto power = [&](std::uint64_t b) {
      u128 p = 1;
      for (int i = 0; i < e; ++i) {
        p *= b;
        if (p > n) return u128{n} + 1;
      }
      return p;
    };
    std::uint64_t b = static_cast<std::uint64_t>(std::llround(std::pow(static_cast<double>(n), 1.0 / e)));
    for (std::uint64_t c : {b > 0 ? b - 1 : 0, b, b + 1})
      if (c >= 2 && power(c) == n) return {c, e};
  }
  return {n, 1};
}

std::uint64_t aks_r(std::uint64_t n) {
  if (n < 2) throw std::invalid_argument("aks_r: n must be at least 2");
  const double log2n = std::log2(static_cast<double>(n));
  const double bound = log2n * log2n;
  for (std::uint64_t r = 2;; ++r) {
    if (r >= n) return r;
    if (std::gcd(n % r, r) != 1) continue;
    if (static_cast<double>(multiplicative_order(n, r)) > bound) return r;
  }
}

std::vector<std::uint64_t> aks_power(std::uint64_t a, std::uint64_t n, std::uint64_t r) {
  if (n < 2) throw std::invalid_argument("aks_power: n must be at least 2");
  if (r == 0) throw std::invalid_argument("aks_power: r must be positive");
  std::vector<std::uint64_t> result(r, 0), base(r, 0);
  result[0] = 1;
  base[0] = a % n;
  base[1 % r] = (base[1 % r] + 1) % n;
  for (std::uint64_t e = n; e > 0; e >>= 1) {
    if (e & 1U) result = cyclic_mul(result, base, n);
    if (e > 1) base = cyclic_mul(base, base, n);
  }
  return result;
}

bool aks_congruence_holds(std::uint64_t n, std::uint64_t r, std::uint64_t a) {
  const auto lhs = aks_power(a, n, r);
  std::vector<std::uint64_t> rhs(r, 0);
  rhs[n % r] = 1;
  rhs[0] = (rhs[0] + a) % n;
  return lhs == rhs;
}

AksTrace aks_trace(std::uint64_t n, bool check_congruences) {
  AksTrace t;
  t.n = n;
  if (n < 2) {
    t.decided_by = "n < 2";
    return t;
  }
  if (const auto [b, e] = perfect_power(n); e >= 2) {
    t.decided_by = "perfect power";
    t.power_base = b;
    t.power_exponent = e;
    return t;
  }
  t.r = aks_r(n);
  t.order = std::gcd(n % t.r, t.r) == 1 ? multiplicative_order(n, t.r) : 0;
  for (std::uint64_t a = 2; a <= t.r && a < n; ++a)
    if (const std::uint64_t g = std::gcd(a, n); g > 1 && g < n) {
      t.decided_by = "small factor";
      t.factor = g;
      return t;
    }
  if (n <= t.r) {
    t.decided_by = "n <= r";
    t.is_prime = true;
    return t;
  }
  const std::uint64_t phi = euler_totient(t.r);
  t.a_max = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(phi)) * std::log2(static_cast<double>(n)));
  if (!check_congruences) {
    t.decided_by = "congruences not checked";
    return t;
  }
  for (std::uint64_t a = 1; a <= t.a_max; ++a) {
    ++t.congruences_checked;
    if (!aks_congruence_holds(n, t.r, a)) {
      t.decided_by = "congruence fails";
      t.witness_a = a;
      return t;
    }
  }
  t.decided_by = "all congruences hold";
  t.is_prime = true;
  return t;
}

} // namespace datamunge::algorithms
