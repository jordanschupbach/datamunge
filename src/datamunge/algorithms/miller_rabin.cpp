/// @file miller_rabin.cpp
/// @brief Shared exact primality producer with optional actual round capture.
#include <bit>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private single-round producer and fixed deterministic witness set.
namespace {
/// @brief Fixed first twelve prime bases, also used for trial division.
constexpr std::uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
/// @brief Execute one round, optionally capturing its actual states.
/// @param n Candidate with full unsigned range.
/// @param a Arbitrary base.
/// @param trace Optional owned sink; null avoids history allocation.
/// @param cap Optional upfront record bound; zero disables.
/// @return Actual strong-round outcome.
/// @throws std::length_error If the capture bound exceeds cap.
bool round(std::uint64_t n, std::uint64_t a, MillerRabinRoundTrace* trace, std::size_t cap) {
  if (n < 2)
    return false;
  if (n % 2 == 0)
    return n == 2;
  a %= n;
  if (trace)
    trace->reduced_base = a;
  if (a == 0)
    return true;
  const unsigned s = std::countr_zero(n - 1);
  const auto     d = (n - 1) >> s;
  if (trace) {
    const auto bound = std::size_t(std::bit_width(d)) + s;
    if (cap && bound > cap)
      throw std::length_error("miller_rabin_round_trace: record bound exceeded");
    trace->d = d;
    trace->s = s;
    trace->residues.reserve(s);
    trace->powering = mod_pow_trace(a, d, n);
  }
  auto x = trace ? trace->powering.result : mod_pow(a, d, n);
  if (trace)
    trace->residues.push_back(x);
  if (x == 1 || x == n - 1)
    return true;
  for (unsigned r = 1; r < s; ++r) {
    x = mul_mod(x, x, n);
    if (trace)
      trace->residues.push_back(x);
    if (x == n - 1)
      return true;
  }
  return false;
}
} // namespace
bool is_strong_probable_prime_base(std::uint64_t n, std::uint64_t a) {
  return round(n, a, nullptr, 0);
}
MillerRabinRoundTrace miller_rabin_round_trace(std::uint64_t n, std::uint64_t a, std::size_t cap) {
  MillerRabinRoundTrace t;
  t.n      = n;
  t.base   = a;
  t.passed = round(n, a, &t, cap);
  return t;
}
bool is_probable_prime(std::uint64_t n) {
  if (n < 2)
    return false;
  for (auto p : witnesses) {
    if (n == p)
      return true;
    if (n % p == 0)
      return false;
  }
  for (auto a : witnesses)
    if (!round(n, a, nullptr, 0))
      return false;
  return true;
}
} // namespace datamunge::algorithms
