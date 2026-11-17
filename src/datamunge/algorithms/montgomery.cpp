/// @file montgomery.cpp
/// @brief Carry-safe canonical REDC and context-based powering.
#include <bit>
#include <datamunge/algorithms/montgomery.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
MontgomeryContext::MontgomeryContext(std::uint64_t n) : n_(n) {
  if (!n || !(n & 1))
    throw std::invalid_argument("MontgomeryContext: modulus must be odd and nonzero");
  auto inv    = n;
  inverse_[0] = inv;
  for (unsigned i = 0; i < 5; ++i) {
    inv *= 2 - n * inv;
    inverse_[i + 1] = inv;
  }
  nprime_ = 0 - inv;
  r_      = static_cast<std::uint64_t>((static_cast<__uint128_t>(1) << 64) % n);
  r2_     = static_cast<std::uint64_t>(static_cast<__uint128_t>(r_) * r_ % n);
}
std::uint64_t MontgomeryContext::redc(std::uint64_t a, std::uint64_t b, MontgomeryReduction* out)
    const {
  const auto t          = static_cast<__uint128_t>(a) * b;
  const auto q          = static_cast<std::uint64_t>(t) * nprime_;
  const auto correction = static_cast<__uint128_t>(q) * n_;
  const auto sum        = t + correction;
  const bool carry      = sum < t;
  const auto high       = static_cast<std::uint64_t>(sum >> 64);
  const bool sub        = carry || high >= n_;
  const auto result     = sub ? high - n_ : high;
  if (out)
    *out = {
        a,
        b,
        static_cast<std::uint64_t>(t),
        static_cast<std::uint64_t>(t >> 64),
        q,
        carry,
        high,
        sub,
        result};
  return result;
}
std::uint64_t MontgomeryContext::reduce(std::uint64_t a, std::uint64_t b) const {
  if (a >= n_ || b >= n_)
    throw std::invalid_argument("MontgomeryContext::reduce: operands must be reduced");
  return redc(a, b, nullptr);
}
MontgomeryReduction MontgomeryContext::reduce_trace(std::uint64_t a, std::uint64_t b) const {
  if (a >= n_ || b >= n_)
    throw std::invalid_argument("MontgomeryContext::reduce_trace: operands must be reduced");
  MontgomeryReduction t;
  redc(a, b, &t);
  return t;
}
std::uint64_t MontgomeryContext::multiply(std::uint64_t a, std::uint64_t b) const {
  return redc(redc(a % n_, r2_, nullptr), b % n_, nullptr);
}
std::uint64_t MontgomeryContext::run_power(
    std::uint64_t         base,
    std::uint64_t         exponent,
    MontgomeryPowerTrace* out) const {
  if (n_ == 1)
    return 0;
  auto running = redc(base % n_, r2_, nullptr), acc = r_;
  if (out) {
    out->initial_running     = running;
    out->initial_accumulator = acc;
    out->reductions          = 1;
  }
  while (exponent) {
    const auto before = acc;
    const bool fold   = exponent & 1;
    if (fold)
      acc = redc(acc, running, nullptr);
    const auto squared = redc(running, running, nullptr);
    if (out) {
      out->steps.push_back({exponent, running, before, fold, acc, squared});
      out->reductions += 1 + fold;
    }
    running = squared;
    exponent >>= 1;
  }
  if (out)
    ++out->reductions;
  return redc(acc, 1, nullptr);
}
std::uint64_t MontgomeryContext::power(std::uint64_t base, std::uint64_t exponent) const {
  return run_power(base, exponent, nullptr);
}
MontgomeryPowerTrace
MontgomeryContext::power_trace(std::uint64_t base, std::uint64_t exponent, std::size_t cap) const {
  const auto bits = n_ == 1 ? 0 : std::bit_width(exponent);
  if (cap && bits > cap)
    throw std::length_error("MontgomeryContext::power_trace: bit cap exceeded");
  MontgomeryPowerTrace t;
  t.base     = base;
  t.exponent = exponent;
  t.modulus  = n_;
  t.steps.reserve(bits);
  t.result = run_power(base, exponent, &t);
  return t;
}
std::uint64_t montgomery_multiply(std::uint64_t a, std::uint64_t b, std::uint64_t n) {
  return MontgomeryContext(n).multiply(a, b);
}
} // namespace datamunge::algorithms
