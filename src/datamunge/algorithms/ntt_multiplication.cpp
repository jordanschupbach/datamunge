/// @file ntt_multiplication.cpp
/// @brief Exact bounded single-prime decimal convolution with actual butterfly and carry records.
#include <algorithm>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <datamunge/algorithms/ntt_multiplication.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical fixed-prime decimal implementation.
namespace datamunge::algorithms {
/// @brief Private validated decimal/transform producer.
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Fixed NTT prime with a primitive2^23 root subgroup.
constexpr std::uint64_t modulus = 998244353;
/// @brief Primitive root used by every actual stage.
constexpr std::uint64_t primitive_root = 3;
/// @brief Validate the whole original operand and normalize its leading zeros.
/// @param value Complete borrowed decimal input.
/// @param cap Complete original input policy.
/// @return Owned canonical unsigned decimal input.
std::string decimal(const std::string& value, std::size_t cap) {
  if (value.empty())
    throw std::invalid_argument("NTT: empty decimal");
  if (cap && value.size() > cap)
    throw std::length_error("NTT: original digit policy");
  for (unsigned char c : value)
    if (c < '0' || c > '9')
      throw std::invalid_argument("NTT: nondecimal character");
  const auto first = value.find_first_not_of('0');
  return first == std::string::npos ? "0" : value.substr(first);
}
/// @brief Actual per-product transform work and optional state capture.
struct Work {
  NTTMultiplicationRun& out;     ///< Owned caller result currently being produced.
  bool                  capture; ///< Whether actual states are retained.

  /// @brief Multiply two valid residues and count the real word operation.
  /// @param a First residue below the fixed prime.
  /// @param b Second residue below the fixed prime.
  /// @return Exact modular product; a*b fits uint64 under this prime.
  std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    ++out.modular_products;
    return a * b % modulus;
  }
  /// @brief Compute one actual root/inverse with canonical counted binary powering.
  /// @param a Base residue.
  /// @param exponent Actual exponent.
  /// @return Actual canonical modular power.
  std::uint64_t power(std::uint64_t a, std::uint64_t exponent) {
    const auto result = mod_pow_trace(a, exponent, modulus);
    out.root_products += result.work.products;
    out.modular_products += result.work.products;
    return result.result;
  }
  /// @brief Retain an actual already completed phase without recomputing it.
  /// @param values Actual current residues.
  /// @param which Actual input/inverse stream.
  /// @param phase Actual completed phase.
  /// @param length Current butterfly length, or zero.
  /// @param root Actual twiddle or inverse-size residue.
  /// @param butterflies Actual phase butterfly count.
  void record(
      const std::vector<std::uint64_t>& values,
      NTTTransform                      which,
      NTTPhase                          phase,
      std::size_t                       length,
      std::uint64_t                     root,
      std::size_t                       butterflies) {
    if (capture)
      out.stages.push_back({which, phase, length, root, butterflies, values});
  }
  /// @brief Execute one actual bit reversal and radix-two transform in place.
  /// @param values Actual owned input/product residues.
  /// @param which Actual forward/inverse stream.
  void transform(std::vector<std::uint64_t>& values, NTTTransform which) {
    const auto n       = values.size();
    const bool inverse = which == NTTTransform::Inverse;
    for (std::size_t i = 1, j = 0; i < n; ++i) {
      ++out.bit_reversal_visits;
      std::size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j) {
        std::swap(values[i], values[j]);
        ++out.bit_reversal_swaps;
      }
    }
    record(values, which, NTTPhase::BitReverse, 0, 0, 0);
    const auto root = inverse ? power(primitive_root, modulus - 2) : primitive_root;
    for (std::size_t length = 2; length <= n; length <<= 1) {
      const auto  w                 = power(root, (modulus - 1) / length);
      std::size_t stage_butterflies = 0;
      for (std::size_t i = 0; i < n; i += length) {
        std::uint64_t twiddle = 1;
        for (std::size_t k = 0; k < length / 2; ++k) {
          const auto u = values[i + k], v = multiply(values[i + k + length / 2], twiddle);
          values[i + k]              = (u + v) % modulus;
          values[i + k + length / 2] = (u + modulus - v) % modulus;
          twiddle                    = multiply(twiddle, w);
          ++out.butterflies;
          ++stage_butterflies;
        }
      }
      record(values, which, NTTPhase::Butterfly, length, w, stage_butterflies);
    }
    if (inverse) {
      const auto scale = power(n, modulus - 2);
      for (auto& x : values) {
        x = multiply(x, scale);
        ++out.inverse_scale_products;
      }
      record(values, which, NTTPhase::Scale, 0, scale, 0);
    }
  }
};
/// @brief Execute the checked actual three-transform decimal product.
/// @param a Complete first operand.
/// @param b Complete second operand.
/// @param o Complete policies.
/// @param capture Retain actual phases and carries.
/// @return Owned exact product/work/history.
NTTMultiplicationRun
solve(const std::string& a, const std::string& b, const NTTMultiplicationOptions& o, bool capture) {
  NTTMultiplicationRun out;
  out.first  = decimal(a, o.max_digits);
  out.second = decimal(b, o.max_digits);
  if (out.first == "0" || out.second == "0") {
    out.product = "0";
    return out;
  }
  const auto max = std::numeric_limits<std::size_t>::max();
  if (out.first.size() > max - out.second.size())
    throw std::length_error("NTT: digit sum capacity");
  const auto  required = out.first.size() + out.second.size();
  std::size_t n = 1, levels = 0;
  while (n < required) {
    if (n >= 8388608)
      throw std::length_error("NTT: hard root capacity");
    n *= 2;
    ++levels;
  }
  if ((o.max_transform_size && n > o.max_transform_size) ||
      std::min(out.first.size(), out.second.size()) > (modulus - 1) / 81)
    throw std::length_error("NTT: transform/coefficient bound");
  if (capture) {
    const auto stages = 3 * levels + 4;
    if (n > max / 5 || n == max || stages > max / (n + 1) || 5 * n > max - stages * (n + 1) ||
        (o.max_trace_entries && 5 * n + stages * (n + 1) > o.max_trace_entries))
      throw std::length_error("NTT: complete capture geometry");
  }
  out.transform_size = n;
  std::vector<std::uint64_t> first(n), second(n);
  for (std::size_t i = 0; i < out.first.size(); ++i)
    first[i] = out.first[out.first.size() - 1 - i] - '0';
  for (std::size_t i = 0; i < out.second.size(); ++i)
    second[i] = out.second[out.second.size() - 1 - i] - '0';
  if (capture) {
    out.first_padded  = first;
    out.second_padded = second;
  }
  Work work{out, capture};
  work.transform(first, NTTTransform::First);
  work.transform(second, NTTTransform::Second);
  for (std::size_t i = 0; i < n; ++i) {
    first[i] = work.multiply(first[i], second[i]);
    ++out.pointwise_products;
  }
  if (capture)
    out.pointwise = first;
  work.transform(first, NTTTransform::Inverse);
  if (capture)
    out.coefficients = first;
  std::string reversed;
  reversed.reserve(n);
  std::uint64_t carry = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto before = carry, total = first[i] + carry;
    const auto digit = static_cast<unsigned>(total % 10);
    carry            = total / 10;
    reversed.push_back(static_cast<char>('0' + digit));
    ++out.carry_updates;
    if (capture)
      out.carries.push_back({i, first[i], before, digit, carry});
  }
  if (carry)
    throw std::logic_error("NTT: recovered carry invariant");
  while (reversed.size() > 1 && reversed.back() == '0')
    reversed.pop_back();
  std::reverse(reversed.begin(), reversed.end());
  out.product = std::move(reversed);
  return out;
}
} // namespace
NTTMultiplicationRun
ntt_multiply_run(const std::string& a, const std::string& b, const NTTMultiplicationOptions& o) {
  return solve(a, b, o, false);
}
NTTMultiplicationRun
ntt_multiply_trace(const std::string& a, const std::string& b, const NTTMultiplicationOptions& o) {
  return solve(a, b, o, true);
}
std::string ntt_mul(const std::string& a, const std::string& b) {
  return solve(a, b, {}, false).product;
}
} // namespace datamunge::algorithms
