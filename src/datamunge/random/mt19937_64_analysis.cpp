/// @file mt19937_64_analysis.cpp
/// @brief MT19937-64 twist/tempering inspection, inverse output map and conformance tables.
#include <datamunge/random/mt19937_64_analysis.hpp>
#include <datamunge/random/random.hpp>
#include <format>
#include <random>
#include <stdexcept>
namespace datamunge::random {
std::array<std::uint64_t, 4>
mt19937_64_twist_steps(std::uint64_t left, std::uint64_t right, std::uint64_t lag) noexcept {
  const auto          joined     = (left & 0xffffffff80000000ULL) | (right & 0x7fffffffULL);
  const auto          shifted    = joined >> 1U;
  const std::uint64_t correction = (joined & 1U) ? 0xb5026f5aa96619e9ULL : 0;
  return {joined, shifted, correction, lag ^ shifted ^ correction};
}
std::array<std::uint64_t, 5> mt19937_64_temper_steps(std::uint64_t word) noexcept {
  std::array<std::uint64_t, 5> out{word};
  out[1] = out[0] ^ ((out[0] >> 29U) & 0x5555555555555555ULL);
  out[2] = out[1] ^ ((out[1] << 17U) & 0x71d67fffeda60000ULL);
  out[3] = out[2] ^ ((out[2] << 37U) & 0xfff7eee000000000ULL);
  out[4] = out[3] ^ (out[3] >> 43U);
  return out;
}
/// @brief Internal finite inverse series for fixed masked shifts.
namespace {
/// @brief Invert x XOR ((x >> shift) AND mask) by fixed-point substitution.
/// @param output The shifted-XOR result.
/// @param shift Positive bit shift below 64.
/// @param mask Mask on the shifted operand.
/// @return Unique 64-bit preimage after enough substitutions to propagate all bits.
std::uint64_t undo_right(std::uint64_t output, unsigned shift, std::uint64_t mask) noexcept {
  auto word = output;
  for (unsigned offset = shift; offset < 64; offset += shift)
    word = output ^ ((word >> shift) & mask);
  return word;
}
/// @brief Invert x XOR ((x << shift) AND mask) by fixed-point substitution.
/// @param output The shifted-XOR result.
/// @param shift Positive bit shift below 64.
/// @param mask Mask on the shifted operand.
/// @return Unique 64-bit preimage under unsigned truncated shifts.
std::uint64_t undo_left(std::uint64_t output, unsigned shift, std::uint64_t mask) noexcept {
  auto word = output;
  for (unsigned offset = shift; offset < 64; offset += shift)
    word = output ^ ((word << shift) & mask);
  return word;
}
} // namespace
std::uint64_t mt19937_64_untemper(std::uint64_t output) noexcept {
  auto word = undo_right(output, 43, ~std::uint64_t{0});
  word      = undo_left(word, 37, 0xfff7eee000000000ULL);
  word      = undo_left(word, 17, 0x71d67fffeda60000ULL);
  return undo_right(word, 29, 0x5555555555555555ULL);
}
MersenneTwister64 MersenneTwister64::from_output_block(
    const std::array<std::uint64_t, 312>& outputs) {
  MersenneTwister64 clone(0);
  for (std::size_t i = 0; i < outputs.size(); ++i)
    clone.state_[i] = mt19937_64_untemper(outputs[i]);
  clone.index_ = 312;
  return clone;
}
dstruct::DataFrame mt19937_64_conformance(
    const std::vector<std::uint64_t>& seeds,
    std::size_t                       draws) {
  if (!draws)
    throw std::invalid_argument("MT19937-64 conformance: draws must be positive");
  std::vector<std::string> seed, nfirst, sfirst, nlast, slast;
  std::vector<double>      counts, mismatches;
  for (auto value : seeds) {
    MersenneTwister64 native(value);
    std::mt19937_64   standard(value);
    std::uint64_t     a = 0, b = 0, first_a = 0, first_b = 0;
    std::size_t       errors = 0;
    for (std::size_t i = 0; i < draws; ++i) {
      a = native.next_u64();
      b = standard();
      if (!i) {
        first_a = a;
        first_b = b;
      }
      errors += a != b;
    }
    seed.push_back(std::format("{:016x}", value));
    counts.push_back(draws);
    mismatches.push_back(errors);
    nfirst.push_back(std::format("{:016x}", first_a));
    sfirst.push_back(std::format("{:016x}", first_b));
    nlast.push_back(std::format("{:016x}", a));
    slast.push_back(std::format("{:016x}", b));
  }
  dstruct::DataFrame out;
  out.add_column("seed", seed);
  out.add_column("draws", counts);
  out.add_column("mismatches", mismatches);
  out.add_column("native_first", nfirst);
  out.add_column("standard_first", sfirst);
  out.add_column("native_last", nlast);
  out.add_column("standard_last", slast);
  return out;
}
} // namespace datamunge::random
