/// @file rice_coding.cpp
/// @brief Native Rice adapters to one Golomb execution and actual conditional length studies.
#include <cmath>
#include <datamunge/algorithms/rice_coding.hpp>
#include <stdexcept>
/// @brief Canonical power-of-two integer coding implementation.
namespace datamunge::algorithms {
GolombTrace rice_encode_trace(const std::vector<std::uint64_t>& values, int k, std::size_t cap) {
  const auto parameter = golomb_detail::rice_parameter(k);
  const auto size      = golomb_detail::total(values, parameter);
  if (cap && size > cap)
    throw std::length_error("Rice: emitted bit policy");
  return golomb_encode_trace(values, parameter, 0);
}
GolombTrace rice_decode_trace(const std::vector<bool>& bits, int k, int count, std::size_t cap) {
  const auto parameter = golomb_detail::rice_parameter(k);
  if (cap && bits.size() > cap)
    throw std::length_error("Rice: complete input bit policy");
  auto trace = golomb_decode_trace(bits, parameter, count, 0);
  if (trace.words.size() != static_cast<std::size_t>(count))
    throw std::invalid_argument("Rice: missing requested complete words");
  return trace;
}
std::vector<RiceLengthRow>
rice_length_study(int max_k, std::size_t largest, double half_life, std::size_t max_bits) {
  golomb_detail::rice_parameter(max_k);
  if (!std::isfinite(half_life) || half_life <= 0)
    throw std::invalid_argument("Rice: half-life");
  if (largest > 1000000)
    throw std::length_error("Rice: support policy");
  std::vector<std::uint64_t> values(largest + 1);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = i;
  std::vector<RiceLengthRow> out;
  for (int k = 0; k <= max_k; ++k) {
    const auto  t    = rice_encode_trace(values, k, max_bits);
    long double norm = 0, total = 0;
    for (const auto& w : t.words) {
      const auto weight = std::exp2(-static_cast<long double>(w.value) / half_life);
      norm += weight;
      total += weight * (w.end - w.begin);
    }
    out.push_back(
        {k,
         golomb_detail::rice_parameter(k),
         static_cast<double>(total / norm),
         static_cast<double>(norm),
         t.bits.size()});
  }
  return out;
}
} // namespace datamunge::algorithms
