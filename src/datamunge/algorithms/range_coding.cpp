/// @file range_coding.cpp
/// @brief One checked carry-free integer interval producer with actual byte-shift records.
#include <algorithm>
#include <datamunge/algorithms/range_coding.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical static range coding implementation.
namespace datamunge::algorithms {
/// @brief Private checked model and shared encode/decode register producer.
namespace {
/// @brief Actual stable top-byte threshold2^24.
constexpr std::uint32_t top = std::uint32_t{1} << 24;
/// @brief Actual minimum normalized range2^16.
constexpr std::uint32_t bottom = std::uint32_t{1} << 16;
/// @brief Validate all model entries and build the actual checked prefix table.
/// @param frequency Complete borrowed static frequency vector.
/// @param options Complete alphabet policy.
/// @return Owned prefix table ending in a positive total at most65536.
std::vector<std::uint32_t> model(
    const std::vector<std::uint32_t>& frequency,
    const RangeCodingOptions&         options) {
  if (frequency.empty())
    throw std::invalid_argument("range coding: empty model");
  if (frequency.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (options.max_alphabet && frequency.size() > options.max_alphabet))
    throw std::length_error("range coding: alphabet policy");
  std::vector<std::uint32_t> cumulative{0};
  cumulative.reserve(frequency.size() + 1);
  std::uint64_t total = 0;
  for (auto f : frequency) {
    total += f;
    if (total > bottom)
      throw std::invalid_argument("range coding: model total exceeds65536");
    cumulative.push_back(static_cast<std::uint32_t>(total));
  }
  if (!total)
    throw std::invalid_argument("range coding: zero total");
  return cumulative;
}
/// @brief Execute the shared actual unsigned interval producer.
/// @param input_symbols Complete encoder symbols, empty for decoder.
/// @param input_bytes Complete decoder bytes, empty for encoder.
/// @param frequency Static checked model.
/// @param count Actual symbols to process.
/// @param o Execution policies.
/// @param decoding Read input code bytes instead of emitting them.
/// @param capture Retain real subdivision and byte-shift records.
/// @return Owned actual coding outcome.
RangeCodingResult solve(
    const std::vector<int>&           input_symbols,
    const std::vector<std::uint8_t>&  input_bytes,
    const std::vector<std::uint32_t>& frequency,
    std::size_t                       count,
    const RangeCodingOptions&         o,
    bool                              decoding,
    bool                              capture) {
  RangeCodingResult result;
  result.cumulative = model(frequency, o);
  result.decoding   = decoding;
  const auto& cum   = result.cumulative;
  const auto  total = cum.back();
  if (o.max_symbols && count > o.max_symbols)
    throw std::length_error("range coding: symbol policy");
  if (decoding && o.max_bytes && input_bytes.size() > o.max_bytes)
    throw std::length_error("range coding: input byte policy");
  if (!decoding)
    for (int symbol : input_symbols)
      if (symbol < 0 || static_cast<std::size_t>(symbol) >= frequency.size() ||
          frequency[symbol] == 0)
        throw std::invalid_argument("range coding: invalid/zero-frequency symbol");
  if (decoding)
    result.bytes = input_bytes;
  else
    result.symbols = input_symbols;
  std::uint32_t low = 0, range = 0xffffffffU, code = 0;
  std::size_t   pos = 0, history = 0;
  /// @brief Admit one real symbol/byte record before its corresponding update.
  const auto admit = [&]() {
    if (!capture)
      return;
    if (history == std::numeric_limits<std::size_t>::max() ||
        (o.max_trace_entries && history >= o.max_trace_entries))
      throw std::length_error("range coding: history policy");
    ++history;
  };
  /// @brief Consume one real decoder byte with no implicit zero padding.
  /// @return Actual unsigned input byte.
  const auto read = [&]() {
    if (pos >= input_bytes.size())
      throw std::runtime_error("range coding: truncated byte stream");
    return static_cast<std::uint32_t>(input_bytes[pos++]);
  };
  /// @brief Emit one actual byte after output-policy admission.
  /// @param byte Actual top byte to append.
  const auto emit = [&](std::uint8_t byte) {
    if (o.max_bytes && result.bytes.size() >= o.max_bytes)
      throw std::length_error("range coding: output byte policy");
    result.bytes.push_back(byte);
    ++pos;
  };
  if (decoding) {
    for (int i = 0; i < 4; ++i)
      code = (code << 8) | read();
    result.symbols.reserve(count);
  }
  for (std::size_t index = 0; index < count; ++index) {
    admit();
    RangeSymbolStep step;
    step.low_before          = low;
    step.range_before        = range;
    step.code_before         = code;
    step.position_before     = pos;
    const std::uint32_t unit = range / total;
    if (!unit)
      throw std::runtime_error("range coding: zero subdivision");
    int symbol;
    if (decoding) {
      const std::uint32_t value = (code - low) / unit;
      if (value >= total)
        throw std::runtime_error("range coding: quotient outside model");
      symbol = static_cast<int>(std::upper_bound(cum.begin(), cum.end(), value) - cum.begin() - 1);
      if (symbol < 0 || static_cast<std::size_t>(symbol) >= frequency.size() ||
          frequency[symbol] == 0)
        throw std::runtime_error("range coding: invalid decoded symbol");
    } else
      symbol = input_symbols[index];
    step.symbol = symbol;
    step.unit   = unit;
    low += cum[symbol] * unit;
    range               = unit * frequency[symbol];
    step.low_narrowed   = low;
    step.range_narrowed = range;
    for (;;) {
      const auto before    = range;
      const bool stable    = (low ^ (low + range)) < top;
      bool       corrected = false;
      if (!stable) {
        if (range >= bottom)
          break;
        range     = (0U - low) & (bottom - 1);
        corrected = true;
      }
      admit();
      RangeByteStep byte_step;
      byte_step.low_before     = low;
      byte_step.range_before   = before;
      byte_step.adjusted_range = range;
      byte_step.corrected      = corrected;
      byte_step.code_before    = code;
      if (decoding) {
        const auto byte = read();
        byte_step.byte  = static_cast<std::uint8_t>(byte);
        code            = (code << 8) | byte;
      } else {
        byte_step.byte = static_cast<std::uint8_t>(low >> 24);
        emit(byte_step.byte);
      }
      low <<= 8;
      range <<= 8;
      byte_step.low_after   = low;
      byte_step.range_after = range;
      byte_step.code_after  = code;
      if (result.renormalizations == std::numeric_limits<std::size_t>::max())
        throw std::length_error("range coding: shift counter capacity");
      ++result.renormalizations;
      if (capture)
        step.shifts.push_back(byte_step);
    }
    step.low_after      = low;
    step.range_after    = range;
    step.code_after     = code;
    step.position_after = pos;
    if (decoding)
      result.symbols.push_back(symbol);
    if (capture)
      result.steps.push_back(std::move(step));
  }
  result.final_low   = low;
  result.final_range = range;
  result.final_code  = code;
  if (!decoding)
    for (int i = 0; i < 4; ++i) {
      emit(static_cast<std::uint8_t>(low >> 24));
      low <<= 8;
    }
  result.stream_position = pos;
  return result;
}
} // namespace
RangeCodingResult range_encode_run(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& frequency,
    const RangeCodingOptions&         o) {
  return solve(symbols, {}, frequency, symbols.size(), o, false, false);
}
RangeCodingResult range_encode_trace(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& frequency,
    const RangeCodingOptions&         o) {
  return solve(symbols, {}, frequency, symbols.size(), o, false, true);
}
RangeCodingResult range_decode_run(
    const std::vector<std::uint8_t>&  bytes,
    const std::vector<std::uint32_t>& frequency,
    std::size_t                       count,
    const RangeCodingOptions&         o) {
  return solve({}, bytes, frequency, count, o, true, false);
}
RangeCodingResult range_decode_trace(
    const std::vector<std::uint8_t>&  bytes,
    const std::vector<std::uint32_t>& frequency,
    std::size_t                       count,
    const RangeCodingOptions&         o) {
  return solve({}, bytes, frequency, count, o, true, true);
}
std::vector<std::uint8_t> range_encode(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& frequency) {
  return range_encode_run(symbols, frequency).bytes;
}
std::vector<int> range_decode(
    const std::vector<std::uint8_t>&  bytes,
    const std::vector<std::uint32_t>& frequency,
    int                               count) {
  if (count < 0)
    throw std::invalid_argument("range coding: negative symbol count");
  return range_decode_run(bytes, frequency, static_cast<std::size_t>(count)).symbols;
}
std::vector<int> range_frequency_pattern(
    const std::vector<std::uint32_t>& frequency,
    std::size_t                       repetitions,
    std::size_t                       max_symbols) {
  RangeCodingOptions o;
  const auto         prefix = model(frequency, o);
  const auto         total  = prefix.back();
  if (repetitions > std::numeric_limits<std::size_t>::max() / total ||
      (max_symbols && repetitions * total > max_symbols))
    throw std::length_error("range pattern: complete allocation policy");
  std::vector<int> out;
  out.reserve(repetitions * total);
  for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
    for (std::size_t symbol = 0; symbol < frequency.size(); ++symbol)
      for (std::uint32_t i = 0; i < frequency[symbol]; ++i)
        out.push_back(static_cast<int>(symbol));
  return out;
}
} // namespace datamunge::algorithms
