/// @file levenshtein_coding.cpp
/// @brief Shared actual full-range recursive emitter and checked exact-prefix parser.
#include <bit>
#include <datamunge/algorithms/levenshtein_coding.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked bit geometry and shared actual emission/parsing helpers.
namespace levcode_detail {
/// @brief Admit additional logical records before copying/appending actual capture.
/// @param trace Optional mutable complete capture.
/// @param amount Actual additional logical count.
/// @param cap Optional logical cap,0 off.
/// @throws std::length_error On actual count/cap overflow.
void admit(LevenshteinCodingTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  auto& n = trace->captured_entries;
  if (amount > std::numeric_limits<std::size_t>::max() - n ||
      (cap && (n > cap || amount > cap - n)))
    throw std::length_error("Levenshtein coding actual capture quota");
  n += amount;
}
/// @brief Emit one actual bit and count the selected prefix/payload write.
/// @param bits Mutable actual output stream.
/// @param bit Actual bit to append.
/// @param payload Whether this is an explicit binary payload bit.
/// @param work Mutable real write counts.
/// @param trace Optional actual capture.
/// @param cap Actual record cap.
/// @throws std::length_error On output/capture geometry.
/// @throws std::bad_alloc On bit-vector growth.
void write(
    std::vector<bool>&      bits,
    bool                    bit,
    bool                    payload,
    LevenshteinCodingWork&  work,
    LevenshteinCodingTrace* trace,
    std::size_t             cap) {
  if (bits.size() == bits.max_size())
    throw std::length_error("Levenshtein bit container");
  admit(trace, 1, cap);
  bits.push_back(bit);
  payload ? ++work.payload_bits : ++work.unary_bits;
  ++work.bit_visits;
}
/// @brief Emit a complete value sequence once, optionally retaining actual word/block records.
/// @param values Borrowed original unsigned64 sequence.
/// @param work Mutable actual write work.
/// @param trace Optional capture with original-value admission already completed.
/// @param cap Actual record cap.
/// @return Owned actual emitted stream.
/// @throws std::length_error On output/capture geometry.
/// @throws std::bad_alloc On runtime/output/capture allocations.
std::vector<bool> encode(
    const std::vector<std::uint64_t>& values,
    LevenshteinCodingWork&            work,
    LevenshteinCodingTrace*           trace,
    std::size_t                       cap) {
  std::vector<bool> bits;
  for (auto value : values) {
    LevenshteinCodeword word;
    word.value = value;
    word.begin = bits.size();
    std::vector<std::uint64_t> levels;
    if (value) {
      auto n = value;
      while (true) {
        levels.push_back(n);
        const auto width = static_cast<std::size_t>(std::bit_width(n) - 1);
        if (width == 0)
          break;
        n = width;
      }
    }
    word.unary_ones = levels.size();
    admit(trace, 1, cap);
    for (std::size_t i = 0; i < levels.size(); ++i)
      write(bits, true, false, work, trace, cap);
    write(bits, false, false, work, trace, cap);
    for (auto i = levels.size(); i-- > 0;) {
      const auto           n     = levels[i];
      const auto           width = static_cast<std::size_t>(std::bit_width(n) - 1);
      LevenshteinCodeBlock block{bits.size(), width, n};
      admit(trace, 1, cap);
      for (auto k = width; k-- > 0;)
        write(bits, (n >> k) & 1, true, work, trace, cap);
      if (trace)
        word.blocks.push_back(block);
    }
    word.end = bits.size();
    if (trace)
      trace->words.push_back(std::move(word));
    ++work.values;
  }
  return bits;
}
/// @brief Read one admitted actual input bit and count its prefix/payload visit.
/// @param bits Borrowed complete original input.
/// @param pos Mutable actual consumed offset.
/// @param payload Whether the read belongs to explicit binary payload.
/// @param work Mutable actual read counts.
/// @return Actual read bit.
/// @throws std::invalid_argument On missing requested bit.
bool read(
    const std::vector<bool>& bits,
    std::size_t&             pos,
    bool                     payload,
    LevenshteinCodingWork&   work) {
  if (pos >= bits.size())
    throw std::invalid_argument("truncated Levenshtein codeword");
  bool bit = bits[pos++];
  payload ? ++work.payload_bits : ++work.unary_bits;
  ++work.bit_visits;
  return bit;
}
/// @brief Parse exactly the requested prefix once with checked widths/extents before shifts/indexing.
/// @param bits Borrowed original complete bitstream.
/// @param count Nonnegative admitted requested word count.
/// @param work Mutable actual read work.
/// @param pos Mutable actual consumed offset, initially0.
/// @param trace Optional input-admitted complete capture.
/// @param cap Actual logical record cap.
/// @return Owned actual requested complete values.
/// @throws std::invalid_argument On malformed/truncated/unrepresentable requested prefix.
/// @throws std::length_error On output/capture geometry.
/// @throws std::bad_alloc On output/capture allocations.
std::vector<std::uint64_t> decode(
    const std::vector<bool>& bits,
    int                      count,
    LevenshteinCodingWork&   work,
    std::size_t&             pos,
    LevenshteinCodingTrace*  trace,
    std::size_t              cap) {
  if (count < 0 || static_cast<std::size_t>(count) > bits.size())
    throw std::invalid_argument("Levenshtein exact nonnegative word count");
  std::vector<std::uint64_t> values;
  values.reserve(count);
  for (int c = 0; c < count; ++c) {
    LevenshteinCodeword word;
    word.begin       = pos;
    std::size_t ones = 0;
    while (read(bits, pos, false, work)) {
      if (++ones > 5)
        throw std::invalid_argument("Levenshtein unary depth exceeds uint64 domain");
    }
    word.unary_ones = ones;
    std::uint64_t n = ones ? 1 : 0;
    admit(trace, 2, cap);
    if (ones) {
      admit(trace, 1, cap);
      if (trace)
        word.blocks.push_back({pos, 0, 1});
      for (std::size_t i = 1; i < ones; ++i) {
        if (n > 63 || n > bits.size() - pos)
          throw std::invalid_argument("Levenshtein block width/extent");
        const auto    width = static_cast<std::size_t>(n), begin = pos;
        std::uint64_t payload = 0;
        for (std::size_t j = 0; j < width; ++j)
          payload = (payload << 1) | read(bits, pos, true, work);
        n = (std::uint64_t{1} << width) | payload;
        admit(trace, 1, cap);
        if (trace)
          word.blocks.push_back({begin, width, n});
      }
    }
    word.value = n;
    word.end   = pos;
    values.push_back(n);
    if (trace)
      trace->words.push_back(std::move(word));
    ++work.values;
  }
  return values;
}
} // namespace levcode_detail
void levenshtein_encode_value(std::uint64_t n, std::vector<bool>& out) {
  LevenshteinCodingWork work;
  auto                  word = levcode_detail::encode({n}, work, nullptr, 0);
  if (word.size() > out.max_size() - out.size())
    throw std::length_error("Levenshtein append geometry");
  out.insert(out.end(), word.begin(), word.end());
}
std::vector<bool> levenshtein_encode(const std::vector<std::uint64_t>& values) {
  LevenshteinCodingWork work;
  return levcode_detail::encode(values, work, nullptr, 0);
}
std::vector<std::uint64_t> levenshtein_decode(const std::vector<bool>& bits, int count) {
  LevenshteinCodingWork work;
  std::size_t           pos = 0;
  return levcode_detail::decode(bits, count, work, pos, nullptr, 0);
}
LevenshteinCodingTrace levenshtein_encode_trace(
    const std::vector<std::uint64_t>& values,
    std::size_t                       max_entries) {
  LevenshteinCodingTrace out;
  levcode_detail::admit(&out, values.size(), max_entries);
  out.values        = values;
  out.bits          = levcode_detail::encode(values, out.work, &out, max_entries);
  out.consumed_bits = out.bits.size();
  return out;
}
LevenshteinCodingTrace
levenshtein_decode_trace(const std::vector<bool>& bits, int count, std::size_t max_entries) {
  if (count < 0)
    throw std::invalid_argument("Levenshtein negative count");
  LevenshteinCodingTrace out;
  out.decoding = true;
  levcode_detail::admit(&out, bits.size(), max_entries);
  out.bits   = bits;
  out.values = levcode_detail::decode(bits, count, out.work, out.consumed_bits, &out, max_entries);
  return out;
}
} // namespace datamunge::algorithms
