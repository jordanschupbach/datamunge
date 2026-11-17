/// @file salsa20.cpp
/// @brief One specified Salsa20/20 ARX core and checked actual stream/block producers.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/salsa20.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical Salsa20 conformance implementation.
namespace datamunge::algorithms {
/// @brief Private little-endian state and actual ARX execution.
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Decode one actual four-byte little-endian word.
/// @param p Borrowed pointer to four available bytes.
/// @return Actual uint32 word.
std::uint32_t load(const std::uint8_t* p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) |
         (std::uint32_t(p[3]) << 24);
}
/// @brief Store one actual word in four little-endian bytes.
/// @param p Borrowed writable four-byte span.
/// @param word Actual word.
void store(std::uint8_t* p, std::uint32_t word) {
  for (int i = 0; i < 4; ++i)
    p[i] = static_cast<std::uint8_t>(word >> (8 * i));
}
/// @brief Execute exactly one canonical core with optional actual quarter capture.
/// @param in Borrowed64-byte input.
/// @param capture Retain real word/quarter states.
/// @param cap Complete optional logical-entry quota.
/// @return Owned actual output and optional word histories.
Salsa20CoreTrace core(const std::array<std::uint8_t, 64>& in, bool capture, std::size_t cap) {
  if (capture && cap && cap < 768)
    throw std::length_error("Salsa20: core capture policy");
  Salsa20CoreTrace              out;
  std::array<std::uint32_t, 16> initial{}, words{};
  for (std::size_t i = 0; i < 16; ++i)
    words[i] = initial[i] = load(in.data() + 4 * i);
  if (capture) {
    out.initial = initial;
    out.quarters.reserve(80);
  }
  const std::array<std::array<std::size_t, 4>, 4> columns{
      {{0, 4, 8, 12}, {5, 9, 13, 1}, {10, 14, 2, 6}, {15, 3, 7, 11}}},
      rows{{{0, 1, 2, 3}, {5, 6, 7, 4}, {10, 11, 8, 9}, {15, 12, 13, 14}}};
  for (std::size_t round = 0; round < 20; ++round)
    for (std::size_t q = 0; q < 4; ++q) {
      const auto     ids = (round % 2 ? rows : columns)[q];
      Salsa20Quarter event;
      event.round   = round;
      event.quarter = q;
      event.indices = ids;
      if (capture)
        for (std::size_t k = 0; k < 4; ++k)
          event.before[k] = words[ids[k]];
      auto& a = words[ids[0]];
      auto& b = words[ids[1]];
      auto& c = words[ids[2]];
      auto& d = words[ids[3]];
      b ^= std::rotl(std::uint32_t(a + d), 7);
      c ^= std::rotl(std::uint32_t(b + a), 9);
      d ^= std::rotl(std::uint32_t(c + b), 13);
      a ^= std::rotl(std::uint32_t(d + c), 18);
      if (capture) {
        for (std::size_t k = 0; k < 4; ++k)
          event.after[k] = words[ids[k]];
        out.quarters.push_back(event);
      }
    }
  if (capture)
    out.mixed = words;
  for (std::size_t i = 0; i < 16; ++i) {
    const std::uint32_t result = words[i] + initial[i];
    if (capture)
      out.output_words[i] = result;
    store(out.bytes.data() + 4 * i, result);
  }
  return out;
}
/// @brief Construct the actual specified key/nonce/counter input layout.
/// @param key Borrowed key bytes.
/// @param counter Actual block id.
/// @param nonce Borrowed nonce bytes.
/// @return Owned64 input bytes.
std::array<std::uint8_t, 64> layout(
    const std::array<std::uint8_t, 32>& key,
    std::uint64_t                       counter,
    const std::array<std::uint8_t, 8>&  nonce) {
  std::array<std::uint32_t, 16> words{};
  words[0]  = 0x61707865u;
  words[5]  = 0x3320646eu;
  words[10] = 0x79622d32u;
  words[15] = 0x6b206574u;
  for (std::size_t i = 0; i < 4; ++i) {
    words[1 + i]  = load(key.data() + 4 * i);
    words[11 + i] = load(key.data() + 16 + 4 * i);
  }
  words[6] = load(nonce.data());
  words[7] = load(nonce.data() + 4);
  words[8] = static_cast<std::uint32_t>(counter);
  words[9] = static_cast<std::uint32_t>(counter >> 32);
  std::array<std::uint8_t, 64> bytes{};
  for (std::size_t i = 0; i < 16; ++i)
    store(bytes.data() + 4 * i, words[i]);
  return bytes;
}
/// @brief Check the exact final required counter before any core call.
/// @param counter First counter.
/// @param blocks Actual requested block count.
void counters(std::uint64_t counter, std::size_t blocks) {
  if (blocks && blocks - 1 > std::numeric_limits<std::uint64_t>::max() - counter)
    throw std::overflow_error("Salsa20: counter range");
}
/// @brief Execute one complete checked XOR stream and capture only actual used blocks.
/// @param key Borrowed key.
/// @param counter First block id.
/// @param nonce Borrowed nonce.
/// @param data Borrowed complete raw message.
/// @param o Policies.
/// @param capture Retain actual block key streams.
/// @return Owned XOR result and selected work.
Salsa20StreamRun stream(
    const std::array<std::uint8_t, 32>& key,
    std::uint64_t                       counter,
    const std::array<std::uint8_t, 8>&  nonce,
    const std::string&                  data,
    const Salsa20Options&               o,
    bool                                capture) {
  const auto blocks = data.size() / 64 + (data.size() % 64 != 0),
             max    = std::numeric_limits<std::size_t>::max();
  if ((o.max_bytes && data.size() > o.max_bytes) || blocks > max / 65 ||
      (capture && o.max_trace_entries && blocks * 65 > o.max_trace_entries))
    throw std::length_error("Salsa20: complete stream policy");
  counters(counter, blocks);
  Salsa20StreamRun out;
  out.output = data;
  if (capture)
    out.blocks.reserve(blocks);
  std::size_t offset = 0;
  while (offset < data.size()) {
    const auto          count = std::min<std::size_t>(64, data.size() - offset);
    const std::uint64_t id    = counter + out.block_calls;
    const auto          bytes = core(layout(key, id, nonce), false, 0).bytes;
    ++out.block_calls;
    for (std::size_t j = 0; j < count; ++j) {
      out.output[offset + j] =
          static_cast<char>(static_cast<std::uint8_t>(out.output[offset + j]) ^ bytes[j]);
      ++out.byte_xors;
    }
    if (capture)
      out.blocks.push_back({id, offset, count, bytes});
    offset += count;
  }
  return out;
}
} // namespace
std::array<std::uint8_t, 64> salsa20_core(const std::array<std::uint8_t, 64>& in) {
  return core(in, false, 0).bytes;
}
Salsa20CoreTrace salsa20_core_trace(const std::array<std::uint8_t, 64>& in, std::size_t cap) {
  return core(in, true, cap);
}
std::array<std::uint8_t, 64> salsa20_block(
    const std::array<std::uint8_t, 32>& key,
    std::uint64_t                       counter,
    const std::array<std::uint8_t, 8>&  nonce) {
  return core(layout(key, counter, nonce), false, 0).bytes;
}
Salsa20StreamRun salsa20_stream_run(
    const std::array<std::uint8_t, 32>& k,
    std::uint64_t                       c,
    const std::array<std::uint8_t, 8>&  n,
    const std::string&                  d,
    const Salsa20Options&               o) {
  return stream(k, c, n, d, o, false);
}
Salsa20StreamRun salsa20_stream_trace(
    const std::array<std::uint8_t, 32>& k,
    std::uint64_t                       c,
    const std::array<std::uint8_t, 8>&  n,
    const std::string&                  d,
    const Salsa20Options&               o) {
  return stream(k, c, n, d, o, true);
}
std::string salsa20_encrypt(
    const std::array<std::uint8_t, 32>& k,
    std::uint64_t                       c,
    const std::array<std::uint8_t, 8>&  n,
    const std::string&                  d) {
  return stream(k, c, n, d, {}, false).output;
}
Salsa20ByteStudy salsa20_byte_study(
    const std::array<std::uint8_t, 32>& key,
    const std::array<std::uint8_t, 8>&  nonce,
    std::uint64_t                       counter,
    std::size_t                         blocks,
    std::size_t                         max_bytes) {
  if (blocks > std::numeric_limits<std::size_t>::max() / 64 ||
      (max_bytes && blocks * 64 > max_bytes))
    throw std::length_error("Salsa20: complete study bytes");
  counters(counter, blocks);
  Salsa20ByteStudy out;
  for (std::size_t i = 0; i < blocks; ++i) {
    const auto bytes = salsa20_block(key, counter + i, nonce);
    ++out.blocks;
    for (auto byte : bytes) {
      ++out.counts[byte];
      ++out.groups[byte >> 4];
      ++out.bytes;
    }
  }
  return out;
}
Salsa20Example salsa20_example() {
  Salsa20Example out;
  out.input    = {211, 159, 13,  115, 76, 55,  82,  183, 3,   117, 222, 37,  191, 187, 234, 136,
                  49,  237, 179, 48,  1,  106, 178, 219, 175, 199, 166, 48,  86,  16,  179, 207,
                  31,  240, 32,  63,  15, 83,  93,  161, 116, 147, 48,  113, 238, 55,  204, 36,
                  79,  201, 235, 79,  3,  81,  156, 47,  203, 26,  244, 243, 88,  118, 104, 54};
  out.expected = {109, 42,  178, 168, 156, 240, 248, 238, 168, 196, 190, 203, 26,  110, 170, 154,
                  29,  29,  150, 26,  150, 30,  235, 249, 190, 163, 251, 48,  69,  144, 51,  57,
                  118, 40,  152, 157, 180, 57,  27,  94,  107, 42,  236, 35,  27,  111, 114, 114,
                  219, 236, 232, 135, 111, 155, 110, 18,  24,  232, 95,  158, 179, 19,  48,  202};
  for (std::size_t i = 0; i < 32; ++i)
    out.key[i] = static_cast<std::uint8_t>(i * 7 + 1);
  out.nonce     = {1, 2, 3, 4, 5, 6, 7, 8};
  out.plaintext = "Salsa20 stream cipher round-trip over multiple 64-byte blocks!!";
  return out;
}
} // namespace datamunge::algorithms
